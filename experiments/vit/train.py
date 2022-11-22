########################################################################
# This contains implementation of ViT + TokenMixup                     #
# Code modified from https://github.com/jeonsworld/ViT-pytorch         #
# Copyright MLV Lab @ Korea University                                 #
########################################################################
# coding=utf-8
from __future__ import absolute_import, division, print_function
import torch.nn.functional as F

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist
from timm.data.mixup import mixup_target
from timm.scheduler import create_scheduler
from timm.utils import update_summary, NativeScaler
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp

from torch.nn.parallel import DistributedDataParallel as NativeDDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, get_imagenet_loader
from utils.dist_util import get_world_size
from utils.utils import str2bool, AverageMeter

logger = logging.getLogger(__name__)



def top1_accuracy(preds, labels):
    return (preds == labels).mean()


def top5_accuracy(logits, labels):
    return np.sum(np.argsort(logits[0],-1)[:,-5:] == np.expand_dims(labels, -1).repeat(5,-1), -1).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    if args.dataset == "imagenet":
        args.num_classes = 1000
    else:
        args.num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, 
                              zero_head=False, 
                              num_classes=args.num_classes,
                              horizontal_mixup=args.horizontal_mixup,
                              horizontal_layer=args.horizontal_layer,
                              rho=args.rho,
                              tau=args.tau,
                              scorenet_stopgrad=args.scorenet_stopgrad,
                              scorenet_train=args.scorenet_train,
                              vertical_mixup=args.vertical_mixup,
                              vertical_layer=args.vertical_layer,
                              kappa=args.kappa,
                              vertical_stopgrad=args.vertical_stopgrad)
    if args.eval :
        model.load_state_dict(torch.load(args.pretrained_dir, map_location='cpu'), strict=False)
    elif args.pretrained_dir is not None :
        model.load_from(np.load(args.pretrained_dir))
    model.cuda()
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label, all_logits = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_logits[0] = np.append(
                all_logits[0], logits.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    top1_acc = top1_accuracy(all_preds, all_label)
    top5_acc = top5_accuracy(all_logits, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid TOP1 Accuracy: %2.5f" % top1_acc)
    logger.info("Valid TOP5 Accuracy: %2.5f" % top5_acc)

    writer.add_scalar("test/top1_accuracy", scalar_value=top1_acc, global_step=global_step)
    writer.add_scalar("test/top5_accuracy", scalar_value=top5_acc, global_step=global_step)

    return top1_acc, top5_acc


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    if args.dataset == "imagenet":
        train_loader, test_loader = get_imagenet_loader(args)
    else:
        train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    loss_fn = CrossEntropyLoss()
    scorenet_loss_fn = CrossEntropyLoss()

    t_total = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, min_lr=args.min_lr, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, min_lr=args.min_lr, t_total=t_total)

    if args.distributed:
        model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        if args.local_rank == 0:
            logger.info('Using native Torch AMP. Training in mixed precision.')


    if args.eval :
        valid(args, model, writer, test_loader, 0)
        return 0

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    if args.distributed :
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    global_step, best_top1, best_top5, epoch = 0, 0, 0, 0

    losses_m        = AverageMeter()
    scorenet_loss_m = AverageMeter()

    while True:
        model.train()
        e_step = len(train_loader)
        epoch_iterator = tqdm(train_loader,
                              desc="Training E : (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])


        for step, batch in enumerate(epoch_iterator):

            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            y = mixup_target(y, num_classes=args.num_classes, smoothing=args.smoothing, device=y.device)
            output, label, scorenet_output, attns = model(x, y)
           
            loss = loss_fn(output, label)
            scorenet_loss = scorenet_loss_fn(scorenet_output, y) if scorenet_output != None else 0
            loss_all = loss + args.scorenet_lambda * scorenet_loss

            if args.gradient_accumulation_steps > 1:
                loss_all = loss_all / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss_all, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_all.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses_m.update(loss.item() * args.gradient_accumulation_steps)
                if scorenet_output != None: 
                    scorenet_loss_m.update(scorenet_loss.item() * args.gradient_accumulation_steps)

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step(epoch)

                torch.cuda.synchronize()

                epoch_iterator.set_description(
                    "Training %d/%d (%d / %d Steps) (loss=%2.5f)" % (epoch, args.epochs, step, e_step, losses_m.val)
                )

                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses_m.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)


        if args.local_rank in [-1, 0]:
            top1, top5 = valid(args, model, writer, test_loader, global_step)
            if best_top1 < top1:
                save_model(args, model)
                best_top1 = top1
                best_top5 = top5
            epoch += 1
            
        losses_m.reset(), scorenet_loss_m.reset()


    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best TOP1 Accuracy: \t%f" % best_top1)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default='test',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="weights/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output/", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-3, type=float,
                        help="The initial learning rate for SGD.")

    parser.add_argument("--min_lr", default=0.001, type=float, help="")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=300, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--eval', action='store_true')


    ######## TokenMixup args
    # HTM
    parser.add_argument("--horizontal-mixup", type=str2bool, default=False)
    parser.add_argument("--horizontal-layer", type=int, default=-1, 
                        help="horizontal mixup layer")
    parser.add_argument('--tau', type=float, default=0.0, help="difficulty threshold")
    parser.add_argument('--rho', type=float, default=0.0, help="saliency difference threshold")
    parser.add_argument('--scorenet-stopgrad', type=str2bool, default=True)
    parser.add_argument('--scorenet-train', type=str2bool, default=False)
    parser.add_argument('--scorenet-lambda', type=float, default=1.)

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument("--warmup_epochs", default=0, type=int,
                        help="Step of training to perform learning rate warmup for.")

    # VTM
    parser.add_argument('--vertical-mixup', type=str2bool, default=False)
    parser.add_argument('--vertical-stopgrad', type=str2bool, default=False)
    parser.add_argument("--vertical-layer", type=int, default=-1, help="vertical mixup layer")
    parser.add_argument('--kappa', type=int, default=0, help="number of tokens to select in VTM")

    # Training Configurations
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument('--run_name', default='test', type=str)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--smoothing', type=float, default=0.)
    parser.add_argument('--distributed', action='store_true', default=False)

    # Imagenet configurations
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const',
                        help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--train-interpolation', type=str, default='random')

    args = parser.parse_args()

    from datetime import datetime
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.run_name,
    ])
    args.output_dir = args.output_dir + exp_name

    if args.debug: os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)


    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
