########################################################################
# This contains implementation of ViT + TokenMixup                     #
# Code modified from https://github.com/jeonsworld/ViT-pytorch         #
# Copyright MLV Lab @ Korea University                                 #
########################################################################
import logging

import torch
from timm.data import create_loader, create_dataset, Mixup, FastCollateMixup, AugMixDataset

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root=args.data_dir,
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir,
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None

    train_sampler = RandomSampler(trainset) if not args.distributed else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader


def get_imagenet_loader(args):

    dataset_train = create_dataset(
        args.dataset,
        download=True,
        root=args.data_dir, split="train", is_training=True,
        batch_size=args.train_batch_size, repeats=0)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split='validation', is_training=False, batch_size=args.eval_batch_size)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = 'bicubic'
    loader_train = create_loader(
        dataset_train,
        input_size=[3, 224, 224],
        batch_size=args.train_batch_size,
        is_training=True,
        use_prefetcher=not args.no_prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=0,
        interpolation=train_interpolation,
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=None,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=[3, 224, 224],
        batch_size=args.eval_batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation='bicubic',
        mean=[0.485, 0.456, 0.406],
        std=[0.229,0.224,0.225],
        num_workers=args.workers,
        distributed=False,
        crop_pct=0.0,
        pin_memory=args.pin_mem,
    )

    return loader_train, loader_eval