########################################################################
# This contains implementation of ViT + TokenMixup                     #
# Code modified from https://github.com/jeonsworld/ViT-pytorch         #
# Copyright MLV Lab @ Korea University                                 #
########################################################################
import argparse
import ast

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif s.lower() in ('none', 'None'):
        return None
    else:
        v = ast.literal_eval(s)
        if type(v) is not list:
            raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
        return v


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)

def horizontalMixup(x, y, Asum, rho=0.015, mix_logit=None, mix_cls=False):
    x_mix_I_Y, x_mix_N = x[mix_logit], x[~mix_logit]
    y_mix_Y, y_mix_N = y[mix_logit], y[~mix_logit]
    Asum_i = Asum[mix_logit]
    B, _, D = x.shape
    B_M, T = Asum_i.shape

    # (B_M, B, T)
    saliency_map = torch.clamp(Asum.repeat(B_M, 1, 1) - Asum_i.reshape(B_M, 1, T).repeat(1, B, 1) - rho, 0.0)
    
    # Hungarian Matching
    saliency_index = torch.tensor(linear_sum_assignment(-saliency_map.sum(dim=2).detach().cpu())[1])

    # compute mask
    saliency_mask  = torch.gather(saliency_map, 1, saliency_index.unsqueeze(dim=-1).repeat(1, 1, T).cuda()).squeeze() > 0
    Asum_j = Asum[saliency_index]

    U = ~saliency_mask    
    V =  saliency_mask    

    # ----------------------- A. Re-Labeling -----------------------
    score_i = (Asum_i * U).sum(dim=1)
    score_j = (Asum_j * V).sum(dim=1)
    score_x_i = (score_i / (score_i + score_j)).unsqueeze(-1)
    score_x_j = (score_j / (score_i + score_j)).unsqueeze(-1)

    y_mix = score_x_i * y_mix_Y + score_x_j * y[saliency_index]
    y = torch.cat([y_mix, y_mix_N], dim=0)

    # ----------------------- B.2. CLS Token -----------------------
    x_mix_I_Y_cls, x_mix_I_Y_other = x_mix_I_Y[:, :1], x_mix_I_Y[:, 1:]
    x_mix_cls = score_x_i * x_mix_I_Y_cls + score_x_j * x[saliency_index, :1] if mix_cls else x_mix_I_Y_cls

    # ----------------------- B.3 OTHER Tokens -----------------------
    x_mix_other = x_mix_I_Y_other * U.unsqueeze(-1).repeat(1, 1, D) + x[saliency_index, 1:] * V.unsqueeze(-1).repeat(1, 1, D)
    x = torch.cat([torch.cat([x_mix_cls, x_mix_other], dim=1), x_mix_N], dim=0)

    return x, y, saliency_mask.sum() / B_M


