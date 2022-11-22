###############################################################################
# This contains implementation of CCT + TokenMixup                            #
# Code modified from https://github.com/SHI-Labs/Compact-Transformers         #
# Copyright MLV Lab @ Korea University                                        #
###############################################################################
import argparse
import ast

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


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




def to_np(d):
    return d.detach().cpu().numpy()


def horizontalMixup(x, y, Asum, mix_logit=None, rho = 0.01):

    x_mix_I_Y, x_mix_N = x[mix_logit], x[~mix_logit]
    y_mix_Y, y_mix_N = y[mix_logit], y[~mix_logit]
    Asum_i  = Asum[mix_logit]
    B, T, D = x.shape
    B_M, _  = Asum_i.shape

    # (B_M, B, T)
    saliency_map   = torch.clamp(Asum.repeat(B_M, 1, 1) - Asum_i.reshape(B_M, 1, T).repeat(1, B, 1) - rho, 0.0)
                        
    saliency_shift = saliency_map.sum(dim=2).cpu()
    saliency_index = torch.tensor(linear_sum_assignment(-saliency_shift)[1], device=saliency_map.device)
    saliency_mask  = torch.gather(saliency_map, 1, saliency_index.unsqueeze(dim=-1).repeat(1, 1, T)).squeeze() > 0
    Asum_j = Asum[saliency_index]

    U = ~saliency_mask 
    V =  saliency_mask 

    x_mix = x_mix_I_Y * U.unsqueeze(-1).repeat(1,1, D) + x[saliency_index] * V.unsqueeze(-1).repeat(1,1, D)
    x     = torch.cat([x_mix, x_mix_N], dim=0)


    score_i = (Asum_i * U).sum(dim=1)
    score_j = (Asum_j * V).sum(dim=1)
    y_mix = (score_i / (score_i + score_j)).unsqueeze(-1) * y_mix_Y + (score_j / (score_i + score_j)).unsqueeze(-1) * y[saliency_index]
    y = torch.cat([y_mix, y_mix_N], dim=0)

    return x, y, saliency_mask.sum() / B_M




def horizontalMixupCLS(x, y, Asum, mix_logit=None, rho=0.01, cls_mixup=False):
    x_mix_I_Y, x_mix_N = x[mix_logit], x[~mix_logit]
    y_mix_Y, y_mix_N = y[mix_logit], y[~mix_logit]
    Asum_i = Asum[mix_logit]
    B, _, D = x.shape
    B_M, T = Asum_i.shape

    saliency_map   = torch.clamp(Asum.repeat(B_M, 1, 1) - Asum_i.reshape(B_M, 1, T).repeat(1, B, 1) - rho, 0.0)
        # swap based on saliency difference (J - I) 

    saliency_index = torch.tensor(linear_sum_assignment(-saliency_map.sum(dim=2).detach().cpu())[1])
    saliency_mask  = torch.gather(saliency_map, 1, saliency_index.unsqueeze(dim=-1).repeat(1, 1, T).cuda()).squeeze() > 0
    Asum_j = Asum[saliency_index]

    U = ~saliency_mask    
    V =  saliency_mask    

    # ----------------------- A. Scoring -----------------------
    score_i = (Asum_i * U).sum(dim=1)
    score_j = (Asum_j * V).sum(dim=1)
    score_x_i = (score_i / (score_i + score_j)).unsqueeze(-1)
    score_x_j = (score_j / (score_i + score_j)).unsqueeze(-1)

    y_mix = score_x_i * y_mix_Y + score_x_j * y[saliency_index]
    y = torch.cat([y_mix, y_mix_N], dim=0)


    # ----------------------- B.2. CLS Token -----------------------
    x_mix_I_Y_cls, x_mix_I_Y_other = x_mix_I_Y[:, :1], x_mix_I_Y[:, 1:]
    x_mix_cls = score_x_i * x_mix_I_Y_cls + score_x_j * x[saliency_index, :1] if cls_mixup else x_mix_I_Y_cls

    # ----------------------- B.3 OTHER Token -----------------------

    x_mix_other = x_mix_I_Y_other * U.unsqueeze(-1).repeat(1, 1, D) + x[saliency_index, 1:] * V.unsqueeze(-1).repeat(1, 1, D)
    x = torch.cat([torch.cat([x_mix_cls, x_mix_other], dim=1), x_mix_N], dim=0)

    return x, y, saliency_mask.sum() / B_M


from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init


class Attention_Vertical(nn.Module):

    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.query = Linear(dim, dim, bias=False)
        self.key = Linear(dim, dim, bias=False)
        self.value = Linear(dim, dim, bias=False)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, kv=None):
        B, N, C = x.shape
        if kv is None :
            q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.key(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.value(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else :
            Bkv, Nkv, Ckv = kv.shape
            q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    # (128(B), 4(H), 256(T), 64(D))
            k = self.key(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # (128(B), 4(H), 256(T)+3(AT), 64(D))
            v = self.value(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (128(B), 4(H), 256(T)+3(AT), 64(D))


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_out = attn.data[:,:,:,:attn.shape[2]].data

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_out



class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16


    def forward(self, output, target, smoothing=0.1):
        target = (target > smoothing * 1/output.shape[-1]).float()

        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target != 0).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)
        loss = pos_loss + neg_loss
        return loss.mean()



def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    mixed_y = lam * y + (1 - lam) * y[index,:]
    return mixed_x, mixed_y


