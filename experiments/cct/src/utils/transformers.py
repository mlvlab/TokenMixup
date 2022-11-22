###############################################################################
# This contains implementation of CCT + TokenMixup                            #
# Code modified from https://github.com/SHI-Labs/Compact-Transformers         #
# Copyright MLV Lab @ Korea University                                        #
###############################################################################
import random

import torch
from timm.utils import AverageMeter
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath
from .utils import horizontalMixup, horizontalMixupCLS, Attention_Vertical


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_out = attn.data
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_out


class MaskedAttention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, l_idx, seq_pool, num_classes,
                 dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1,
                 **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)


        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.activation = F.gelu

        self.seq_pool = seq_pool
        self.num_classes = num_classes
        self.use_htm = kwargs["horizontal_mixup"]
        self.horizontal_layer = kwargs["horizontal_layer"]
        self.rho = kwargs["rho"]
        self.tau = kwargs["tau"]
        self.scorenet_stopgrad = kwargs["scorenet_stopgrad"] 
        self.scorenet_train = kwargs["scorenet_train"]
        self.use_vtm = kwargs["vertical_mixup"]
        self.vertical_layer = kwargs["vertical_layer"] 
        self.VTM_stopgrad = kwargs["vertical_stopgrad"]
        self.kappa = kwargs["kappa"]

        if l_idx == self.vertical_layer and self.use_vtm:
            self.self_attn = Attention_Vertical(dim=d_model, num_heads=nhead,
                                                attention_dropout=attention_dropout, projection_dropout=dropout)
        else:
            self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                       attention_dropout=attention_dropout, projection_dropout=dropout)

        if l_idx == self.horizontal_layer and self.scorenet_train:
            self.small_h_attention_pool = Linear(d_model, 1) # scorenet pooling layer
            self.small_h_fc = Linear(d_model, num_classes) # scorenet classifier layer

        # counters
        self.K1, self.K2 = 0, 0

    def HTM(self, src, scorenet_src, y):
        mix_score = torch.sum(-y * F.log_softmax(scorenet_src, dim=-1), dim=-1).data / 10.0
        mix_logit = mix_score <= self.tau  # score↓ : mixup↑(True), score↑ : mixup↓,  Sample Easy Score : Loss

        self.K1, self.K2 = mix_logit.sum().item(), 0
        attn = self.self_attn(self.pre_norm(src.data))[1]
        Asum = attn.mean(1)[:, 0, 1:] if not self.seq_pool else attn.mean(dim=[1, 2])

        if self.K1 != 0:
            if self.seq_pool:
                src, y, self.K2 = horizontalMixup(src, y, Asum, rho=self.rho, mix_logit=mix_logit)
            else:
                src, y, self.K2 = horizontalMixupCLS(src, y, Asum, rho=self.rho, mix_logit=mix_logit)

        return src, y, attn



    def forward(self, src: torch.Tensor, y, scorenet_h_src, vTokens, *args, **kwargs) -> torch.Tensor:
        attn, vToken = None, None

        # Horizontal TokenMixup
        if (self.use_htm and kwargs["lidx"] == self.horizontal_layer and self.scorenet_train) and self.training:
            scorenet_h_src = src.detach() if self.scorenet_stopgrad else src
            scorenet_h_src = torch.matmul(F.softmax(self.small_h_attention_pool(scorenet_h_src), dim=1).transpose(-1, -2), scorenet_h_src).squeeze(-2) \
                            if self.seq_pool else scorenet_h_src[:, 0]
            scorenet_h_src = self.small_h_fc(scorenet_h_src)  # (128(B), 100(Num_Class))
            if self.use_htm:
                src, y, attn = self.HTM(src, scorenet_h_src.data, y)
        src_ = src

        # Vertical TokenMixup
        if (self.use_vtm and kwargs["lidx"] == self.vertical_layer) and self.training:
            vToken_all = torch.stack(vTokens)
            L, B, T, D = vToken_all.shape # L, B, T, D
            vToken_all = vToken_all.transpose(0, 1).reshape(B, -1, D) # (B, L*T, D)
            vToken_all = vToken_all.detach() if self.VTM_stopgrad else vToken_all

            x, attn_ = self.self_attn(self.pre_norm(src), kv = self.pre_norm(torch.cat([src, vToken_all], dim=1)))  # (1, 256, 256)
        else:
            x, attn_ = self.self_attn(self.pre_norm(src))  # (1, 256, 256)

        src = src + self.drop_path(x)
        if (self.use_vtm and kwargs["lidx"] < self.vertical_layer) and self.training:
            attn = attn_.data if attn is None else attn
            if self.training:
                Asum = attn.mean(1)[:, 0, 1:] if not self.seq_pool else attn.mean(dim=[1, 2])
                
                _, top_idx = torch.topk(Asum, k=self.kappa, dim=1, largest=True)
                vToken = torch.gather(src_, 1, top_idx.unsqueeze(-1).expand(-1,-1, src_.shape[-1]))

        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src, scorenet_h_src, y, attn_, vToken


class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead,
                                         attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0


        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    l_idx=i, seq_pool=seq_pool, num_classes=num_classes,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i],
                                    **kwargs)
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)


    def forward(self, x, y):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        attns, scorenet_h_src, vTokens = [], None, []

        for l, blk in enumerate(self.blocks):
            x, scorenet_h_src, y, attn, vToken = blk(x, y, scorenet_h_src=scorenet_h_src, lidx=l, vTokens= vTokens)
            if attn != None: 
                attns.append(attn)
            if vToken != None: 
                vTokens.append(vToken)

        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)

        if self.training:
            return x, y, scorenet_h_src, attns, sum([b.K1 for b in self.blocks]), sum([b.K2 for b in self.blocks])
        else:
            return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MaskedTransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='sine',
                 seq_len=None,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert seq_len is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len,
                                                                          embedding_dim,
                                                                          padding_idx=True),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            MaskedTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                          dim_feedforward=dim_feedforward, dropout=dropout,
                                          attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask=None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
                mask = (mask > 0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim, padding_idx=False):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
        return pe
