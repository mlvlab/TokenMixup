import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from scipy.optimize import linear_sum_assignment
import numpy as np

class HorizontalTokenMixupLayer(nn.Module):
    def __init__(self, layer, 
                       tau, 
                       rho, 
                       d_model, 
                       num_classes,
                       use_scorenet=True,
                       scorenet_lambda=0.001,
                       scorenet_type='global_average_pooling',
                       scorenet_stopgrad=False,
                       aspect_ratio=1.0,
                       batch_first=True,
                       has_cls=True,
                       cls_first=True,
                       report_every=50,
                       layer_name=None,
                       verbose=True):
        super().__init__()
        """
        (Horizontal) TokenMixup wrapper module that wraps an encoder layer

        :layer: encoder block HorzizontalTokenMixupLayer wraps
        :tau: sample difficulty threshold
        :rho: saliency difference threshold
        :d_model: feature dimension of input tokens
        :num_classes: number of classes of the dataset
        :use_scorenet: if True, apply based on difficulty; if False, always apply HTM
        :scorenet_type: how scorenet aggregates token information
        :scorenet_stopgrad: stopgrad before token aggregation for scorenet
        :aspect_ratio: the aspect ratio (= W / H) of tokens when spatial structure is restored. \
                      Is used to resize cases with different Query & Key numbers.
        :batch_first: if True, input dim is (B, N, D); if False, input dim is (N, D, B)
        :has_cls: if True, input token tensor contains a CLS token.
        :cls_first: if True, the first token is the CLS; if False, the last token is the CLS
        :report_every: train step interval of statistics report
        :layer_name: name of the layer HTM wraps; will appear in the report
        :verbose: if True, will report statistics; if False, no report will be shown
        """ 

        # configs
        self.tau = tau
        self.rho = rho
        self.scorenet_lambda = scorenet_lambda
        self.aspect_ratio = aspect_ratio

        self.d_model = d_model
        self.batch_first = batch_first
        self.has_cls = has_cls
        self.cls_first = cls_first
        self.report_every = report_every
        self.verbose = verbose

        self.score_scale_factor = 0.1
        self.step_count = 0


        # encoder block
        self.layer_name = layer_name
        self.layer = layer
        
        # scorenet
        assert scorenet_type in ('average_pooling', 'attention_pooling', 'cls_pooling')
        self.use_scorenet = use_scorenet
        if self.use_scorenet :
            self.scorenet_type = scorenet_type
            self.scorenet_stopgrad = scorenet_stopgrad
            if self.scorenet_type == 'average_pooling' :
                self.scorenet_fc = nn.Linear(d_model, num_classes)
            elif self.scorenet_type == 'cls_pooling' :
                assert self.has_cls, "can't use CLS pooling without cls token; use average_pooling or attention_pooling instead"
                self.scorenet_fc = nn.Linear(d_model, num_classes)
            elif self.scorenet_type == 'attention_pooling' :
                self.scorenet_attn = nn.Linear(d_model, 1)
                self.scorenet_fc = nn.Linear(d_model, num_classes)



    def report_status(self, device):
        if self.verbose :
            if self.step_count % self.report_every == 1 :
                print("\n----------------------------------HTM Stats-----------------------------------")
                if self.layer_name is not None :
                    print("  Layer : "+str(self.layer_name)+"\tDevice : "+str(device))
                print("     Sample Difficulty Score min (avg) \t[tau = {tau:.4f}] : {min:.4f} ({avg:.4f})".format(
                        tau=self.tau, min=self.difficulty_score.min().item(), avg=self.difficulty_score.mean().item()))
                print("\t ==> Mixed Sample Count per Batch : {}".format(self.mix_sample_num))                   
                print("     Saliency Difference max (avg) \t[rho = {rho:.4f}] : {max:.4f} ({avg:.4f})".format(
                        rho=self.rho, max=self.s_diff_max, avg=self.s_diff_avg))
                print("\t ==> Avg. Mixed Token Count per Sample : {}".format(int(self.mix_token_num)))
                print("------------------------------------------------------------------------------")



    def forward(self, x, y=None, *args, **kwargs):
        
        if not self.training :
            return self.layer(x, y, *args, **kwargs)
        else :
            self.step_count += 1

        assert hasattr(self.layer, 'get_attention_map') and callable(getattr(self.layer, 'get_attention_map')), \
                "please define function get_attention_map in the encoder block \n \
                 that takes the same input as the encoder block and returns the self attention map"
        
        assert y is not None, "please pass in the target label into your encoder forward function"

        if self.batch_first :
            B, N, D = x.shape 
        else :
            N, D, B = x.shape

        assert D == self.d_model, "please check the token feature dimension"
        
        if isinstance(y, torch.Tensor) :
            prev_scorenet_cumloss = 0
        elif isinstance(y, tuple):
            y, prev_scorenet_cumloss = y
        else :
            raise "check input y data type; should be one of torch.Tensor or tuple"

        # retrieve the attention map
        attn = self.layer.get_attention_map(x.data, y, *args, **kwargs).data
        assert attn.shape[0] == B, "attention map batch size error; attention dim should be (batch, head, query, key)"
        _, attn_heads, attn_qdim, attn_kdim = attn.shape
        
        if attn_qdim != attn_kdim : # resize saliency map if query and key dimension differs
            qh = int((attn_qdim / self.aspect_ratio)**0.5)
            qw = int(attn_qdim / qh)
            assert qh * qw == attn_qdim, "wrong aspect_ratio; please recheck"
            kh = int((attn_kdim / self.aspect_ratio)**0.5)
            kw = int(attn_kdim / kh)
            assert kh * kw == attn_kdim, "wrong aspect_ratio; please recheck"
            _attn = attn.reshape(B, attn_heads, attn_qdim, kh, kw)
            _attn = F.interpolate(_attn, (attn_qdim, qh, qw))
            attn = _attn.reshape(B, attn_heads, attn_qdim, attn_qdim)
            

        # align order to (Batch, Token, Feature) 
        if not self.batch_first :
            x = x.permute(2,0,1)

        # align order of Token so that CLS token comes first
        if not self.cls_first :
            x = torch.cat([x[:,-1:], x[:,:-1]], 1)


        # 1. Sample Difficulty Assessment
        self.difficulty_score = torch.zeros(B, dtype=float)
        if self.use_scorenet :
            scorenet_x = x.detach() if self.scorenet_stopgrad else x
            if self.has_cls and self.scorenet_type != 'cls_pooling':
                scorenet_x = scorenet_x[:,1:]

            if self.scorenet_type == 'average_pooling' :
                scorenet_x = scorenet_x.mean(1)
            elif self.scorenet_type == 'cls_pooling' :
                scorenet_x = scorenet_x[:,0]
            elif self.scorenet_type == 'attention_pooling' :
                a = F.softmax(self.scorenet_attn(scorenet_x), dim=1).transpose(-1, -2)
                scorenet_x = torch.matmul(a, scorenet_x).squeeze(-2)
            scorenet_pred = self.scorenet_fc(scorenet_x)
            scorenet_loss = torch.sum(-y * F.log_softmax(scorenet_pred, dim=-1), dim=-1)

            difficulty_score = scorenet_loss.data * self.score_scale_factor
            apply_tokenmixup = difficulty_score <= self.tau
            self.difficulty_score = difficulty_score.cpu()
        else :
            apply_tokenmixup = torch.ones(B, dtype=bool)

        mix_sample_num = apply_tokenmixup.sum().item()
        self.mix_sample_num, self.mix_token_num = mix_sample_num, 0
        self.s_diff_max, self.s_diff_avg = 0.0, 0.0
        if mix_sample_num == 0 :
            # if no samples are mixed, wrap up and continue on
            self.report_status(x.device)
            return self.layer(x, (y, prev_scorenet_cumloss + self.scorenet_lambda * scorenet_loss), *args, **kwargs)

        # 3. Attention-based Saliency Detection & Optimal Assignment
        saliency = attn.mean(1)[:, 0, 1:] if self.has_cls else attn.mean(dim=[1, 2])
        x_to_mix, x_no_mix = x[apply_tokenmixup], x[~apply_tokenmixup]
        y_to_mix, y_no_mix = y[apply_tokenmixup], y[~apply_tokenmixup]
        saliency_to_mix = saliency[apply_tokenmixup] 

        raw_saliency_diff = saliency.repeat(mix_sample_num,1,1) - saliency_to_mix.reshape(mix_sample_num,1,N).repeat(1,B,1)
        tmp = torch.clamp(raw_saliency_diff, 0.0).cpu()
        self.s_diff_max, self.s_diff_avg = tmp.max().item(), tmp.mean().item()
        saliency_diff = torch.clamp(raw_saliency_diff - self.rho, 0.0)
        saliency_diff_agg = saliency_diff.sum(dim=2).cpu()
        saliency_index = torch.tensor(linear_sum_assignment(-saliency_diff_agg)[1], device=saliency.device)
        saliency_pair_to_mix = saliency[saliency_index]

        M  = torch.gather(saliency_diff, 1, saliency_index.unsqueeze(dim=-1).repeat(1, 1, N)).squeeze() > 0
        
        avg_mix_token_num = M.sum() / mix_sample_num
        self.mix_token_num = avg_mix_token_num.item()
        
        # 4. Token-level Mixup
        # actual mixup for x
        if self.has_cls :
            x_to_mix_cls, x_to_mix_noncls = x_to_mix[:, :1], x_to_mix[:, 1:]
            x_mixed = (~M).unsqueeze(-1).repeat(1, 1, D) * x_to_mix_noncls + M.unsqueeze(-1).repeat(1, 1, D) * x[saliency_index, 1:]
            x_mixed = torch.cat([torch.cat([x_to_mix_cls, x_mixed], dim=1), x_no_mix], dim=0)
        else :
            x_mixed = (~M).unsqueeze(-1).repeat(1, 1, D) * x_to_mix + M.unsqueeze(-1).repeat(1, 1, D) * x[saliency_index]
            x_mixed = torch.cat([x_mixed, x_no_mix], dim=0)

        # actual mixup for y
        i = (saliency_to_mix * (~M)).sum(1)
        j = (saliency_pair_to_mix * M).sum(1)
        ratio_to_mix = (i / (i + j)).unsqueeze(-1)
        ratio_pair_to_mix = (j / (i + j)).unsqueeze(-1)

        y_mixed = ratio_to_mix * y_to_mix + ratio_pair_to_mix * y[saliency_index]
        y_mixed = torch.cat([y_mixed, y_no_mix], dim=0)



        # propagate original encoder layer
        if not self.cls_first :
            x_mixed = torch.cat([x_mixed[:,1:], x_mixed[:,:1]], 1)

        if not self.batch_first :
            x_mixed = x_mixed.permute(1,2,0)

        self.report_status(x_mixed.device)
        return self.layer(x_mixed, (y_mixed, prev_scorenet_cumloss + self.scorenet_lambda * scorenet_loss), *args, **kwargs)