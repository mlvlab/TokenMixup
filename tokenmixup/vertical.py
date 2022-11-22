import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class VTM_ATTN(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, kv=None):
        B, N, C = x.shape
        if kv is None :
            q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = self.key(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = self.value(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else :
            Bkv, Nkv, Ckv = kv.shape
            q = self.query(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   
            k = self.key(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  
            v = self.value(kv).reshape(B, Nkv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_out = attn.data[:,:,:,:attn.shape[2]].data

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_out


class VerticalTokenMixupLayer(nn.Module):

    def __init__(self, attention,
                       layer_index,
                       apply_layers, 
                       kappa,
                       vtm_attn_dim,
                       vtm_attn_numheads,
                       vtm_stopgrad=False,
                       has_cls=True,
                       report_every=50,
                       verbose=True):
        super().__init__()
        """
        Vertical TokenMixup wrapper module that wraps a self attention module

        :attention: the self attention module VTM needs
        :layer_index: current layer index number
        :apply_layers: list of encoder block indices Vertical TokenMixup is applied to
        :kappa: number of tokens to gather from each layer
        :vtm_attn_dim: dimension of model used for self attention and VTM cross attention
        :vtm_attn_numheads: number of heads for VTM cross attention
        :vtm_stopgrad: if True, no gradient for vtm tokens; if False, gradient flows in vtm tokens
        :has_cls: if True, input token tensor contains a CLS token
        :report_every: train step interval of statistics report
        :verbose: if True, will report statistics; if False, no report will be shown
        """ 

        # configs
        assert type(apply_layers) is list, "apply_layers should be a list of integers"
        assert 0 not in apply_layers, "VTM cannot be applied to the first layer (layer index 0)"
        self.apply_layers = apply_layers
        self.layer_index = layer_index
        self.kappa = kappa
        self.vtm_stopgrad = vtm_stopgrad
        self.has_cls = has_cls
        self.report_every = report_every
        self.verbose = verbose

        self.step_count = 0
        self.warnings_count = 0

        # module
        if self.layer_index in self.apply_layers :
            self.attention = VTM_ATTN(dim=vtm_attn_dim, num_heads=vtm_attn_numheads)
        else :
            self.attention = attention

        # TODO setting global is not the best thing to do; can be improved
        self.reset()


    def register_memory(self, src, saliency):
        global __VTM_MEMORY__
        saliency = saliency.data
        B, H, Qdim, Kdim = saliency.shape

        saliency = saliency.mean(1)[:, 0, 1:] if self.has_cls else saliency.mean(dim=[1, 2])
        _, top_idx = torch.topk(saliency, k=self.kappa, dim=1, largest=True)
        salient_tokens = torch.gather(src, 1, top_idx.unsqueeze(-1).expand(-1,-1, src.shape[-1]))
        
        if str(self.layer_index) in list(__VTM_MEMORY__.keys()) and self.warnings_count < 8:
            print("WARNING: Overwriting previous layer vertical tokens. This can be due to duplicate calls of the attention module in a single encoder block.")
            self.warnings_count += 1
        __VTM_MEMORY__[str(self.layer_index)] = salient_tokens


    def get_memory_tokens(self):
        return [token for token in __VTM_MEMORY__.values()]

    def reset(self):
        global __VTM_MEMORY__
        __VTM_MEMORY__ = {}


    def report_status(self, vtm_token_num, device):
        if self.verbose :
            if self.step_count % self.report_every == 1 :
                print("\n----------------------------------VTM Stats-----------------------------------")
                print("  Layer : "+str(self.layer_index+1)+"\tDevice : "+str(device))
                print(f"     [kappa = {self.kappa}]")
                print(f"     {vtm_token_num} Multi-scale Tokens were Cross-Attended!")
                print("------------------------------------------------------------------------------")


    def forward(self, src, *args, **kwargs):
        if self.training :
            if self.layer_index in self.apply_layers :
                self.step_count += 1
                vtm_memories = self.get_memory_tokens()
                vtm_tokens = torch.stack(vtm_memories)
                L, B, T, D = vtm_tokens.shape
                vtm_tokens = vtm_tokens.transpose(0, 1).reshape(B, -1, D) # (B, L*T, D)
                vtm_tokens = vtm_tokens.detach() if self.vtm_stopgrad else vtm_tokens

                # NOTE slightly different from paper experiment version
                #    Paper : prenorm( concat( src, vtm_tokens ) )
                #    Here  : concat( prenorm(src), prenorm(vtm_tokens) )
                # You may need to modify the code depending on the location of normalization layer
                out = self.attention(src, kv=torch.cat([src, vtm_tokens], dim=1))
                x, attn = out
                self.report_status(vtm_tokens.shape[1], x.device)
                if self.layer_index < max(self.apply_layers):
                    self.register_memory(src, attn)
                return out 
            else :
                out = self.attention(src, *args, **kwargs)
                assert (type(out) in (tuple, list)) and (len(out) == 2), "attention module must return (output, attention map)"
                x, attn = out
                if self.layer_index < max(self.apply_layers):
                    self.register_memory(src, attn)
                return out

        else :
            if self.layer_index in self.apply_layers :
                return self.attention(src, kv=src)
            else :
                return self.attention(src, *args, **kwargs)
