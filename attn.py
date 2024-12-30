import json
import torch 
import einops

from torch import einsum 
from torch import nn
from jaxtyping import Array, Float, jaxtyped
from torch import Tensor
from einops import rearrange 
from typeguard import typechecked as typechecker

@jaxtyped(typechecker=typechecker)
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                dim_d: int = 512, 
                num_heads: int = 8, causal_mask: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.dim_d = dim_d
        self.dim_head = dim_d // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_d, dim_d)
        self.w_k = nn.Linear(dim_d, dim_d)
        self.w_v = nn.Linear(dim_d, dim_d)
        self.w_out = nn.Linear(dim_d, dim_d)
        self.causal_mask = causal_mask

    def forward(self, 
                X: Float[Tensor, "btz seq_len dim_d"], # batch, sequence length, dimension # type: ignore
                temperature: float = 1.0
            ) -> Float[Tensor, "btz seq_len dim_d"]: # type: ignore

        if self.causal_mask:
            mask = torch.tril(torch.ones(X.size(1), X.size(1)), diagonal = 0).to(X.device)
        Xq = rearrange(self.w_q(X), 'b t (h d) -> b h t d', h = self.num_heads)
        Xk = rearrange(self.w_k(X), 'b t (h d) -> b h t d', h = self.num_heads)
        Xv = rearrange(self.w_v(X), 'b t (h d) -> b h t d', h = self.num_heads)

        dot_prod = einsum('b h q d, b h t d -> b h q t', Xq, Xk)
        if self.causal_mask:
            dot_prod = dot_prod.masked_fill(mask == 0, float('-inf'))
        attn = nn.functional.softmax(dot_prod * temperature / (self.dim_head ** 0.5), dim = -1)
        out = einsum('b h q t, b h t d -> b h q d', attn, Xv)
        out = rearrange(out, 'b h q d -> b q (h d)')
        
        return self.w_out(out)
