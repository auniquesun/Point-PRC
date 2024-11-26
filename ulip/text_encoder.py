from collections import OrderedDict

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                i=0, design_details=None):
        super().__init__()

        # NOTE in nn.MultiheadAttention, `batch_first=False` by default，so the input dimension is (seq_len, batch, dim_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        if i > 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                self.n_text_ctx = design_details["language_ctx"]  # hyperparameter
                self.text_ctx = nn.Parameter(torch.empty(self.n_text_ctx, d_model))
                nn.init.normal_(self.text_ctx, std=0.02)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # x: (seq_len, batch, d_model)

        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Appending the learnable tokens in different way
            # x -> [77, NCLS, DIM]
            # First remove the learnable tokens from previous layer
            prefix = x[:1, :, :]
            suffix = x[1 + self.n_text_ctx:, :, :]
            # Create/configure learnable tokens of this layer
            textual_context = self.text_ctx.expand(x.shape[1], -1, -1).permute(1, 0, 2)
            # Add the learnable tokens of this layer with the input, replaced by previous
            # layer learnable tokens
            x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TextTransformer(nn.Module):
    ''' Adapt this part to SLIP-style transformer '''
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            # add learnable prompts in first `text_depth` layers, `add_prompt=True`; otherwise, `add_prompt=Fasle``
            resblocks = []
            for i in range(layers):
                if i < prompts_needed:
                    resblocks.append(ResidualAttentionBlock_IVLP(width, heads, attn_mask, True, i, design_details))
                else:
                    resblocks.append(ResidualAttentionBlock_IVLP(width, heads, attn_mask, False, i, design_details))
            self.resblocks = nn.Sequential(*resblocks)
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)