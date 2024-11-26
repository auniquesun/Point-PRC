import numpy as np

import torch
from torch import nn

from ulip.pointbert.point_encoder import PointTransformer
from ulip.text_encoder import LayerNorm, TextTransformer


class ULIP(nn.Module):
    def __init__(self, cfg, design_details):
        super().__init__()

        # NOTE PointEncoder <= configs/trainers/PointPRC/custom_ulip.yaml
        pc_feat_dim = 2*cfg.PointEncoder.trans_dim # 2*384
        embed_dim = cfg.TextEncoder.embed_dim

        self.point_encoder = PointTransformer(cfg.PointEncoder, design_details)
        self.pc_projection = nn.Parameter(torch.empty(pc_feat_dim, embed_dim))
        self.prompt_depth_pc = design_details['point_depth']        # 9

        trans_width = cfg.TextEncoder.trans_width
        trans_layers = cfg.TextEncoder.trans_layers
        trans_heads = cfg.TextEncoder.trans_heads
        self.prompt_depth_text = design_details['language_depth']   # 9

        vocab_size = cfg.TextEncoder.vocab_size
        self.ctx_len = cfg.TextEncoder.ctx_len
        self.token_embedding = nn.Embedding(vocab_size, trans_width).cuda()
        self.positional_embedding = nn.Parameter(torch.empty(self.ctx_len, trans_width))
        self.ln_final = LayerNorm(trans_width)

        self.transformer = TextTransformer(
            width=trans_width,
            layers=trans_layers,
            heads=trans_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=self.prompt_depth_text,
            design_details=design_details
        )

        self.text_projection = nn.Parameter(torch.empty(trans_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        ''' NOTE initializing learnable parameters is enough, since other parameters
            are loaded into ULIP later. 
        '''
        # --- 1. initialize `point_encoder` params ---
        if self.prompt_depth_pc > 0:
            nn.init.normal_(self.point_encoder.point_ctx, std=0.02)
            nn.init.normal_(self.point_encoder.point_ctx_pos, std=0.02)

        for layer_idx, block in enumerate(self.point_encoder.blocks.blocks):
            if self.prompt_depth_pc > layer_idx and layer_idx > 0:
                nn.init.normal_(block.point_ctx, std=0.02)

        nn.init.normal_(self.pc_projection, std=512 ** -0.5)

        # --- 2. initialize `text_encoder` params ---
        for layer_idx, block in enumerate(self.transformer.resblocks):
            if self.prompt_depth_text > layer_idx and layer_idx > 0:
                nn.init.normal_(block.text_ctx, std=0.02)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.logit_scale, std=0.02)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.ctx_len, self.ctx_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_pc(self, pts):
        pc_feat = self.point_encoder(pts)
        pc_feat = pc_feat @ self.pc_projection

        return pc_feat

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.expand(x.shape[0], -1, -1)
        # NOTE in nn.MultiHeadAttention, `batch_first=False` by default, so the input dimension is (n_ctx, batch, d_model)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, pc, text):
        pc_features = self.encode_pc(pc)
        text_features = self.encode_text(text)

        # normalized features
        pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_pc = logit_scale * pc_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ pc_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_pc, logits_per_text


def build_model(cfg, design_details):
    ''' Define text/point encode & load their pre-trained weights '''

    model = ULIP(cfg, design_details)

    # 0. select `ulip1` or `ulip2`
    ckpts = f'ulip/pretrained_models/pointbert_{cfg.MODEL.ULIP_VERSION}.pt'

    # 1. load the pretrained pointbert model
    pretrain_point_model = torch.load(ckpts, map_location=torch.device('cpu'))
    pretrain_point_model_params = pretrain_point_model['state_dict']
    pretrain_point_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                    pretrain_point_model_params.items()}
    
    # 2. load the pretrained slip model
    pretrain_slip_model = torch.load('ulip/pretrained_models/slip_base_100ep.pt', map_location=torch.device('cpu'))
    pretrain_slip_model_params = pretrain_slip_model['state_dict']
    pretrain_slip_model_params = {param_name.replace('module.', ''): param for param_name, param in
                                    pretrain_slip_model_params.items()}
    
    # 3. load the pretrained pointbert/slip weights into `model`
    print("Turning off gradients in both the point and the text encoder")
    for name, param in model.named_parameters():
        if name not in pretrain_point_model_params and name not in pretrain_slip_model_params:
            # print(f'>>>>>> learnable params: {name}')
            continue

        if name in pretrain_point_model_params:
            if isinstance(pretrain_point_model_params[name], nn.Parameter):
                param_new = pretrain_point_model_params[name].data
            else:
                param_new = pretrain_point_model_params[name]
        else:
            if isinstance(pretrain_slip_model_params[name], nn.Parameter):
                param_new = pretrain_slip_model_params[name].data
            else:
                param_new = pretrain_slip_model_params[name]
                
        param.requires_grad = False
        # print('load {} and freeze'.format(name))
        param.data.copy_(param_new)

    return model