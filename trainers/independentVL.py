import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import ulip
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_ulip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.PointEncoder.NAME

    if 'pointbert' in backbone_name.lower():
        print(f'Using {backbone_name} as Point Encoder ...')
    else:
        raise NotImplementedError(f'Point Encoder: {backbone_name} not supported!')

    # 1. prompted model 
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "point_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_POINT,    # 9
                          "language_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT,    # 9
                          "point_ctx": cfg.TRAINER.IVLP.N_CTX_POINT,             # 4
                          "language_ctx": cfg.TRAINER.IVLP.N_CTX_TEXT}             # 5
        model = ulip.build_model(cfg, design_details)
    # 2. frozen model
    else:
        # Return original ULIP model for generating frozen point/text features
        design_details = {"trainer": 'IVLP', "point_depth": 0, "language_depth": 0, 
                          "point_ctx": 0, "language_ctx": 0}
        model = ulip.build_model(cfg, design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, ulip_model):
        super().__init__()
        self.transformer = ulip_model.transformer
        self.positional_embedding = ulip_model.positional_embedding
        self.ln_final = ulip_model.ln_final
        self.text_projection = ulip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, ulip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_text_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        n_point_ctx = cfg.TRAINER.IVLP.N_CTX_POINT
        txt_ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        txt_ctx_dim = cfg.TextEncoder.trans_width
        
        if txt_ctx_init and n_text_ctx <= 4:
            # use given words to initialize context vectors
            txt_ctx_init = txt_ctx_init.replace("_", " ")
            # prompt: (ctx_len,)
            prompt = clip.tokenize(txt_ctx_init).cuda()
            with torch.no_grad():
                # (ctx_len, trans_dim)
                embedding = ulip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + n_text_ctx, :]
            prompt_prefix = txt_ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_text_ctx, txt_ctx_dim)
            prompt_prefix = " ".join(["X"] * n_text_ctx)

        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_text_ctx}")
        print(f"Number of context words (tokens) for Point prompting: {n_point_ctx}")

        self.text_ctx = nn.Parameter(ctx_vectors)
        nn.init.normal_(self.text_ctx, std=0.02)

        split_clsnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in split_clsnames]
        prompts = [prompt_prefix + " " + name + "." for name in split_clsnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, ctx_len=77)
        # embedding: (n_cls, ctx_len, trans_dim)
        embedding = ulip_model.token_embedding(tokenized_prompts.cuda())

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_text_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_text_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.text_ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomULIP(nn.Module):
    def __init__(self, cfg, classnames, ulip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, ulip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.point_encoder = ulip_model.point_encoder
        self.pc_projection = ulip_model.pc_projection
        self.text_encoder = TextEncoder(ulip_model)
        self.logit_scale = ulip_model.logit_scale

    def forward(self, pts, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        point_features = self.point_encoder(pts)
        point_features = point_features @ self.pc_projection
        point_features = point_features / point_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * point_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class IVLP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading ULIP (backbone: {cfg.PointEncoder.NAME})")
        ulip_model = load_ulip_to_cpu(cfg)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            # CLIP's default precision is fp16
            ulip_model.float()

        print("Building custom ULIP")
        self.model = CustomULIP(cfg, classnames, ulip_model)

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('\n', '='*30, '\n >>> Learnable Params:', params, '\n', '='*30, '\n')

        # Double check
        enabled_names = list()
        enabled_params = list()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled_names.append(name)
                enabled_params.append({'params': param})
        print(f"Parameters to be updated: {enabled_names}")
        print(f"Parameters count: {len(enabled_names)}")

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        pts, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IVLP.PREC
        if prec == "amp":
            with autocast():
                loss = model(pts, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(pts, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)