import copy
import json
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import clip, ulip
from clip.clip import _tokenizer
from .templates import MULTI_TEMPLATES, SINGLE_TEMPLATE


def load_ulip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.PointEncoder.NAME

    if 'pointbert' in backbone_name.lower():
        print(f'Using {backbone_name} as Point Encoder ...')
    else:
        raise NotImplementedError(f'Point Encoder: {backbone_name} not supported!')

    # --- prompted model ---
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "point_depth": cfg.TRAINER.PointPRC.PROMPT_DEPTH_POINT,    # 9
                          "language_depth": cfg.TRAINER.PointPRC.PROMPT_DEPTH_TEXT,    # 9
                          "point_ctx": cfg.TRAINER.PointPRC.N_CTX_POINT,             # 4
                          "language_ctx": cfg.TRAINER.PointPRC.N_CTX_TEXT}             # 5
    # --- frozen model ---
    else:
        # Return original ULIP model for generating frozen point/text features
        design_details = {"trainer": 'IVLP', "point_depth": 0, "language_depth": 0, 
                          "point_ctx": 0, "language_ctx": 0}

    model = ulip.build_model(cfg, design_details)
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
        # tokenized_prompts: [batch, n_token], here `n_token` is set to 77 in clip
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, ulip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PointPRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_text_ctx = cfg.TRAINER.PointPRC.N_CTX_TEXT
        n_point_ctx = cfg.TRAINER.PointPRC.N_CTX_POINT
        txt_ctx_init = cfg.TRAINER.PointPRC.CTX_INIT
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

        # fixed_embeddings: (n_cls, dim_emb)
        self.fixed_embeddings = self.frozen_text_emb(cfg, classnames, split_clsnames)

    @torch.no_grad()
    def frozen_text_emb(self, cfg, classnames, split_clsnames):
        # create frozen ULIP
        frozen_ulip = load_ulip_to_cpu(cfg, zero_shot_model=True).cuda()
        self.frozen_point_encoder = frozen_ulip.point_encoder

        prompt_type = cfg.TRAINER.PointPRC.PROMPT_TYPE
        all_teacher_features = []

        # Using multiple text templates to ensure textual diversity during training
        print(f"*********** prompt_type: {prompt_type} ***********")
        if prompt_type == "manual64":
            if cfg.TRAINER.PointPRC.NO_TDC:
                TEMPLATE = SINGLE_TEMPLATE
            else:
                TEMPLATE = MULTI_TEMPLATES

            for single_template in TEMPLATE:
                # `x` is a list of length `num_classes`
                x = [single_template.replace("{}", name) for name in split_clsnames]
                # x_tokenized: (n_cls, n_tkn)
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                # text_features: (n_cls, dim_emb)
                text_features = frozen_ulip.encode_text(x_tokenized.cuda())
                # --- text_features.unsqueeze(1): (n_cls, 1, dim_emb)
                all_teacher_features.append(text_features.unsqueeze(1))
            fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        else:   # gpt35, gpt4, pointllm
            dataset = cfg.DATASET.NAME
            if "modelnet40" in dataset.lower():
                dset = "mn40"
            elif "scanobjectnn" in dataset.lower():
                dset = "sonn"
            elif "shapenetcorev2" in dataset.lower():
                dset = "snv2"
            elif "pointda_modelnet" in dataset.lower():
                dset = "da_mn10"
            elif "pointda_shapenet" in dataset.lower():
                dset = "da_sn10"
            elif "pointda_scannet" in dataset.lower():
                dset = "da_scan10"
            elif "sim2real_mn11" in dataset.lower():
                dset = "mn11"
            elif "sim2real_sn9" in dataset.lower():
                dset = "sn9"
            elif "sim2real_sonn" in dataset.lower():
                dset = "sonn"
            else:
                raise ValueError(f"No corresponding prompts file in `llm/` for {dataset} dataset!")
            prompts_file = f'llm/{dset}_{prompt_type}_prompts.json'
            
            with open(prompts_file) as fin:
                descriptions_from_llm = json.load(fin)

            for name in classnames:
                cat_prompts = descriptions_from_llm[name]
                li = []
                for p in cat_prompts:
                    if len(p.split()) < 50:
                        li.append(clip.tokenize(p))
                    else:
                        words = p.split()
                        dash_cnt = " ".join(words[:50]).count('-')
                        li.append(clip.tokenize(" ".join(words[:50-dash_cnt])))
                # cp_tokenized: (n_des, n_tkn), here `n_des` means number of descriptions for each class
                cp_tokenized = torch.cat(li)
                # cp_features: (n_des, dim_emb)
                cp_features = frozen_ulip.encode_text(cp_tokenized.cuda())
                # cp_feature.unsqueeze(0): (1, n_des, dim_emb)
                all_teacher_features.append(cp_features.unsqueeze(0))
            # fixed_embeddings: (n_cls, dim_emb)
            fixed_embeddings = torch.cat(all_teacher_features, dim=0).mean(dim=1)
        return fixed_embeddings

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
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.no_mac = cfg.TRAINER.PointPRC.NO_MAC
        self.n_cls = len(classnames)

    def forward(self, pts, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted pts and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        prompt_txt_feats = text_features / text_features.norm(dim=-1, keepdim=True)

        # It's neccessary to `project` point_features to desired dimension
        point_features = self.point_encoder(pts)
        point_features = point_features @ self.pc_projection
        prompt_point_feats = point_features / point_features.norm(dim=-1, keepdim=True)
        
        # Compute the prompted logits
        prompt_logits = logit_scale * prompt_point_feats @ prompt_txt_feats.t()

        if self.prompt_learner.training:
            if self.no_mac:
                return F.cross_entropy(prompt_logits,label)
            else:
                # Now calculate the frozen pre-trained features
                # --- fixed_embeddings: (n_cls, dim_emb) ---
                fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
                frozen_txt_feats = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
                with torch.no_grad():
                    # --- zero_shot_features: (batch, dim_emb) ---
                    zero_shot_features = self.prompt_learner.frozen_point_encoder(pts)
                    zero_shot_features = zero_shot_features @ self.pc_projection
                    frozen_point_feats = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                    # Compute pre-trained frozen visual features
                    # --- frozen_logits: (batch, n_cls) ---
                    frozen_logits = logit_scale * frozen_point_feats.cuda() @ frozen_txt_feats.cuda().t()

                # 按照我改过后的变量命名，很容易看出每个返回值的含义
                return F.cross_entropy(prompt_logits,label), frozen_txt_feats, prompt_txt_feats, frozen_point_feats, \
                    prompt_point_feats, frozen_logits, prompt_logits
        else:
            return prompt_logits
            

@TRAINER_REGISTRY.register()
class PointPRC(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PointPRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading ULIP (3D Backbone: {cfg.PointEncoder.NAME})")
        ulip_model = load_ulip_to_cpu(cfg)

        if cfg.TRAINER.PointPRC.PREC == "fp32" or cfg.TRAINER.PointPRC.PREC == "amp":
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
            # load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # `register_model` is a member of `TrainerX`, which inherits SimpleTrainer
        self.register_model("pointprc", self.model, self.optim, self.sched)

        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PointPRC.MEC_MEAN   # 15
        stdev = cfg.TRAINER.PointPRC.MEC_STD   # 1
        gauss = self.get_gauss(mean, stdev)     # `lambda` expression
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])  # self.gauss: np.array([num_1, num_2, ..., num_N])
        self.gauss = self.gauss / sum(self.gauss)   # normalization
        self.previous_model_mec = None

        self.scaler = GradScaler() if cfg.TRAINER.PointPRC.PREC == "amp" else None
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

        prec = self.cfg.TRAINER.PointPRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(pts, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            if self.cfg.TRAINER.PointPRC.NO_MAC:
                loss = model(pts, label)
            else:
                def get_loss():
                    loss_ce, frozen_txt_feats, prompt_txt_feats, frozen_point_feats, prompt_point_feats, \
                    frozen_logits, prompt_logits = model(pts, label)
                    # Calculate the L_MAC_text loss
                    loss_mac_text = F.l1_loss(prompt_txt_feats, frozen_txt_feats.cuda(),
                                            reduction='mean') * self.cfg.TRAINER.PointPRC.TEXT_LOSS_WEIGHT
                    # Calculate the L_MAC_point loss
                    loss_mac_point = F.l1_loss(prompt_point_feats, frozen_point_feats.cuda(),
                                            reduction='mean') * self.cfg.TRAINER.PointPRC.POINT_LOSS_WEIGHT
                    
                    # Compute `kl_div` with torch.nn.functional
                    L_MAC_logits = F.kl_div(
                        F.log_softmax(prompt_logits / 1, dim=1),
                        F.log_softmax(frozen_logits / 1, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * (1 * 1) / prompt_logits.numel()
                    L_MAC = (L_MAC_logits + loss_mac_text + loss_mac_point)

                    loss = (loss_ce + L_MAC)
                    return loss, loss_mac_text, loss_mac_text, L_MAC_logits

                def closure():
                    loss, _, _, _ = get_loss()
                    loss.backward()
                    return loss
            
                loss, loss_mac_text, loss_mac_point, L_MAC_logits = get_loss()
                loss.backward()
                optim.step(closure)
                optim.zero_grad()

        if self.cfg.TRAINER.PointPRC.NO_MAC:
            loss_summary = {"loss": loss.item()}
        else:
            loss_summary = {"loss": loss.item(), "loss_mac_text": loss_mac_text.item(), 
                        "loss_mac_point": loss_mac_point.item(), "loss_mac_logits": L_MAC_logits.item()}

        # MEC weighting
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            if not self.cfg.TRAINER.PointPRC.NO_MEC:
                # NOTE Once a epoch is completed, perform MEC - model ensemble constraint
                # `self.step_counter` is set to 1 initially
                self.step_counter = self.step_counter + 1
                current_epoch_weight = self.gauss[self.step_counter - 2]
                current_model_weights = copy.deepcopy(model.state_dict())
                weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
                if self.previous_model_mec is None:
                    self.previous_model_mec = weighted_state_dict
                else:
                    self.previous_model_mec = self.state_dict_add(weighted_state_dict, self.previous_model_mec)

                if self.step_counter == self.model.total_epochs + 1:
                    print("Using MEC model for final inference...")
                    model.load_state_dict(self.previous_model_mec)
                    self.model.load_state_dict(self.previous_model_mec)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["pc"]
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