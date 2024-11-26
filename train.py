import os
import time
import shutil
import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# --- cocoop tyle
import datasets.modelnet40
import datasets.scanobjectnn
import datasets.shapenetcorev2
import datasets.modelnet40_c
import datasets.modelnet_c
import datasets.omniobject3d
import datasets.objaverse_lvis
# --- sim2real
import datasets.sim2real_mn11
import datasets.sim2real_sn9
import datasets.sim2real_sonn
# --- pointda
import datasets.pointda_modelnet
import datasets.pointda_shapenet
import datasets.pointda_scannet

import trainers.maple
import trainers.independentVL

import trainers.pointprc
# NOTE for ablation purpose
# import trainers.pointprc_uniform
# import trainers.pointprc_mse
# import trainers.pointprc_cosine


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.max_epoch:
        cfg.OPTIM.MAX_EPOCH = args.max_epoch

    if args.eval_only:
        cfg.TRAINER.PointPRC.EVAL_ONLY = args.eval_only

    if args.no_mac:
        cfg.TRAINER.PointPRC.NO_MAC = args.no_mac

    if args.no_tdc:
        cfg.TRAINER.PointPRC.NO_TDC = args.no_tdc

    if args.no_mec:
        cfg.TRAINER.PointPRC.NO_MEC = args.no_mec

    if args.ulip_version:
        cfg.MODEL.ULIP_VERSION = args.ulip_version


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for PointPRC
    cfg.TRAINER.PointPRC = CN()
    cfg.TRAINER.PointPRC.EVAL_ONLY = False
    cfg.TRAINER.PointPRC.N_CTX_POINT = 4  # number of context vectors at the point branch
    cfg.TRAINER.PointPRC.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.PointPRC.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.PointPRC.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PointPRC.PROMPT_DEPTH_POINT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PointPRC.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.PointPRC.TEXT_LOSS_WEIGHT = 25     # a hyper-parameter, decided by ablation studies
    cfg.TRAINER.PointPRC.POINT_LOSS_WEIGHT = 10    # a hyper-parameter, decided by ablation studies
    cfg.TRAINER.PointPRC.MEC_MEAN = 15
    cfg.TRAINER.PointPRC.MEC_STD = 1   # std dev. can be float
    cfg.TRAINER.PointPRC.PROMPT_TYPE = "manual64"    # manual64, gpt35, gpt4, pointllm4, pointllm
    cfg.TRAINER.PointPRC.NO_MAC = False
    cfg.TRAINER.PointPRC.NO_TDC = False
    cfg.TRAINER.PointPRC.NO_MEC = False

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.EVAL_ONLY = False
    cfg.TRAINER.IVLP.N_CTX_POINT = 4  # number of context vectors at the image branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_POINT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.TRAINER.IVLP.PROMPT_TYPE = "manual64"    # manual64, gpt35, gpt4, pointllm4, pointllm

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.SONN_VARIANT = "hardest"
    cfg.DATASET.CORRUPTION_TYPE = "distortion"
    cfg.DATASET.TYPE = "other"  # choose from ["other", "pointda", "sim2real"]

    cfg.MODEL.ULIP_VERSION = "ulip2"

    # Config for point encoder
    cfg.PointEncoder = CN()
    cfg.PointEncoder.NAME = "pointbert"
    cfg.PointEncoder.trans_dim = 384
    cfg.PointEncoder.depth = 12
    cfg.PointEncoder.drop_path_rate = 0.1
    cfg.PointEncoder.cls_dim = 40
    cfg.PointEncoder.num_heads = 6
    cfg.PointEncoder.group_size = 32
    cfg.PointEncoder.num_group = 512
    cfg.PointEncoder.encoder_dims = 256
    cfg.PointEncoder.num_points = 1024

    # Config for text encoder
    cfg.TextEncoder = CN()
    cfg.TextEncoder.embed_dim = 512
    cfg.TextEncoder.ctx_len = 77
    cfg.TextEncoder.vocab_size = 49408
    cfg.TextEncoder.trans_width = 512 
    cfg.TextEncoder.trans_heads = 8
    cfg.TextEncoder.trans_layers = 12


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        if args.load_epoch > 0:
            epoch = args.load_epoch
        else:
            epoch = None
        print("Evaluating model ...")
        print(f"Runing the job and save the output to {args.output_dir}")
        trainer.load_model(args.model_dir, epoch=epoch)
        trainer.test()

        src_f = os.path.join(args.output_dir, 'run.log')
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        dst_f = os.path.join(args.output_dir, f'finished-{timestamp}.log')
        shutil.copy(src_f, dst_f)
        print(f"Copy run log done: finished-{timestamp}.log")
        
        return

    if not args.no_train:
        print(f"Run this job and save the output to {args.output_dir}")
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="the gpu id to use")
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument('--max-epoch', default=20, type=int, help='max epoches of training a model')
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--ulip-version", type=str, choices=["ulip1", "ulip2"], default="ulip2"
    )
    parser.add_argument(
        "--load-epoch", type=int, default=-1, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--no-mac", action="store_true", help="do not use Mutual Agreement Constraint during training"
    )
    parser.add_argument(
        "--no-tdc", action="store_true", help="do not use Text Diversity Constraint during training"
    )
    parser.add_argument(
        "--no-mec", action="store_true", help="do not use Model Ensemble Constraint during training"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)