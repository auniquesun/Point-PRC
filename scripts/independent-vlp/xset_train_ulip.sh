# 2024/01/19

# custom config
GPU_ID=0
TRAINER=IVLP
DEPTH=9
PROMPT_TYPE=manual64
CFG=custom_ulip
SHOTS=16
PREFIX=output/xset/train
PRINT_FREQ=100

DATA=$1                         # e.g., data/xset/shapenetcorev2
DATASET=$2                      # e.g., shapenetcorev2
MAX_EPOCH=$3
SEED=$4

DIR=${PREFIX}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${MAX_EPOCH}/seed${SEED}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_xset \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --max-epoch ${MAX_EPOCH} \
    TRAIN.PRINT_FREQ ${PRINT_FREQ} \
    TRAINER.IVLP.PROMPT_TYPE ${PROMPT_TYPE} \
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.NUM_SHOTS ${SHOTS}
