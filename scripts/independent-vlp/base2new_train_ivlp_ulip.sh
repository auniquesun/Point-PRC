# 2024/01/19

TRAINER=IVLP
CFG=custom_ulip
SHOTS=16
PREFIX=output/base2new/train_base
PROMPT_DEPTH=9
PROMPT_TYPE=manual64

# custom config
GPU_ID=$1       # 0
DATA=$2         # data/base2new/shapenetcorev2
DATASET=$3      # shapenetcorev2
MAX_EPOCH=$4    # 20
SEED=$5         # 1
VARIANT=$6      # hardest

if [ ${DATASET} = "scanobjectnn" ]; then
    DIR=${PREFIX}/${DATASET}_${VARIANT}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
else
    DIR=${PREFIX}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
pueue add -g 3d_ivlp_base2new \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --max-epoch ${MAX_EPOCH} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.IVLP.PROMPT_TYPE ${PROMPT_TYPE} \
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${PROMPT_DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SONN_VARIANT ${VARIANT} \
    DATASET.SUBSAMPLE_CLASSES base
