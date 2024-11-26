#!/bin/bash

GPU_ID=0
TRAINER=IVLP
DEPTH=9
PROMPT_TYPE=manual64
CFG=custom_ulip
SHOTS=16
PREFIX=output/xset/eval

# custom config
DATA=$1                     # e.g., data/modelnet40
DATASET=$2                  # e.g., modelnet40
SOURCE=$3                   # e.g., shapenetcorev2
EPOCH=$4                    # e.g., 20
SEED=$5                     # e.g., 3
VARIANT=$6                  # e.g., hardest, obj_bg, obj_only

if [ ${DATASET} = "scanobjectnn" ]; then
    DIR=${PREFIX}/${TRAINER}/${CFG}_${SHOTS}shots/depth${DEPTH}_e${EPOCH}/${DATASET}_${VARIANT}/seed${SEED}
else
    DIR=${PREFIX}/${TRAINER}/${CFG}_${SHOTS}shots/depth${DEPTH}_e${EPOCH}/${DATASET}/seed${SEED}
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_xset \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/xset/train/${SOURCE}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${EPOCH}/seed${SEED} \
    --eval-only \
    TRAINER.IVLP.PROMPT_TYPE ${PROMPT_TYPE} \
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.SONN_VARIANT ${VARIANT}
