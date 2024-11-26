# 2024/01/19

TRAINER=IVLP
DATA_TYPE=pointda
CFG=custom_ulip
PROMPT_TYPE=manual64
SHOTS=16
DEPTH=9

# custom config
GPU_ID=$1                   # e.g., 0
DATA=$2                     # e.g., data/pointda/shapenet
DATASET=$3                  # e.g., pointda_shapenet
SOURCE=$4                   # e.g., pointda_modelnet
EPOCH=$5                    # e.g., 20
SEED=$6                     # e.g., 1


PREFIX=output/${DATA_TYPE}/evaluation/source_${SOURCE}
DIR=${PREFIX}/${TRAINER}/shots_${SHOTS}/${CFG}_depth${DEPTH}_e${EPOCH}/${DATASET}/seed${SEED}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_pointda \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/${DATA_TYPE}/train/${SOURCE}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${EPOCH}/seed${SEED} \
    --eval-only \
    TRAINER.IVLP.PROMPT_TYPE ${PROMPT_TYPE} \
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.TYPE ${DATA_TYPE} 
