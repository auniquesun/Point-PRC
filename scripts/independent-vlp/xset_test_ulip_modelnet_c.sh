# 2024/01/19

GPU_ID=0
TRAINER=IVLP
DEPTH=9
PROMPT_TYPE=manual64
CFG=custom_ulip
SHOTS=16
PREFIX=output/corruption/evaluation

# custom config
DATA=$1                     # e.g., data/corruption/modelnet_c
DATASET=$2                  # e.g., modelnet_c
SOURCE=$3                   # e.g., modelnet40
EPOCH=$4                    # e.g., 20
SEED=$5                     # e.g., 3
COR_TYPE=${6}               # e.g., add_global_2, add_local_2, dropout_global_2, dropout_local_2, jitter_2, rotate_2, scale_2

echo "... COR_TYPE: ${COR_TYPE} ..."
DIR=${PREFIX}/${TRAINER}/${CFG}_${SHOTS}shots/depth${DEPTH}_e${EPOCH}/${DATASET}/${COR_TYPE}/seed${SEED}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_xset \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/corruption/train/${SOURCE}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${EPOCH}/seed${SEED} \
    --eval-only \
    TRAINER.IVLP.PROMPT_TYPE ${PROMPT_TYPE} \
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.CORRUPTION_TYPE ${COR_TYPE}
