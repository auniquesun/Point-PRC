# 2024/01/19

GPU_ID=0
TRAINER=IVLP
DEPTH=9
PROMPT_TYPE=manual64
CFG=custom_ulip
SHOTS=16
PREFIX=output/xset/eval

# custom config
DATA=$1                     # e.g., data/modelnet40_c
DATASET=$2                  # e.g., modelnet40_c
SOURCE=$3                   # e.g., modelnet40
EPOCH=$4                    # e.g., 20
SEED=$5                     # e.g., 3
COR_TYPE=${6}               # e.g., distortion, guassian, occlusion, uniform, rotation ...

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
    --model-dir output/xset/train/${SOURCE}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${EPOCH}/seed${SEED} \
    --eval-only \
    TRAINER.IVLP.PROMPT_TYPE ${PROMPT_TYPE} \
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.CORRUPTION_TYPE ${COR_TYPE}
