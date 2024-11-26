# 2024/01/19

TRAINER=IVLP   # 对于类别特别少的数据集，shapenet_9, modelnet_11，训练 10 个epoch就够了
DATA_TYPE=pointda
CFG=custom_ulip
SHOTS=16

PRINT_FREQ=10
PREFIX=output/${DATA_TYPE}/train

# custom config
GPU_ID=$1       # e.g., 0
DATA=$2         # e.g., data/pointda/modelnet
DATASET=$3      # e.g., pointda_modelnet
MAX_EPOCH=$4    # e.g., 10
DEPTH=$5        # e.g., 9
SEED=$6         # e.g., 1

DIR=${PREFIX}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${MAX_EPOCH}/seed${SEED}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_pointda \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --max-epoch ${MAX_EPOCH} \
    TRAINER.PointPRC.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.PointPRC.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.TYPE ${DATA_TYPE} \
    TRAIN.PRINT_FREQ ${PRINT_FREQ}
