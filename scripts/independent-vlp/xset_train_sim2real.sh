# 2024/01/19

TRAINER=IVLP   # 对于类别特别少的数据集，shapenet_9, modelnet_11，训练 10 个epoch就够了
DATA_TYPE=sim2real
CFG=custom_ulip
NUM_POINTS=2048
SHOTS=16
PRINT_FREQ=20

# custom config
GPU_ID=$1       # e.g., 0
DATA=$2         # e.g., data/sim2real/modelnet_11
DATASET=$3      # e.g., modelnet_11
MAX_EPOCH=$4    # e.g., 20
DEPTH=$5        # e.g., 9
SEED=$6         # e.g., 1

PREFIX=output/${DATA_TYPE}/train
DIR=${PREFIX}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${MAX_EPOCH}/seed${SEED}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_sim2real \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --max-epoch ${MAX_EPOCH} \
    PointEncoder.num_points ${NUM_POINTS}\
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.TYPE ${DATA_TYPE} \
    TRAIN.PRINT_FREQ ${PRINT_FREQ}
