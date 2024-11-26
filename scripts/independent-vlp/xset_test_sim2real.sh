# 2024/01/19

TRAINER=IVLP
DATA_TYPE=sim2real
CFG=custom_ulip
SHOTS=16

# custom config
GPU_ID=$1                   # e.g., 0
DATA=$2                     # e.g., data/sim2real/scanobjectnn_9
DATASET=$3                  # e.g., scanobjectnn_9
SOURCE=$4                   # e.g., shapenet_9
DEPTH=$5                    # e.g., 9
EPOCH=$6                    # e.g., 20
SEED=$7                     # e.g., 1
VARIANT=$8                  # e.g., hardest, obj_bg, obj_only
NUM_POINTS=$9               # e.g., 1024; for sim-to-real, set to 2048


PREFIX=output/sim2real/evaluation
if [ ${DATASET} = "scanobjectnn_9" ] || [ ${DATASET} = "scanobjectnn_11" ]; then
    DIR=${PREFIX}/${TRAINER}/shots_${SHOTS}/${CFG}_depth${DEPTH}_e${EPOCH}/${DATASET}_${VARIANT}/${NUM_POINTS}pts/seed${SEED}
else
    DIR=${PREFIX}/${TRAINER}/shots_${SHOTS}/${CFG}_depth${DEPTH}_e${EPOCH}/${DATASET}/${NUM_POINTS}pts/seed${SEED}
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}

pueue add -g 3d_ivlp_sim2real \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/sim2real/train/${SOURCE}/shots_${SHOTS}/${TRAINER}/${CFG}_depth${DEPTH}_e${EPOCH}/seed${SEED} \
    --eval-only \
    PointEncoder.num_points ${NUM_POINTS}\
    TRAINER.IVLP.PROMPT_DEPTH_POINT ${DEPTH} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH} \
    DATASET.TYPE ${DATA_TYPE} \
    DATASET.SONN_VARIANT ${VARIANT}
