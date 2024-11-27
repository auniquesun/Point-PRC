# 2024/05/02

TRAINER=PointPRC
DATA_TYPE=pointda
CFG=custom_ulip
SHOTS=16                    # e.g., 16

# custom config
GPU_ID=$1                   # e.g., 0
DATA=$2                     # e.g., data/xset/pointda/shapenet
DATASET=$3                  # e.g., pointda_shapenet
SOURCE=$4                   # e.g., pointda_modelnet
DEPTH=$5                    # e.g., 9
N_CTX_POINT=$6              # e.g., 4
N_CTX_TEXT=$7               # e.g., 5
EPOCH=$8                    # e.g., 10
VERSION=$9                  # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
EXP_TYPE=${10}              # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${11}                # eg.., False
NO_TDC=${12}                # eg.., False
NO_MEC=${13}                # eg.., False
PROMPT_TYPE=${14}           # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${15}          # e.g., ulip1, ulip2
DIST_TYPE=${16}             # e.g., l1_dist, mse_dist, cosine_dist

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/eval/${DATA_TYPE}

for SEED in 1 2 3
do
    DIR=${PREFIX}/source_${SOURCE}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED}

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_xset_pointda_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/train/${DATA_TYPE}/${SOURCE}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED} \
        --eval-only \
        TRAINER.PointPRC.PROMPT_TYPE ${PROMPT_TYPE} \
        TRAINER.PointPRC.PROMPT_DEPTH_POINT ${DEPTH} \
        TRAINER.PointPRC.PROMPT_DEPTH_TEXT ${DEPTH} \
        TRAINER.PointPRC.N_CTX_POINT ${N_CTX_POINT} \
        TRAINER.PointPRC.N_CTX_TEXT ${N_CTX_TEXT} \
        DATASET.TYPE ${DATA_TYPE} \
        TRAINER.PointPRC.NO_MAC ${NO_MAC} \
        TRAINER.PointPRC.NO_TDC ${NO_TDC} \
        TRAINER.PointPRC.NO_MEC ${NO_MEC} 
done