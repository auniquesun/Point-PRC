# 2024/05/02

TRAINER=PointPRC   # For datasets which contain only a few classes, such as `shapenet_9` and `modelnet_11`, training 10 epochs is enough
DATA_TYPE=pointda
CFG=custom_ulip
SHOTS=16
PRINT_FREQ=20

# custom config
GPU_ID=$1       # e.g., 0
DATA=$2         # e.g., data/xset/pointda/modelnet
DATASET=$3      # e.g., pointda_modelnet
MAX_EPOCH=$4    # e.g., 20, 10
DEPTH=$5        # e.g., 9
N_CTX_POINT=$6  # e.g., 4
N_CTX_TEXT=$7   # e.g., 5
VERSION=$8      # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
EXP_TYPE=$9     # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${10}    # e.g., False
NO_TDC=${11}    # e.g., False
NO_MEC=${12}    # e.g., False
PROMPT_TYPE=${13}           # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${14}          # e.g., ulip1, ulip2
DIST_TYPE=${15} # e.g., l1_dist, mse_dist, cosine_dist

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/train/${DATA_TYPE}

for SEED in 1 2 3
do
    DIR=${PREFIX}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${MAX_EPOCH}/seed${SEED}

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_xset_pointda_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --max-epoch ${MAX_EPOCH} \
        TRAINER.PointPRC.PROMPT_TYPE ${PROMPT_TYPE} \
        TRAINER.PointPRC.PROMPT_DEPTH_POINT ${DEPTH} \
        TRAINER.PointPRC.PROMPT_DEPTH_TEXT ${DEPTH} \
        TRAINER.PointPRC.N_CTX_POINT ${N_CTX_POINT} \
        TRAINER.PointPRC.N_CTX_TEXT ${N_CTX_TEXT} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.TYPE ${DATA_TYPE} \
        TRAIN.PRINT_FREQ ${PRINT_FREQ} \
        TRAINER.PointPRC.NO_MAC ${NO_MAC} \
        TRAINER.PointPRC.NO_TDC ${NO_TDC} \
        TRAINER.PointPRC.NO_MEC ${NO_MEC} 
done