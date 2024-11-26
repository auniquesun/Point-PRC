# 2024/04/08

TRAINER=PointPRC   # For datasets which contain only a few classes, such as `shapenet_9` and `modelnet_11`, training 10 epochs is enough
DATA_TYPE=sim2real
CFG=custom_ulip
SHOTS=16
PRINT_FREQ=20

# custom config
GPU_ID=$1       # e.g., 0
DATA=$2         # e.g., data/xset/sim2real/modelnet_11, data/xset/sim2real/shapenet_9
DATASET=$3      # e.g., sim2real_mn11, sim2real_sn9
MAX_EPOCH=$4    # e.g., 20
DEPTH=$5        # e.g., 9
N_CTX_POINT=$6  # e.g., 4
N_CTX_TEXT=$7   # e.g., 5
VERSION=$8      # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
NUM_POINTS=$9   # e.g., 2048, 1024  metasets use `2048` by default
EXP_TYPE=${10}  # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${11}    # eg.., False
NO_TDC=${12}    # eg.., False
NO_MEC=${13}    # eg.., False
PROMPT_TYPE=${14}           # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${15}          # e.g., ulip1, ulip2
DIST_TYPE=${16} # e.g., l1_dist, mse_dist, cosine_dist

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/train/${DATA_TYPE}

for SEED in 1 2 3
do
    DIR=${PREFIX}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${MAX_EPOCH}/${NUM_POINTS}pts/seed${SEED}

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_xset_sim2real_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --max-epoch ${MAX_EPOCH} \
        PointEncoder.num_points ${NUM_POINTS} \
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