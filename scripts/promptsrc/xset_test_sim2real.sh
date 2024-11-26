# 2024/04/09

TRAINER=PointPRC
DATA_TYPE=sim2real
CFG=custom_ulip

# custom config
GPU_ID=$1                   # e.g., 0
DATA=$2                     # e.g., data/xset/sim2real/so_hardest_9, data/xset/sim2real/so_hardest_11
DATASET=$3                  # e.g., sim2real_sonn
SOURCE=$4                   # e.g., sim2real_sn9, sim2real_mn11
DEPTH=$5                    # e.g., 9
N_CTX_POINT=$6              # e.g., 4
N_CTX_TEXT=$7               # e.g., 5
EPOCH=$8                    # e.g., 20
SHOTS=$9                    # e.g., 16
VERSION=${10}               # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
NUM_POINTS=${11}            # e.g., 2048, 1024  metasets use `2048` by default
EXP_TYPE=${12}              # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${13}                # eg.., False
NO_TDC=${14}                # eg.., False
NO_MEC=${15}                # eg.., False
PROMPT_TYPE=${16}           # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${17}          # e.g., ulip1, ulip2
DIST_TYPE=${18}             # e.g., l1_dist, mse_dist, cosine_dist

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/eval/${DATA_TYPE}

if echo "$DATA" | grep -q "obj_only"; then
    VARIANT=obj_only
elif echo "$DATA" | grep -q "obj_bg"; then
    VARIANT=obj_bg
elif echo "$DATA" | grep -q "hardest"; then
    VARIANT=hardest
fi

for SEED in 1 2 3
do
    DIR=${PREFIX}/source_${SOURCE}/${DATASET}_${VARIANT}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/${NUM_POINTS}pts/seed${SEED}

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_xset_sim2real_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/train/${DATA_TYPE}/${SOURCE}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/${NUM_POINTS}pts/seed${SEED} \
        --eval-only \
        PointEncoder.num_points ${NUM_POINTS}\
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