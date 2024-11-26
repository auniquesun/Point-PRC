# 2024/03/29
# NOTE: This script is suitable for `modelnet_c` & `modelnet40_c` dataset

TRAINER=PointPRC
PROMPT_TYPE=manual64
CFG=custom_ulip_few_shot
SHOTS=16

# custom config
GPU_ID=$1                   # e.g., 0
DATA=$2                     # e.g., data/xset/corruption/modelnet_c, data/xset/corruption/modelnet40_c
DATASET=$3                  # e.g., modelnet_c, modelnet40_c
SOURCE=$4                   # e.g., modelnet40
DEPTH=$5                    # e.g., 9
N_CTX_POINT=$6              # e.g., 4
N_CTX_TEXT=$7               # e.g., 4
EPOCH=$8                    # e.g., 50
COR_TYPE=$9                 # e.g., add_global_2, add_local_2, `dropout_global_2`, `dropout_local_2`, jitter_2, rotate_2, scale_2 for `modelnet_c`; background, distortion, gaussian, uniform, etc. for `modelnet40_c`
VERSION=${10}               # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
EXP_TYPE=${11}              # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${12}                # eg.., False
NO_TDC=${13}                # eg.., False
NO_MEC=${14}                # eg.., False
PROMPT_TYPE=${15}           # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${16}          # e.g., ulip1, ulip2
DIST_TYPE=${17}             # e.g., l1_dist, mse_dist, cosine_dist

echo "... COR_TYPE: ${COR_TYPE} ..."
export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/eval/corruption

for SEED in 1 2 3
do
    DIR=${PREFIX}/source_${SOURCE}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/${COR_TYPE}/seed${SEED}

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_xset_corruption_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir output/${EXP_TYPE}_${ULIP_VERSION}/fewshot/${DIST_TYPE}/${SOURCE}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED} \
        --eval-only \
        TRAINER.PointPRC.PROMPT_TYPE ${PROMPT_TYPE} \
        TRAINER.PointPRC.PROMPT_DEPTH_POINT ${DEPTH} \
        TRAINER.PointPRC.PROMPT_DEPTH_TEXT ${DEPTH} \
        TRAINER.PointPRC.N_CTX_POINT ${N_CTX_POINT} \
        TRAINER.PointPRC.N_CTX_TEXT ${N_CTX_TEXT} \
        DATASET.CORRUPTION_TYPE ${COR_TYPE} \
        TRAINER.PointPRC.NO_MAC ${NO_MAC} \
        TRAINER.PointPRC.NO_TDC ${NO_TDC} \
        TRAINER.PointPRC.NO_MEC ${NO_MEC} 
done