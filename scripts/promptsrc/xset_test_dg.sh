# 2024/04/12

TRAINER=PointPRC
CFG=custom_ulip_few_shot
SHOTS=16

# custom config
GPU_ID=$1                   # e.g., 0
DATA=$2                     # e.g., data/xset/dg/modelnet40, data/xset/dg/omniobject3d
DATASET=$3                  # e.g., modelnet40, omniobject3d, sonn variants, etc
SOURCE=$4                   # e.g., shapenetcorev2
DEPTH=$5                    # e.g., 12, 9
N_CTX_POINT=$6              # e.g., 4
N_CTX_TEXT=$7               # e.g., 5
EPOCH=$8                    # e.g., 50, 20
VARIANT=$9                  # e.g., hardest, obj_bg, obj_only
VERSION=${10}               # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
EXP_TYPE=${11}              # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${12}                # eg.., False
NO_TDC=${13}                # eg.., False
NO_MEC=${14}                # eg.., False
PROMPT_TYPE=${15}           # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${16}          # e.g., ulip1, ulip2
DIST_TYPE=${17}             # e.g., l1_dist, mse_dist, cosine_dist

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/xset/${DIST_TYPE}/eval/dg

for SEED in 1 2 3
do
    if [ ${DATASET} = "scanobjectnn" ]; then
        DIR=${PREFIX}/${DATASET}_${VARIANT}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED}
    else
        DIR=${PREFIX}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED}
    fi

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_xset_dg_${GPU_ID} \
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
        DATASET.SONN_VARIANT ${VARIANT} \
        TRAINER.PointPRC.NO_MAC ${NO_MAC} \
        TRAINER.PointPRC.NO_TDC ${NO_TDC} \
        TRAINER.PointPRC.NO_MEC ${NO_MEC} 
done