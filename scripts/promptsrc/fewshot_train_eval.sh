# 2024/04/03

TRAINER=PointPRC

# custom config
GPU_ID=$1       # e.g., 0
DATA=$2         # e.g., data/fewshot/modelnet40, data/fewshot/scanobjectnn
DATASET=$3      # e.g., modelnet40, scanobjectnn
CFG=$4          # e.g., custom_ulip_few_shot
PROMPT_DEPTH=$5 # e.g., 9
N_CTX_POINT=$6  # e.g., 4
N_CTX_TEXT=$7   # e.g., 4
SHOTS=$8        # e.g., 16
EPOCH=$9        # e.g., 50, this is longer than domain shift/base2new training
VARIANT=${10}   # e.g., hardest, obj_bg, obj_only
VERSION=${11}   # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
EXP_TYPE=${12}  # e.g., ablate_architecture, ablate_components, ablate_prompt_types, task_perform
NO_MAC=${13}    # eg.., False
NO_TDC=${14}    # eg.., False
NO_MEC=${15}    # eg.., False
PROMPT_TYPE=${16}  # e.g., manual64, gpt35, gpt4, pointllm
ULIP_VERSION=${17}          # e.g., ulip1, ulip2
DIST_TYPE=${18} # e.g., l1_dist, mse_dist, cosine_dist

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/fewshot/${DIST_TYPE}

for SEED in 1 2 3
do
    if [ ${DATASET} = "scanobjectnn" ]; then
        DIR=${PREFIX}/${DATASET}_${VARIANT}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${PROMPT_DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED}
    else
        DIR=${PREFIX}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${PROMPT_DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/seed${SEED}
    fi

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_fs_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --max-epoch ${EPOCH} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.PointPRC.PROMPT_TYPE ${PROMPT_TYPE} \
        TRAINER.PointPRC.PROMPT_DEPTH_POINT ${PROMPT_DEPTH} \
        TRAINER.PointPRC.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH} \
        TRAINER.PointPRC.N_CTX_POINT ${N_CTX_POINT} \
        TRAINER.PointPRC.N_CTX_TEXT ${N_CTX_TEXT} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SONN_VARIANT ${VARIANT} \
        TRAINER.PointPRC.NO_MAC ${NO_MAC} \
        TRAINER.PointPRC.NO_TDC ${NO_TDC} \
        TRAINER.PointPRC.NO_MEC ${NO_MEC} 
done
