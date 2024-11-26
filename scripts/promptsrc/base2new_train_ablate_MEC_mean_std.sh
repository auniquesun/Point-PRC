# 2024/04/03

TRAINER=PointPRC

# custom config
GPU_ID=$1       # e.g., 0
DATA=$2         # e.g., data/base2new/scanobjectnn
DATASET=$3      # e.g., scanobjectnn
CFG=$4          # e.g., custom_ulip
PROMPT_TYPE=$5  # e.g., manual64, gpt35, gpt4, pointllm
PROMPT_DEPTH=$6 # e.g., 9
N_CTX_POINT=$7  # e.g., 4
N_CTX_TEXT=$8   # e.g., 4
SHOTS=$9        # e.g., 16
EPOCH=${10}     # e.g., 20
VARIANT=${11}   # e.g., hardest, obj_bg, obj_only
VERSION=${12}   # e.g., full, no_mac, no_tdc, no_mec, only_mac, only_tdc, only_mec, no_all
EXP_TYPE=${13}  # e.g., task_perform, ablate_architecture, ablate_components, ablate_prompt_types
NO_MAC=${14}    # e.g., False
NO_TDC=${15}    # e.g., False
NO_MEC=${16}    # e.g., False
ULIP_VERSION=${17}  # e.g., ulip1, ulip2
DIST_TYPE=${18} # e.g., l1_dist, mse_dist, cosine_dist
MEC_MEAN=${19}
MEC_STD=${20}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
PREFIX=output/${EXP_TYPE}_${ULIP_VERSION}/base2new/${DIST_TYPE}/train_base

for SEED in 4 5 6
do
    if [ ${DATASET} = "scanobjectnn" ]; then
        DIR=${PREFIX}/${DATASET}_${VARIANT}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${PROMPT_DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/${MEC_MEAN}_${MEC_STD}/seed${SEED}
        PRINT_FREQ=20
    else
        DIR=${PREFIX}/${DATASET}/${SHOTS}shots/${TRAINER}_${VERSION}_${PROMPT_TYPE}/${CFG}_depth${PROMPT_DEPTH}_width${N_CTX_POINT}_${N_CTX_TEXT}_e${EPOCH}/${MEC_MEAN}_${MEC_STD}/seed${SEED}
        PRINT_FREQ=40
    fi

    pueue add -g pointprc_${EXP_TYPE}_${ULIP_VERSION}_base2new_${GPU_ID} \
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --ulip-version ${ULIP_VERSION} \
        --trainer ${TRAINER} \
        --max-epoch ${EPOCH} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAIN.PRINT_FREQ ${PRINT_FREQ} \
        TRAINER.PointPRC.PROMPT_TYPE ${PROMPT_TYPE} \
        TRAINER.PointPRC.PROMPT_DEPTH_POINT ${PROMPT_DEPTH} \
        TRAINER.PointPRC.PROMPT_DEPTH_TEXT ${PROMPT_DEPTH} \
        TRAINER.PointPRC.N_CTX_POINT ${N_CTX_POINT} \
        TRAINER.PointPRC.N_CTX_TEXT ${N_CTX_TEXT} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SONN_VARIANT ${VARIANT} \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.PointPRC.NO_MAC ${NO_MAC} \
        TRAINER.PointPRC.NO_TDC ${NO_TDC} \
        TRAINER.PointPRC.NO_MEC ${NO_MEC} \
        TRAINER.PointPRC.MEC_MEAN ${MEC_MEAN} \
        TRAINER.PointPRC.MEC_STD ${MEC_STD}
done