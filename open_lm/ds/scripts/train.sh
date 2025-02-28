SWEEP_NAME=$1
# Number of GPUs you'd like to train on
NUM_GPUS=$2
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# name of wandb project to log to
WANDB_PROJECT=$3
# HF model name
HF_MODEL=$4
MODEL=$5
# Path to data-bins
TRAIN_DATA=$6
TRAIN_DATA_MANIFEST=$7
VAL_DATA=$8
MODEL_DIR=$9
# path to top-level directory to where you'd like to output the model
SEQ_LEN=${10}
NUM_WARMUP_STEPS=${11}
GLOBAL_BATCH_SIZE=${12}
GLOBAL_VAL_BATCH_SIZE=${13}
# total number of steps in training -- determines lr schedule
NUM_STEPS=${14}
NUM_TRAIN_TOKENS=$(($NUM_STEPS * $SEQ_LEN * $GLOBAL_BATCH_SIZE))
ACCUM_FREQ=${15}
USE_FSDP=${16}
LOG_INTERVAL=${17}
VALIDATION_INTERVAL=${18}
# learning rate
LR=${19}
END_LR_RATIO=${20}
CLIP_NORM=${21}
WD=${22}
ADAM_BETA2=${23}
EPOCH_STEPS=${24}
PROJECT_DIR=${25}
RUN_ID=${26}

CKPT_DIR=${MODEL_DIR}/$RUN_ID

NAME=$SWEEP_NAME/$RUN_ID;

EPOCHS=$((($NUM_STEPS-1) / $EPOCH_STEPS + 1))

NUM_TRAIN_TOKENS_PER_EPOCH=$(($NUM_TRAIN_TOKENS/$EPOCHS))

NUM_NODES=$((($NUM_GPUS - 1)/8 + 1 ))
if [[ $NUM_GPUS -gt 8 ]]
then
  NUM_GPUS_PER_NODE=8; 
else
  NUM_GPUS_PER_NODE=$NUM_GPUS;
fi

if [[ "$GLOBAL_VAL_BATCH_SIZE" != "None" ]]
then 
  GLOBAL_VAL_BATCH_SIZE_STR="--global-val-batch-size $GLOBAL_VAL_BATCH_SIZE"
fi

if [[ "$USE_FSDP" == "True" ]]
then 
  FSDP_STR=" --fsdp --fsdp-limit-all-gathers --fsdp-amp "
fi

echo "node-list: $SLURM_JOB_NODELIST"

stringsum() {
    echo "md5sum,md5" | tr ',' '\n' | while read -r cmd; do
        if [[ -x "$(command -v "${cmd}")" ]]; then
            num=$(( 0x$(echo "$1" | command "${cmd}" | cut -d ' ' -f 1 | head -c 15) ))
            [[ $num -lt 0 ]] && num=$((num * -1))
            echo $num
            return 0
        fi
    done
    return 1
}

export MASTER_PORT=$(( ($(stringsum $RUN_ID) % 10000) + 10000 ))
export WORLD_SIZE=$(($NUM_GPUS))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# zoom zoom - recommended from lightning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

python \
    ${PROJECT_DIR}/open_lm/main.py \
    --train-num-samples $NUM_TRAIN_TOKENS_PER_EPOCH \
    --workers 2 \
    --dataset-manifest $TRAIN_DATA_MANIFEST \
    --val-data $VAL_DATA \
    --precision bf16 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    $GLOBAL_VAL_BATCH_SIZE_STR \
    --accum-freq $ACCUM_FREQ \
    $FSDP_STR \
    --grad-checkpointing \
    --log-every-n-steps $LOG_INTERVAL \
    --grad-clip-norm $CLIP_NORM \
    --lr $LR \
    --lr-cooldown-end-ratio $END_LR_RATIO \
    --warmup $NUM_WARMUP_STEPS \
    --wd $WD \
    --beta2 $ADAM_BETA2 \
    --epochs $EPOCHS \
    --report-to wandb \
    --wandb-project-name $WANDB_PROJECT \
    -l ${CKPT_DIR} \
    --delete-previous-checkpoint \
    --resume latest \
    --data-key 'txt' \
    --val-data-key 'txt' \
    --model-norm rms_norm \
    --model $MODEL \
    --name $NAME 
