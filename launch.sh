#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, no logging)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8

set -euo pipefail

usage() {
    echo "Usage: $0 <mode> <model_size> [--steps=<steps>] [--nodes=<nodes>] [--time=<time>] [--method=<method>] [--sequence-length=<seq_len>] [--mail-to=<mail_address>] [-d]"
    exit 1
}

[[ $# -lt 2 ]] && usage
MODE=${1}
MODEL_SIZE=${2}
shift 2

PARSED=$(getopt -o d --long steps:,nodes:,time:,method:,sequence-length:,mail-to: --name "$0" -- "$@") || usage
eval set -- "$PARSED"

STEPS=""
NODES=""
TIME=""
DRY_RUN=false
METHOD="fused"
SEQ_LEN=4096
SEND_MAIL_TO=""

while true; do
    case "$1" in
        --steps)  STEPS="$2";  shift 2 ;;
        --nodes)  NODES="$2";  shift 2 ;;
        --time)   TIME="$2";   shift 2 ;;
        --method)  METHOD="$2";  shift 2 ;;
        --sequence-length) SEQ_LEN="$2"; shift 2 ;;
        --mail-to) SEND_MAIL_TO="$2"; shift 2 ;;
        -d)       DRY_RUN=true;  shift ;;
        --)       shift; break         ;;
        *)        usage                ;;
    esac
done

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${STEPS:-50}
        NODES=${NODES:-4}
        TIME=${TIME:-00:30:00}
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=false
        ;;
    train)
        TRAINING_STEPS=${STEPS}
        NODES=${NODES:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

ENVIRONMENT="alps3"
ATTN_BACKEND="auto"
TRANSFORMER_IMPL="transformer_engine"
SPEC=""
case $METHOD in
    fused)
        ENVIRONMENT="alps3"
        ATTN_BACKEND="fused"
        TRANSFORMER_IMPL="transformer_engine"
        ;;
    flash)
        ENVIRONMENT=$(realpath ./flashattn3.toml)
        # ATTN_BACKEND="flash" # this works, 1 or AttnBackend.flash don't
        # TRANSFORMER_IMPL="local"
        TRANSFORMER_IMPL="transformer_engine"
        SPEC="megatron.fa3.gpt_layer_spec_fa3 get_gpt_layer_fa3_spec"
        ;;
    *)
        echo "Unknown method: $METHOD. Choose: fused, flash"
        exit 1
        ;;
esac

echo "Mode: $MODE"
echo "Model size: $MODEL_SIZE"
echo "Training steps: $TRAINING_STEPS"
echo "Nodes: $NODES"
echo "Time: $TIME"
echo "Sequence length: $SEQ_LEN"
echo "Attention backend: $ATTN_BACKEND"
echo "Environment: $ENVIRONMENT"
echo "Send mail to: ${SEND_MAIL_TO:-"none"}"

################ Model config ################
case $MODEL_SIZE in
    mini)
        NUM_LAYERS=4;   HIDDEN=128;  FFN=512;   HEADS=1;  KV_HEADS=1;
        MBS=1;  TP=1; PP=1
        ;;
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16; TP=1; PP=1
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8; TP=1; PP=1
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4; TP=1; PP=1
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4; TP=1; PP=1
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4; TP=1; PP=1
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2; TP=1; PP=1
        ;;
    32b) NUM_LAYERS=64; HIDDEN=6144; FFN=16384; HEADS=48; KV_HEADS=8
        MBS=1; TP=4; PP=1
        ;;
    140b) NUM_LAYERS=112; HIDDEN=10240; FFN=27648; HEADS=80; KV_HEADS=8
        MBS=1; TP=4; PP=4
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b, 32b, 140b"
        exit 1
        ;;
esac

GBS=256
# SEQ_LEN=4096
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-${SEQ_LEN}sl-${METHOD}"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=lsaie-ss26
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
#SBATCH -p debug
SBATCH_DIRECTIVES

if [ -n $SEND_MAIL_TO ]; then
    cat >> "$SCRIPT" << MAIL
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$SEND_MAIL_TO
MAIL
fi

cat >> "$SCRIPT" << 'BODY'

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/iopsstor/scratch/cscs/$USER/lsaie-ss26-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
BODY

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}
ATTN_BACKEND=${ATTN_BACKEND}
TRANSFORMER_IMPL=${TRANSFORMER_IMPL}
SPEC="${SPEC}"

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-\${SLURM_NNODES}n
LOG_DIR=\$WORKDIR/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && rm -rf $MEGATRON_LM_DIR/megatron/fa3 && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl ${TRANSFORMER_IMPL}
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --attention-backend ${ATTN_BACKEND}
)

if [ -n "$SPEC" ]; then
    TRANSFORMER_ENGINE_ARGS+=("--spec $SPEC")
fi

# if [ $TRANSFORMER_IMPL == "flash" ]; then
#     TRANSFORMER_ENGINE_ARGS+=("--no-persist-layer-norm")
# fi

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << REST

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'ENV_VARS'
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=1
ENV_VARS

cat >> "$SCRIPT" << FOOTER

echo "CMD: \$TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=$ENVIRONMENT --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 \$TRAINING_CMD"

echo "END TIME: \$(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"

if [ $DRY_RUN != "true" ]; then
    sbatch "$SCRIPT"
else
    echo "Dry run: not submitting to Slurm."
fi
