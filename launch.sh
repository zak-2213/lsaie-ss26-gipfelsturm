#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, no logging)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b, 32b, 140b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Env toggles:
#   FP8=1                  enable FP8 GEMMs via Transformer Engine (E4M3 fwd / E5M2 bwd)
#   NSYS=1                 wrap training in nsys profile (writes .nsys-rep into $LOG_DIR)
#   RECOMPUTE=full|selective|none   override activation recompute (default: full for 32b/140b, none otherwise)
#   OVERLAP_PARAM_GATHER=0|1        toggle --overlap-param-gather (default 1)
#   GBS=<int>              override global batch size (default 256)
#   TIME_LIMIT=HH:MM:SS    override SLURM time limit (default depends on mode and model size)
#
# Examples:
#   ./launch.sh throughput 125m 2 1                  # smoke test, BF16, 2 steps, 1 node
#   FP8=1 ./launch.sh throughput 125m 2 1            # smoke test, FP8
#   ./launch.sh throughput 32b 50 1                  # BF16 baseline, 32B, TP=4, single node
#   FP8=1 ./launch.sh throughput 32b 50 1            # FP8 run, 32B, TP=4, single node
#   NSYS=1 FP8=1 ./launch.sh throughput 32b 20 1     # FP8 + nsys profile

set -euo pipefail

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        if [ "$LR_WARMUP_ITERS" -ge "$TRAINING_STEPS" ]; then
            LR_WARMUP_ITERS=$(( TRAINING_STEPS - 1 ))
        fi
        LOGGING_EXTRA=""
        WANDB=false
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
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

################ Model config ################
TP=1
PP=1
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    32b) NUM_LAYERS=64; HIDDEN=6144; FFN=16384; HEADS=48; KV_HEADS=8
        MBS=1
        TP=4
        ;;
    140b) NUM_LAYERS=112; HIDDEN=10240; FFN=27648; HEADS=80; KV_HEADS=8
        MBS=1
        TP=4
        PP=4
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b, 32b, 140b"
        exit 1
        ;;
esac

SEQ_PARALLEL_ARG=""
if [ "$TP" -gt 1 ]; then
    SEQ_PARALLEL_ARG="--sequence-parallel"
fi

# 32B/140B iter time is ~145s under BF16+TP=4+full-recompute, so 50 iters need ~2h.
# The default 30-min cap (used for smaller models) cuts these runs at iter ~11.
case $MODEL_SIZE in
    32b|140b)
        if [ "$MODE" = "throughput" ]; then
            TIME=02:30:00
        fi
        ;;
esac

# Allow user to override SLURM time limit, e.g. for tighter scheduling:
#   TIME_LIMIT=01:00:00 ./launch.sh throughput 32b 20 1
if [ -n "${TIME_LIMIT:-}" ]; then
    TIME="$TIME_LIMIT"
fi

DEFAULT_RECOMPUTE=none
case $MODEL_SIZE in
    32b|140b) DEFAULT_RECOMPUTE=full ;;
esac
RECOMPUTE=${RECOMPUTE:-$DEFAULT_RECOMPUTE}
RECOMPUTE_ARG=""
if [ "$RECOMPUTE" = "full" ]; then
    RECOMPUTE_ARG="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
elif [ "$RECOMPUTE" = "selective" ]; then
    RECOMPUTE_ARG="--recompute-granularity selective"
fi

GPUS_PER_NODE=4
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
NEEDED_GPUS=$((TP * PP))
if [ "$TOTAL_GPUS" -lt "$NEEDED_GPUS" ]; then
    echo "Error: model $MODEL_SIZE needs TP=$TP PP=$PP ($NEEDED_GPUS GPUs) but only $TOTAL_GPUS GPUs requested ($NODES nodes x $GPUS_PER_NODE)."
    exit 1
fi

GBS=${GBS:-256}
SEQ_LEN=4096
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n"

################ FP8 toggle (Samy / fp8 axis) ################
# FP8=1 enables Transformer Engine FP8 GEMMs. BF16 stays for non-GEMM ops
# (activations, optimizer state) — only the matmul path switches to FP8.
# Format `hybrid` = E4M3 forward / E5M2 backward, the NVIDIA-recommended default.
#
# Memory mitigations for 32B FP8 (BF16 peaks at 87/96 GB; TE FP8 adds ~5 GB):
#   --fp8-param-gather       all-gather params in FP8 instead of BF16 during the
#                            distributed-optimizer step, saves the all-gather buffer
#   OVERLAP_PARAM_GATHER=0   skips the extra param-gather buffer (~500 MB)
# Both are auto-applied when FP8=1 unless the user overrides them.
FP8=${FP8:-0}
FP8_ARGS=""
FP8_DEFAULT_OVERLAP=1
if [ "$FP8" = "1" ]; then
    FP8_ARGS="--fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-margin 0 --fp8-param-gather"
    JOB_NAME="${JOB_NAME}-fp8"
    FP8_DEFAULT_OVERLAP=0   # save ~500 MB by default for FP8 32B runs
fi

LIGER=${LIGER:-0}
if [ "$LIGER" = "1" ]; then
    USE_LIGER_SWIGLU=${USE_LIGER_SWIGLU:-1}
    USE_LIGER_ROPE=${USE_LIGER_ROPE:-1}
    USE_LIGER_RMSNORM=${USE_LIGER_RMSNORM:-1}
    USE_LIGER_CE=${USE_LIGER_CE:-1}
fi
USE_LIGER_SWIGLU=${USE_LIGER_SWIGLU:-0}
USE_LIGER_ROPE=${USE_LIGER_ROPE:-0}
USE_LIGER_RMSNORM=${USE_LIGER_RMSNORM:-0}
USE_LIGER_CE=${USE_LIGER_CE:-0}
USE_LIGER_FUSED_LINEAR_CE=${USE_LIGER_FUSED_LINEAR_CE:-0}

OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-$FP8_DEFAULT_OVERLAP}
if [ "$USE_LIGER_FUSED_LINEAR_CE" = "1" ]; then
    OVERLAP_PARAM_GATHER=0
fi
OVERLAP_PARAM_GATHER_ARG="--overlap-param-gather"
if [ "$OVERLAP_PARAM_GATHER" = "0" ]; then
    OVERLAP_PARAM_GATHER_ARG=""
fi

NSYS=${NSYS:-0}

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
SBATCH_DIRECTIVES

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

# Liger-Kernel kernel selection (forwarded from launcher env)
export USE_LIGER_SWIGLU=${USE_LIGER_SWIGLU}
export USE_LIGER_ROPE=${USE_LIGER_ROPE}
export USE_LIGER_RMSNORM=${USE_LIGER_RMSNORM}
export USE_LIGER_CE=${USE_LIGER_CE}
export USE_LIGER_FUSED_LINEAR_CE=${USE_LIGER_FUSED_LINEAR_CE}
export NSYS=${NSYS}

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
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Reduce CUDA allocator fragmentation. Critical for 32B FP8: BF16 peaks at
# ~87 GB/96 GB and FP8's TE state pushes it past 95 GB at FusedAdam init
# without expandable_segments. Recommended by the PyTorch OOM message.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

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

PROFILE_ARG=""
if [ "${NSYS}" = "1" ]; then
    PROFILE_ARG="--profile"
fi

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
    ${RECOMPUTE_ARG}
    \$PROFILE_ARG
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

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

REST

cat >> "$SCRIPT" << MIXED_PRECISION
MIXED_PRECISION_ARGS=(
    --bf16
    ${FP8_ARGS}
)
MIXED_PRECISION

cat >> "$SCRIPT" << DISTRIBUTED
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    ${SEQ_PARALLEL_ARG}
    --use-distributed-optimizer
    --overlap-grad-reduce
    ${OVERLAP_PARAM_GATHER_ARG}
)
DISTRIBUTED

cat >> "$SCRIPT" << 'REST'

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

NSYS_PREFIX=""
if [ "${NSYS:-0}" = "1" ]; then
    NSYS_OUT=$LOG_DIR/nsys-$EXP_NAME-$SLURM_JOB_ID
    NSYS_PREFIX="nsys profile --output=$NSYS_OUT --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none"
fi

TRAINING_CMD="$NSYS_PREFIX torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
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

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
