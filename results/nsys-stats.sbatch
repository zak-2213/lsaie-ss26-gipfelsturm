#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:15:00
#SBATCH --partition=debug
#SBATCH --job-name=nsys-stats
#SBATCH --output=results/nsys-stats-%j.out
#SBATCH --error=results/nsys-stats-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
#
# Run nsys stats on the BF16 and FP8 .nsys-rep traces and dump the kernel
# summary tables into results/. Uses the exact same srun + container setup
# as the working training launcher (launch.sh), so pyxis is happy.
#
# Submit from the repo root on Clariden:
#   sbatch results/nsys-stats.sbatch

WORKDIR=/iopsstor/scratch/cscs/$USER/lsaie-ss26-gipfelsturm
cd $WORKDIR
mkdir -p results

NSYS_DIR=$WORKDIR/gipfelsturm/throughput-8b-1n
BF16_REP=$NSYS_DIR/nsys-throughput-8b-1n-2404787.nsys-rep
FP8_REP=$NSYS_DIR/nsys-throughput-8b-1n-2404788.nsys-rep

echo "===== BF16 trace: $BF16_REP ====="
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "nsys stats --report cuda_gpu_kern_sum --format column $BF16_REP" \
    > results/bf16_kernels.txt 2>&1
echo "  wrote results/bf16_kernels.txt"

echo
echo "===== FP8 trace: $FP8_REP ====="
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK --wait 60 \
    bash -c "nsys stats --report cuda_gpu_kern_sum --format column $FP8_REP" \
    > results/fp8_kernels.txt 2>&1
echo "  wrote results/fp8_kernels.txt"

echo
echo "===== HEAD of BF16 kernel summary ====="
head -n 50 results/bf16_kernels.txt
echo
echo "===== HEAD of FP8 kernel summary ====="
head -n 50 results/fp8_kernels.txt
