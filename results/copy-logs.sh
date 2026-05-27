#!/bin/bash
#
# Copy the experiment logs we cite in the FP8 report into results/logs/,
# renaming each to describe what it measured.
#
# Run on the Clariden cluster after pulling the fp8 branch:
#   cd /iopsstor/scratch/cscs/$USER/lsaie-ss26-gipfelsturm
#   bash results/copy-logs.sh
#   git add results/
#   git commit -m "Add experiment logs cited in the FP8 report"
#   git push
#
# Any logs that have aged out of scratch will be silently skipped.

set -u
mkdir -p results/logs

cp_log() {
    local jobid="$1"
    local newname="$2"
    local pattern="logs/*${jobid}*.log"
    local src=$(ls $pattern 2>/dev/null | head -n 1)
    if [ -z "$src" ]; then
        echo "MISSING: job ${jobid} (would have been ${newname})"
        return
    fi
    cp "$src" "results/logs/${newname}.log"
    echo "  ${jobid}  ->  results/logs/${newname}.log"
}

echo "=== Headline 1.45x apples-to-apples FP8 result ==="
cp_log 2404785 "8b-bf16-recompute-full-20iters"
cp_log 2404786 "8b-fp8-recompute-full-50iters-HEADLINE"

echo "=== BF16 baselines ==="
cp_log 2344091 "32b-bf16-baseline-clean"
cp_log 2368459 "8b-bf16-no-recompute-team-baseline"

echo "=== First successful 8B FP8 run (replicate for confidence) ==="
cp_log 2395583 "8b-fp8-recompute-full-first-success-20iters"

echo "=== SEQ_LEN sweep (Section 3.6) ==="
cp_log 2408449 "sweep-8b-bf16-sl1024"
cp_log 2408451 "sweep-8b-fp8-sl1024"
cp_log 2408452 "sweep-8b-bf16-sl2048"
cp_log 2408453 "sweep-8b-fp8-sl2048"
cp_log 2408454 "sweep-8b-bf16-sl4096"
cp_log 2408455 "sweep-8b-fp8-sl4096"

echo "=== Engineering findings (negative results) ==="
cp_log 2355471 "finding1-fp8-param-gather-incompat-with-PAOPT"
cp_log 2355480 "finding2-PAOPT-off-worse-OOM-than-default"
cp_log 2368462 "finding3-8b-fp8-vocab-parallel-CE-FP32-cast-OOM"
cp_log 2395774 "finding4-32b-fp8-memory-ceiling-NVTE-NO-CACHE-no-effect"

echo "=== Cluster contention / variance (Section 3.5) ==="
cp_log 2355470 "variance-32b-bf16-degraded-nid007044"
cp_log 2355474 "variance-32b-bf16-NCCL-crash-nid006998"

echo
echo "Done. Listing results/logs/:"
ls -la results/logs/
