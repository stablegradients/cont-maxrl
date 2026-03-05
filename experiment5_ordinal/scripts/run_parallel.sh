#!/bin/bash
# Parallel launcher: distributes all experiment jobs across GPUs 0,1,2
# Runs JOBS_PER_GPU concurrent jobs per GPU (default: 2 → 6 total)
#
# Usage:
#   bash experiment5_ordinal/scripts/run_parallel.sh
#   JOBS_PER_GPU=1 bash experiment5_ordinal/scripts/run_parallel.sh   # conservative
set -euo pipefail

export WANDB_ENTITY="stablegradients"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
SCRIPT_DIR="$REPO_ROOT/experiment5_ordinal/scripts/runs"
LOG_DIR="$REPO_ROOT/experiment5_ordinal/logs"
RESULTS_DIR="$REPO_ROOT/experiment5_ordinal/results"

GPUS=(0 1 2 3)
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
MAX_JOBS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

# ─── Helper: run up to MAX_JOBS in parallel ───
job_count=0
gpu_idx=0

run_job() {
    local script="$1"
    local gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
    local name="$(basename "$script" .sh)"
    local logfile="$LOG_DIR/${name}.log"

    echo "[$(date +%H:%M:%S)] GPU $gpu | START  $name"
    CUDA_VISIBLE_DEVICES="$gpu" bash "$script" > "$logfile" 2>&1 &

    gpu_idx=$(( gpu_idx + 1 ))
    job_count=$(( job_count + 1 ))

    # If we've hit the max, wait for one to finish
    if (( job_count >= MAX_JOBS )); then
        wait -n  # wait for ANY one background job to finish
        job_count=$(( job_count - 1 ))
    fi
}

wait_all() {
    wait
    job_count=0
}

# ─── Phase 1: Supervised (L=100) — 15 runs, ~10 min each ───
echo ""
echo "=========================================="
echo "  Phase 1: Supervised training (L=100)"
echo "  15 runs (3 methods × 5 seeds)"
echo "=========================================="
phase1_start=$(date +%s)

for script in "$SCRIPT_DIR"/train_{ordinal_ce,plain_ce,mse}_L100_seed*.sh; do
    run_job "$script"
done
wait_all

phase1_end=$(date +%s)
echo "Phase 1 done in $(( (phase1_end - phase1_start) / 60 )) min"

# ─── Phase 2: RL (L=100) — 100 runs, ~25 min each ───
echo ""
echo "=========================================="
echo "  Phase 2: RL training (L=100)"
echo "  100 runs (4 methods × 5 N × 5 seeds)"
echo "=========================================="
phase2_start=$(date +%s)

# Run smallest N first (faster, verify pipeline works)
for N in 16 64 256 1024 4096; do
    for method in cont_maxrl binary_maxrl grpo reinforce; do
        for script in "$SCRIPT_DIR"/train_${method}_N${N}_L100_seed*.sh; do
            [ -f "$script" ] && run_job "$script"
        done
    done
done
wait_all

phase2_end=$(date +%s)
echo "Phase 2 done in $(( (phase2_end - phase2_start) / 60 )) min"

# ─── Phase 3: Binary recovery (L=2) — 50 runs, ~10 min each ───
echo ""
echo "=========================================="
echo "  Phase 3: Binary recovery (L=2)"
echo "  50 runs"
echo "=========================================="
phase3_start=$(date +%s)

# Supervised L=2
for script in "$SCRIPT_DIR"/train_{ordinal_ce,plain_ce}_L2_seed*.sh; do
    [ -f "$script" ] && run_job "$script"
done
# RL L=2
for script in "$SCRIPT_DIR"/train_*_N*_L2_seed*.sh; do
    [ -f "$script" ] && run_job "$script"
done
wait_all

phase3_end=$(date +%s)
echo "Phase 3 done in $(( (phase3_end - phase3_start) / 60 )) min"

# ─── Phase 4: Gradient analysis — 20 jobs ───
echo ""
echo "=========================================="
echo "  Phase 4: Gradient analysis"
echo "  5 methods × 4 epochs (seed=42 only)"
echo "=========================================="
phase4_start=$(date +%s)

METHODS="ordinal_ce plain_ce cont_maxrl grpo reinforce"
EPOCHS="1 10 25 50"

for METHOD in $METHODS; do
    for EPOCH in $EPOCHS; do
        if [ "$METHOD" = "ordinal_ce" ] || [ "$METHOD" = "plain_ce" ]; then
            RUN_NAME="${METHOD}_L100_N0_seed42"
        else
            RUN_NAME="${METHOD}_L100_N64_seed42"
        fi
        CKPT="$RESULTS_DIR/$RUN_NAME/checkpoints/epoch_${EPOCH}.pt"
        if [ ! -f "$CKPT" ]; then
            echo "  SKIP: $CKPT not found"
            continue
        fi
        local_gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
        logfile="$LOG_DIR/grad_${METHOD}_epoch${EPOCH}.log"
        echo "[$(date +%H:%M:%S)] GPU $local_gpu | GRAD  $METHOD epoch=$EPOCH"
        CUDA_VISIBLE_DEVICES="$local_gpu" $PYTHON -m experiment5_ordinal.gradient_analysis \
            --checkpoint "$CKPT" \
            --data_dir "$REPO_ROOT/data/utkface" \
            --method "$METHOD" \
            --L 100 \
            --seed 42 \
            --output_dir "$RESULTS_DIR/$RUN_NAME/gradient_analysis" \
            --n_images 500 > "$logfile" 2>&1 &
        gpu_idx=$(( gpu_idx + 1 ))
        job_count=$(( job_count + 1 ))
        if (( job_count >= MAX_JOBS )); then
            wait -n
            job_count=$(( job_count - 1 ))
        fi
    done
done
wait_all

phase4_end=$(date +%s)
echo "Phase 4 done in $(( (phase4_end - phase4_start) / 60 )) min"

# ─── Phase 5: Generate figures ───
echo ""
echo "=========================================="
echo "  Phase 5: Generate figures"
echo "=========================================="

$PYTHON -m experiment5_ordinal.plot \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/figures"

# ─── Summary ───
total_end=$(date +%s)
echo ""
echo "=========================================="
echo "  All done!"
echo "  Total time: $(( (total_end - phase1_start) / 60 )) min"
echo "  Results:  $RESULTS_DIR"
echo "  Figures:  $RESULTS_DIR/figures"
echo "  Logs:     $LOG_DIR"
echo "=========================================="

# Check for failures
failed=0
for log in "$LOG_DIR"/*.log; do
    if ! tail -1 "$log" | grep -q "Metrics saved\|complete\|saved"; then
        echo "  FAILED: $(basename "$log")"
        failed=$(( failed + 1 ))
    fi
done
if (( failed > 0 )); then
    echo "  $failed jobs may have failed — check logs"
else
    echo "  All jobs succeeded"
fi
