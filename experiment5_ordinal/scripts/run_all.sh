#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Run ALL ordinal regression experiments from scratch.
#
# Phase 1: Supervised baselines (L=100)       —   9 runs
# Phase 2: RL methods (L=100)                 —  45 runs
# Phase 3: Binary recovery (L=2)              —  33 runs
# Phase 4: Gradient analysis                  —  24 jobs
# Phase 5: Generate figures
#
# Total training runs: 87
#
# Usage:
#   nohup bash experiment5_ordinal/scripts/run_all.sh \
#       > experiment5_ordinal/logs/run_all.log 2>&1 &
#
# Options (env vars):
#   JOBS_PER_GPU=2   — concurrent jobs per GPU (default 2)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

export WANDB_ENTITY="stablegradients"
export WANDB_PROJECT="cont-maxrl-ordinal"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
SCRIPT_DIR="$REPO_ROOT/experiment5_ordinal/scripts/runs"
LOG_DIR="$REPO_ROOT/experiment5_ordinal/logs"
RESULTS_DIR="$REPO_ROOT/experiment5_ordinal/results"

GPUS=(0 1 3 4)
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
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

    # Skip if metrics.json already exists (run completed)
    local run_name="${name#train_}"
    if [[ "$run_name" != *_N*_* ]]; then
        run_name="${run_name%%_seed*}_N0_seed${run_name##*seed}"
    fi
    if [ -f "$RESULTS_DIR/$run_name/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)]        | SKIP   $name (done)"
        return
    fi

    echo "[$(date +%H:%M:%S)] GPU $gpu | START  $name"
    CUDA_VISIBLE_DEVICES="$gpu" bash "$script" > "$logfile" 2>&1 &

    gpu_idx=$(( gpu_idx + 1 ))
    job_count=$(( job_count + 1 ))

    if (( job_count >= MAX_JOBS )); then
        wait -n
        job_count=$(( job_count - 1 ))
    fi
}

wait_all() {
    wait
    job_count=0
}

overall_start=$(date +%s)

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Supervised baselines (L=100)
#   3 methods × 3 seeds = 9 runs
#   Methods: ordinal_ce, plain_ce, mse
#   All use lr=1e-3, SGD+cosine, 50 epochs
#   ~10 min each
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "  Phase 1: Supervised baselines (L=100)"
echo "  9 runs (3 methods × 3 seeds)"
echo "=========================================="
phase1_start=$(date +%s)

for method in ordinal_ce plain_ce mse; do
    for script in "$SCRIPT_DIR"/train_${method}_L100_seed*.sh; do
        [ -f "$script" ] && run_job "$script"
    done
done
wait_all

phase1_end=$(date +%s)
echo "Phase 1 done in $(( (phase1_end - phase1_start) / 60 )) min"

# ═══════════════════════════════════════════════════════════════════
# Phase 2: RL methods (L=100)
#   5 methods × 5 N values × 3 seeds = 75 runs
#   Methods:
#     cont_maxrl       — gap-over-survivors with baseline (our method)
#     cont_maxrl_no_bl — gap-over-survivors WITHOUT baseline subtraction
#     binary_maxrl     — Tajwar et al. binarized at tau=1.0
#     grpo             — z-score normalized
#     reinforce        — mean-centered baseline
#   N: 16, 64, 256
#   Seeds: 42, 43, 44
#   All use lr=1e-3, SGD+cosine, 50 epochs
#   ~25 min each
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "  Phase 2: RL training (L=100)"
echo "  45 runs (5 methods × 3 N × 3 seeds)"
echo "=========================================="
phase2_start=$(date +%s)

for N in 16 64 256; do
    for method in cont_maxrl cont_maxrl_no_bl binary_maxrl grpo reinforce; do
        for script in "$SCRIPT_DIR"/train_${method}_N${N}_L100_seed*.sh; do
            [ -f "$script" ] && run_job "$script"
        done
    done
done
wait_all

phase2_end=$(date +%s)
echo "Phase 2 done in $(( (phase2_end - phase2_start) / 60 )) min"

# ═══════════════════════════════════════════════════════════════════
# Phase 3: Binary recovery (L=2)
#   Supervised: ordinal_ce, plain_ce × 3 seeds = 6 runs
#   RL at N=64: 5 RL methods × 3 seeds = 15 runs
#   RL at higher N (binary_maxrl, cont_maxrl × N=256,1024 × 3 seeds) = 12 runs
#   Total: 33 runs
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "  Phase 3: Binary recovery (L=2)"
echo "  33 runs"
echo "=========================================="
phase3_start=$(date +%s)

# Supervised L=2
for script in "$SCRIPT_DIR"/train_{ordinal_ce,plain_ce}_L2_seed*.sh; do
    [ -f "$script" ] && run_job "$script"
done
# RL L=2 (all methods/N combos)
for script in "$SCRIPT_DIR"/train_*_N*_L2_seed*.sh; do
    [ -f "$script" ] && run_job "$script"
done
wait_all

phase3_end=$(date +%s)
echo "Phase 3 done in $(( (phase3_end - phase3_start) / 60 )) min"

# ═══════════════════════════════════════════════════════════════════
# Phase 4: Gradient analysis
#   6 methods × 4 epochs = 24 jobs (seed=42 only)
#   Methods: ordinal_ce, plain_ce, cont_maxrl, cont_maxrl_no_bl, grpo, reinforce
#   Epochs: 1, 10, 25, 50
#   Compares RL gradients to supervised CE gradient at each checkpoint
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "  Phase 4: Gradient analysis"
echo "  6 methods × 4 epochs = 24 jobs"
echo "=========================================="
phase4_start=$(date +%s)

METHODS="ordinal_ce plain_ce cont_maxrl cont_maxrl_no_bl grpo reinforce"
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

# ═══════════════════════════════════════════════════════════════════
# Phase 5: Generate figures
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "  Phase 5: Generate figures"
echo "=========================================="

$PYTHON -m experiment5_ordinal.plot \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/figures"

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
overall_end=$(date +%s)
echo ""
echo "=========================================="
echo "  All done!"
echo "  Total time: $(( (overall_end - overall_start) / 60 )) min"
echo "  Results:  $RESULTS_DIR"
echo "  Figures:  $RESULTS_DIR/figures"
echo "  Logs:     $LOG_DIR"
echo "=========================================="

# Check for failures
failed=0
for log in "$LOG_DIR"/train_*.log; do
    if [ -f "$log" ] && ! tail -1 "$log" | grep -q "Metrics saved\|complete\|saved"; then
        echo "  FAILED: $(basename "$log")"
        failed=$(( failed + 1 ))
    fi
done
if (( failed > 0 )); then
    echo "  $failed jobs may have failed — check logs"
else
    echo "  All jobs succeeded"
fi
