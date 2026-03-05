#!/bin/bash
# Run cont_maxrl_no_bl (no baseline subtraction) across all N values.
# 4 GPUs, 4 jobs per GPU = 16 concurrent.
#
# Usage:
#   nohup bash experiment5_ordinal/scripts/run_no_baseline.sh > experiment5_ordinal/logs/run_no_bl.log 2>&1 &
set -euo pipefail

export WANDB_ENTITY="stablegradients"
export WANDB_PROJECT="cont-maxrl-ordinal"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
LOG_DIR="$REPO_ROOT/experiment5_ordinal/logs"

GPUS=(0 1 2 3)
JOBS_PER_GPU=4
MAX_JOBS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

SEEDS=(42)
N_VALUES=(256 1024 4096)
METHOD="cont_maxrl_no_bl"
LR="1e-3"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

job_count=0
gpu_idx=0

run_job() {
    local N=$1
    local seed=$2
    local gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
    local name="train_${METHOD}_N${N}_L100_seed${seed}"
    local logfile="$LOG_DIR/${name}.log"

    echo "[$(date +%H:%M:%S)] GPU $gpu | START  $name"
    CUDA_VISIBLE_DEVICES="$gpu" $PYTHON -m experiment5_ordinal.run \
        --method $METHOD \
        --N $N \
        --L 100 \
        --seed $seed \
        --epochs 50 \
        --batch_size 64 \
        --lr $LR \
        --input_size 224 \
        --data_dir "$REPO_ROOT/data/utkface" \
        --output_dir "$REPO_ROOT/experiment5_ordinal/results" \
        --wandb > "$logfile" 2>&1 &

    gpu_idx=$(( gpu_idx + 1 ))
    job_count=$(( job_count + 1 ))

    if (( job_count >= MAX_JOBS )); then
        wait -n
        job_count=$(( job_count - 1 ))
    fi
}

echo ""
echo "=========================================="
echo "  cont_maxrl_no_bl: 3 N × 1 seed = 3 runs"
echo "=========================================="
t0=$(date +%s)

for N in "${N_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_job $N $seed
    done
done
wait

t1=$(date +%s)
echo ""
echo "All done in $(( (t1 - t0) / 60 )) min"

# Check for failures
failed=0
for N in "${N_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log="$LOG_DIR/train_${METHOD}_N${N}_L100_seed${seed}.log"
        if [ -f "$log" ] && ! tail -1 "$log" | grep -q "Metrics saved\|complete\|saved"; then
            echo "  FAILED: $(basename "$log")"
            failed=$(( failed + 1 ))
        fi
    done
done
if (( failed > 0 )); then
    echo "  $failed jobs may have failed"
else
    echo "  All 15 jobs succeeded"
fi
