#!/bin/bash
# LR ablation for GRPO and REINFORCE
# 2 methods × 3 lr × 3 N × 3 seeds = 54 runs
set -euo pipefail

export WANDB_ENTITY="stablegradients"
export WANDB_PROJECT="cont-maxrl-ordinal"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
RESULTS_DIR="$REPO_ROOT/experiment5_ordinal/results"
LOG_DIR="$REPO_ROOT/experiment5_ordinal/logs"

GPUS=(0 1 2 3)
JOBS_PER_GPU=2
MAX_JOBS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

METHODS=(grpo reinforce)
LRS=(1e-4 1e-5 1e-6)
N_VALUES=(16 64 256)
SEEDS=(42 43 44)

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

job_count=0
gpu_idx=0

run_job() {
    local method="$1" N="$2" seed="$3" lr="$4"
    local suffix="_lr${lr}"
    local run_name="${method}_L100_N${N}_seed${seed}${suffix}"
    local gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"
    local logfile="$LOG_DIR/train_${run_name}.log"

    if [ -f "$RESULTS_DIR/$run_name/metrics.json" ]; then
        echo "[$(date +%H:%M:%S)]        | SKIP   $run_name (done)"
        return
    fi

    echo "[$(date +%H:%M:%S)] GPU $gpu | START  $run_name"
    CUDA_VISIBLE_DEVICES="$gpu" $PYTHON -m experiment5_ordinal.run \
        --method "$method" --N "$N" --L 100 --seed "$seed" \
        --epochs 50 --batch_size 64 --lr "$lr" --input_size 224 \
        --data_dir "$REPO_ROOT/data/utkface" \
        --output_dir "$REPO_ROOT/experiment5_ordinal/results" \
        --run_suffix "$suffix" \
        --wandb > "$logfile" 2>&1 &

    gpu_idx=$(( gpu_idx + 1 ))
    job_count=$(( job_count + 1 ))
    if (( job_count >= MAX_JOBS )); then
        wait -n
        job_count=$(( job_count - 1 ))
    fi
}

echo "=========================================="
echo "  LR ablation: GRPO & REINFORCE"
echo "  54 runs (2 methods × 3 lr × 3 N × 3 seeds)"
echo "=========================================="

for lr in "${LRS[@]}"; do
    for method in "${METHODS[@]}"; do
        for N in "${N_VALUES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_job "$method" "$N" "$seed" "$lr"
            done
        done
    done
done
wait
echo "Done."
