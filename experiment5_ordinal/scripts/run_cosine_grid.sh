#!/bin/bash
# Compute G × epoch × reference grid of cosine similarity and magnitude ratio.
# 3 references × 4 epochs = 12 runs, ~8 min each ≈ ~25 min with 4 parallel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
RESULTS_DIR="$REPO_ROOT/experiment5_ordinal/results"

GPUS=(${GPUS:-1 2 3})
N_IMAGES="${N_IMAGES:-200}"
N_TRIALS="${N_TRIALS:-10}"

REFS=(ordinal_ce plain_ce mse)
EPOCHS=(1 10 25 50)

cd "$REPO_ROOT"

job_count=0
gpu_idx=0
MAX_JOBS=${#GPUS[@]}

run_job() {
    local ref="$1" epoch="$2"
    local gpu="${GPUS[$(( gpu_idx % ${#GPUS[@]} ))]}"

    # MSE uses ordinal_ce model (MSE model has 1-output regression head,
    # can't do RL sampling). MSE gradient is computed on expected value
    # from softmax distribution.
    local model_ref="$ref"
    [ "$ref" = "mse" ] && model_ref="ordinal_ce"

    local ckpt="$RESULTS_DIR/${model_ref}_L100_N0_seed42/checkpoints/epoch_${epoch}.pt"
    local out_dir="$RESULTS_DIR/${model_ref}_L100_N0_seed42/gradient_analysis"
    local suffix=""
    [ "$ref" != "ordinal_ce" ] && suffix="_ref_${ref}"
    local out_file="$out_dir/cosine_vs_G_epoch${epoch}${suffix}.json"

    if [ -f "$out_file" ]; then
        echo "[$(date +%H:%M:%S)]        | SKIP  ref=$ref epoch=$epoch (done)"
        return
    fi
    if [ ! -f "$ckpt" ]; then
        echo "[$(date +%H:%M:%S)]        | SKIP  ref=$ref epoch=$epoch (no checkpoint)"
        return
    fi

    echo "[$(date +%H:%M:%S)] GPU $gpu | START ref=$ref epoch=$epoch"
    CUDA_VISIBLE_DEVICES="$gpu" $PYTHON -m experiment5_ordinal.gradient_analysis \
        --mode cosine \
        --checkpoint "$ckpt" \
        --data_dir "$REPO_ROOT/data/utkface" \
        --L 100 --seed 42 \
        --output_dir "$out_dir" \
        --n_images "$N_IMAGES" \
        --n_trials "$N_TRIALS" \
        --ref_method "$ref" &

    gpu_idx=$(( gpu_idx + 1 ))
    job_count=$(( job_count + 1 ))
    if (( job_count >= MAX_JOBS )); then
        wait -n
        job_count=$(( job_count - 1 ))
    fi
}

echo "==========================================="
echo "  Cosine grid: 3 refs × 4 epochs = 12 runs"
echo "  GPUs: ${GPUS[*]}"
echo "  N_IMAGES=$N_IMAGES  N_TRIALS=$N_TRIALS"
echo "==========================================="

for ref in "${REFS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        run_job "$ref" "$epoch"
    done
done
wait
echo "Done. Now run: $PYTHON -m experiment5_ordinal.plot --results_dir $RESULTS_DIR --figures 7"
