#!/bin/bash
# Compute G vs cosine similarity between RL and ordinal CE gradients.
# Uses the ordinal_ce checkpoint (epoch 25) as reference.
# Runs on a single GPU, takes ~5-10 min.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
RESULTS_DIR="$REPO_ROOT/experiment5_ordinal/results"

GPU="${1:-0}"
EPOCH="${2:-25}"
N_IMAGES="${3:-200}"
N_TRIALS="${4:-10}"

CKPT="$RESULTS_DIR/ordinal_ce_L100_N0_seed42/checkpoints/epoch_${EPOCH}.pt"
OUTPUT_DIR="$RESULTS_DIR/ordinal_ce_L100_N0_seed42/gradient_analysis"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

echo "==========================================="
echo "  Cosine analysis: G vs cos(RL, Ord CE)"
echo "  Epoch: $EPOCH"
echo "  Images: $N_IMAGES, Trials: $N_TRIALS"
echo "  GPU: $GPU"
echo "==========================================="

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -m experiment5_ordinal.gradient_analysis \
    --mode cosine \
    --checkpoint "$CKPT" \
    --data_dir "$REPO_ROOT/data/utkface" \
    --L 100 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --n_images "$N_IMAGES" \
    --n_trials "$N_TRIALS"

echo "Done. Now run: $PYTHON -m experiment5_ordinal.plot --results_dir $RESULTS_DIR --figures 6"
