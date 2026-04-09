#!/bin/bash
# Quick sanity check: run 3 small jobs for 2 epochs each on single GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON_BIN:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python}"
OUT="/tmp/sanity_depth_$$"
mkdir -p "$OUT"

cd "$REPO_ROOT"

echo "=== Sanity check 1: ordinal_ce K=100 ==="
$PYTHON -m experiment7_depth.run \
    --method ordinal_ce --K 100 --seed 42 \
    --global_batch_size 8 --epochs 2 --epochs_hard_cap 2 \
    --lr 2e-4 --momentum 0.9 --weight_decay 5e-4 --amp \
    --data_dir "$REPO_ROOT/data/nyu" --output_dir "$OUT/ordinal_ce" \
    --no_wandb --num_workers 2

echo "=== Sanity check 2: cont_maxrl K=10 G=4 ==="
$PYTHON -m experiment7_depth.run \
    --method cont_maxrl --K 10 --G 4 --seed 42 \
    --global_batch_size 8 --epochs 2 --epochs_hard_cap 2 \
    --lr 2e-4 --momentum 0.9 --weight_decay 5e-4 --amp \
    --data_dir "$REPO_ROOT/data/nyu" --output_dir "$OUT/cont_maxrl" \
    --no_wandb --num_workers 2

echo "=== Sanity check 3: plain_ce K=10 ==="
$PYTHON -m experiment7_depth.run \
    --method plain_ce --K 10 --seed 42 \
    --global_batch_size 8 --epochs 2 --epochs_hard_cap 2 \
    --lr 2e-4 --momentum 0.9 --weight_decay 5e-4 --amp \
    --data_dir "$REPO_ROOT/data/nyu" --output_dir "$OUT/plain_ce" \
    --no_wandb --num_workers 2

# Verify outputs
for run in ordinal_ce cont_maxrl plain_ce; do
    if [[ -f "$OUT/$run/metrics.json" ]]; then
        echo "  $run: metrics.json exists"
    else
        echo "  ERROR: $run/metrics.json missing!"
        exit 1
    fi
done

echo "=== All sanity checks passed ==="
rm -rf "$OUT"
