#!/bin/bash
# Supervised: ordinal_ce, L=100, seed=44
# Method: ordinal_ce | N: 0 | L: 100 | Seed: 44
# Expected runtime: ~10 min on A5000
set -euo pipefail

export WANDB_PROJECT="cont-maxrl-ordinal"
export WANDB_RUN_GROUP="ordinal_ce_L100_N0"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"

cd "$REPO_ROOT"

$PYTHON -m experiment5_ordinal.run \
    --method ordinal_ce \
    --N 0 \
    --L 100 \
    --seed 44 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --input_size 224 \
    --data_dir "$REPO_ROOT/data/utkface" \
    --output_dir "$REPO_ROOT/experiment5_ordinal/results" \
    --wandb
