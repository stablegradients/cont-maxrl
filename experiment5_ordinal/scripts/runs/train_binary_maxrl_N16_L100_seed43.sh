#!/bin/bash
# RL: binary_maxrl, N=16, L=100, seed=43
# Method: binary_maxrl | N: 16 | L: 100 | Seed: 43
# Expected runtime: ~25 min on A5000
set -euo pipefail

export WANDB_PROJECT="cont-maxrl-ordinal"
export WANDB_RUN_GROUP="binary_maxrl_L100_N16"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"

cd "$REPO_ROOT"

$PYTHON -m experiment5_ordinal.run \
    --method binary_maxrl \
    --N 16 \
    --L 100 \
    --seed 43 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --input_size 224 \
    --data_dir "$REPO_ROOT/data/utkface" \
    --output_dir "$REPO_ROOT/experiment5_ordinal/results" \
    --wandb
