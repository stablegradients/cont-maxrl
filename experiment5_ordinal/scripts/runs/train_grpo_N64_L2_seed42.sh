#!/bin/bash
# RL: grpo, N=64, L=2, seed=42
# Method: grpo | N: 64 | L: 2 | Seed: 42
# Expected runtime: ~25 min on A5000
set -euo pipefail

export WANDB_PROJECT="cont-maxrl-ordinal"
export WANDB_RUN_GROUP="grpo_L2_N64"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"

cd "$REPO_ROOT"

$PYTHON -m experiment5_ordinal.run \
    --method grpo \
    --N 64 \
    --L 2 \
    --seed 42 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --input_size 224 \
    --data_dir "$REPO_ROOT/data/utkface" \
    --output_dir "$REPO_ROOT/experiment5_ordinal/results" \
    --wandb
