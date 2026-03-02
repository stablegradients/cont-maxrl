#!/bin/bash
# RL: cont_maxrl_no_bl, N=4096, L=100, seed=43
# Method: cont_maxrl_no_bl | N: 4096 | L: 100 | Seed: 43
# Expected runtime: ~25 min on A5000
set -euo pipefail

export WANDB_PROJECT="cont-maxrl-ordinal"
export WANDB_RUN_GROUP="cont_maxrl_no_bl_L100_N4096"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"

cd "$REPO_ROOT"

$PYTHON -m experiment5_ordinal.run \
    --method cont_maxrl_no_bl \
    --N 4096 \
    --L 100 \
    --seed 43 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --input_size 224 \
    --data_dir "$REPO_ROOT/data/utkface" \
    --output_dir "$REPO_ROOT/experiment5_ordinal/results" \
    --wandb
