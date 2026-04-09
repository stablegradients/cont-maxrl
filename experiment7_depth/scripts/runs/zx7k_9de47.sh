#!/bin/bash
# Job: zx7k_9de47 | Method: reinforce | K: 100 | G: 4 | Seed: 42
set -euo pipefail
export WANDB_PROJECT="cont-maxrl-depth"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON_BIN:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python}"

cd "$REPO_ROOT"

# 4-GPU DDP launch via torchrun. CUDA_VISIBLE_DEVICES set by submit_all.sh.
torchrun --standalone --nproc_per_node=4 \
    -m experiment7_depth.run \
    --method reinforce \
    --K 100 \
    --G 4 \
    --seed 42 \
    --global_batch_size 32 \
    --epochs 25 \
    --epochs_hard_cap 35 \
    --epochs_min_before_es 18 \
    --early_stop_patience 8 \
    --lr 2e-4 \
    --lr_schedule polynomial \
    --lr_power 0.9 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --amp \
    --data_dir "$REPO_ROOT/data/nyu" \
    --output_dir "$REPO_ROOT/experiment7_depth/results/zx7k_9de47" \
    --wandb \
    --wandb_run_name "zx7k_9de47"
