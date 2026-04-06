#!/bin/bash
set -euo pipefail
REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== Generating RL data (50K train + 512 val) ==="
$PYTHON -m experiment6_maze.src.data_prep \
  --size 17 --algorithm prim \
  --n_train 50000 --n_val 512 \
  --train_seed_start 0 --val_seed_start 60000 \
  --l_max 64 \
  --output_dir "${REPO_ROOT}/data/maze_17x17"

echo "=== Generating SFT data (1M train + 256 val) ==="
$PYTHON -m experiment6_maze.src.sft_data_prep \
  --size 17 \
  --n_train 1000000 --n_val 256 \
  --train_seed_start 0 --val_seed_start 1100000 \
  --output_dir "${REPO_ROOT}/data/maze_17x17"

echo "=== Done ==="
ls -lh "${REPO_ROOT}/data/maze_17x17/"
