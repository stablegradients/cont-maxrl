#!/bin/bash
#SBATCH --job-name=j_0384a761
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/j_0384a761_%j.out

set -euo pipefail

# wandb reads credentials from ~/.netrc (written by `wandb login`), so no
# WANDB_API_KEY env export is needed. The earlier strict `${VAR:?...}` check
# was killing jobs before they started.
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="rloo_K100_N64"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate base

python -m experiment9_imagenet_localization.run \
    --method rloo \
    --K 100 \
    --N 64 \
    --seed 42 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K100_N64_seed42" \
    --num_workers 8 \
    --wandb
