#!/bin/bash
#SBATCH --job-name=j_0aff8ff8
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/j_0aff8ff8_%j.out

set -euo pipefail

# wandb reads credentials from ~/.netrc.
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="mse_iou_match_K50_N64"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate base

python -m experiment9_imagenet_localization.run \
    --method mse_iou_match \
    --K 50 \
    --N 64 \
    --seed 42 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/mse_iou_match_K50_N64_seed42" \
    --num_workers 8 \
    --wandb
