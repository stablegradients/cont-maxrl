#!/bin/bash
#SBATCH --job-name=j_5c9e87b6
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu24,gpu25,gpu26,gpu28,gpu29,gpu30,gpu31
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=experiment9_imagenet_localization/sweep_scripts/logs/j_5c9e87b6_%j.out

set -euo pipefail

export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N256"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate base

python -m experiment9_imagenet_localization.run \
    --method gos \
    --K 50 \
    --N 256 \
    --seed 42 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N256_seed42" \
    --num_workers 8 \
    --wandb
