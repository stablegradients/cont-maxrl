#!/bin/bash
#SBATCH --job-name=j_79eab191
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/j_79eab191_%j.out

set -euo pipefail

# wandb reads credentials from ~/.netrc.
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="exp_iou_ord_ce_K50_N64"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate base

# DDP launch: 4 GPUs on one node via torchrun.
# Per-GPU batch = 32  ⇒  effective total batch = 128 (matches single-GPU baselines).
torchrun \
    --nproc_per_node=4 \
    --standalone \
    -m experiment9_imagenet_localization.run \
    --method exp_iou_ord_ce \
    --K 50 \
    --N 64 \
    --seed 42 \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed42" \
    --num_workers 8 \
    --wandb
