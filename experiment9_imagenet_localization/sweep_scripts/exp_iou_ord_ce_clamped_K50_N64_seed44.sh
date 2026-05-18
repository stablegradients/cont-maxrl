#!/bin/bash
#SBATCH --job-name=j_clamped_seed44
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu24,gpu25,gpu26,gpu28,gpu29,gpu30,gpu31
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/j_clamped_seed44_%j.out

set -eo pipefail

export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="exp_iou_ord_ce_clamped"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate base

# DDP launch: 4 GPUs on one node (a5000 or a6000; both sm_86 Ampere).
# Per-GPU batch = 32  ⇒  effective total batch = 128.
torchrun \
    --nproc_per_node=4 \
    --standalone \
    -m experiment9_imagenet_localization.run \
    --method exp_iou_ord_ce_clamped \
    --K 50 \
    --N 64 \
    --seed 44 \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-4 \
    --grad_clip 4.0 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_clamped_K50_N64_seed44" \
    --num_workers 8 \
    --wandb
