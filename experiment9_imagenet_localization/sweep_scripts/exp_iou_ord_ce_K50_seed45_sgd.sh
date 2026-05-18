#!/bin/bash
#SBATCH --job-name=exp_iou_ord_ce_K50_seed45_sgd
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/exp_iou_ord_ce_K50_seed45_sgd_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="exp_iou_ord_ce_sgd"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl

python -m experiment9_imagenet_localization.run \
    --method exp_iou_ord_ce \
    --optimizer sgd \
    --momentum 0.9 \
    --K 50 \
    --seed 45 \
    --epochs 30 \
    --batch_size 128 \
    --lr 1e-2 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_seed45_sgd" \
    --num_workers 8 \
    --wandb
