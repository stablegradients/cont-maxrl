#!/bin/bash
#SBATCH --job-name=pkpo_K50_N256_seed44_k16_sgd
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/pkpo_K50_N256_seed44_k16_sgd_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="pkpo_K50_N256_k16_sgd"
export BF16=1
export PKPO_K=16

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl

# SGD: lr=0.01, momentum=0.9, wd=1e-4 (canonical ResNet-50 fine-tuning).
python -m experiment9_imagenet_localization.run \
    --method pkpo \
    --optimizer sgd \
    --momentum 0.9 \
    --K 50 \
    --N 256 \
    --seed 44 \
    --epochs 30 \
    --batch_size 128 \
    --lr 1e-2 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/pkpo_K50_N256_seed44_k16_sgd" \
    --num_workers 8 \
    --wandb
