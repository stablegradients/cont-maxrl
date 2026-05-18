#!/bin/bash
#SBATCH --job-name=gos_K50_N16384_seed43_sgd
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gos_K50_N16384_seed43_sgd_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N16384_sgd"

# Speedup stack: vectorized IoU + online metrics + bf16 + compile (default mode).
export BF16=1

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}

# SGD recipe: lr=0.01, momentum=0.9, wd=1e-4 — standard ResNet-50
# fine-tuning recipe at eff_batch=128 (Faster R-CNN / RetinaNet uses
# 0.02 from-scratch; 0.01 is the safer fine-tuning default).
python -m experiment9_imagenet_localization.run \
    --method gos \
    --optimizer sgd \
    --momentum 0.9 \
    --K 50 \
    --N 16384 \
    --seed 43 \
    --epochs 30 \
    --batch_size 128 \
    --lr 1e-2 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N16384_seed43_sgd" \
    --num_workers 8 \
    --wandb
