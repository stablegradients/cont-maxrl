#!/bin/bash
#SBATCH --job-name=rloo_K50_N16_seed45
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/rloo_K50_N16_seed45_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="rloo_K50_N16_clamped"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}

python -m experiment9_imagenet_localization.run \
    --method rloo \
    --K 50 \
    --N 16 \
    --seed 45 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --grad_clip 4.0 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N16_seed45" \
    --num_workers 8 \
    --wandb
