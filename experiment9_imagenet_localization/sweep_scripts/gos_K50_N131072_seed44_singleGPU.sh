#!/bin/bash
#SBATCH --job-name=gos_K50_N131072_seed44_singleGPU
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gos_K50_N131072_seed44_singleGPU_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N131072_singleGPU"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export DISABLE_COMPILE=1

python -m experiment9_imagenet_localization.run \
    --method gos \
    --K 50 \
    --N 131072 \
    --seed 44 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N131072_seed44_singleGPU" \
    --num_workers 8 \
    --wandb
