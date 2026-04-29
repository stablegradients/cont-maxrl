#!/bin/bash
#SBATCH --job-name=bench_bf16_off
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/bench_bf16_off_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="bench_bf16_off"
export BF16=0

cd /zfsauton/scratch/shrinivr/cont-maxrl
source ~/.bashrc
conda activate cont-maxrl
export DISABLE_COMPILE=1

python -m experiment9_imagenet_localization.run \
    --method gos --K 50 --N 4096 --seed 99 \
    --epochs 1 --warmup_epochs 0 --batch_size 128 --lr 5e-4 \
    --train_subsample 100000 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/tmp/bench_bf16_off" \
    --num_workers 8 \
    --wandb
