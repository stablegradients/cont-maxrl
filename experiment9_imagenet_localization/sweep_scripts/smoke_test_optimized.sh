#!/bin/bash
#SBATCH --job-name=smoke_optimized
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/smoke_optimized_%j.out

set -eo pipefail
cd /zfsauton/scratch/shrinivr/cont-maxrl
source ~/.bashrc
conda activate cont-maxrl

# Quick sanity: 1 epoch with subsample 1000, gos N=4096, single GPU.
# Verifies bf16 + torch.compile + fused-heads path works end-to-end.
python -m experiment9_imagenet_localization.run \
    --method gos \
    --K 50 \
    --N 4096 \
    --seed 99 \
    --epochs 2 \
    --warmup_epochs 0 \
    --batch_size 32 \
    --lr 5e-4 \
    --train_subsample 1000 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/tmp/smoke_optimized_run" \
    --num_workers 4
