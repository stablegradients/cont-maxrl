#!/bin/bash
#SBATCH --job-name=gos_K50_N4096_seed44_matchJord
#SBATCH --partition=general
# Accept either a6000 or a5000 — both have enough VRAM for these configs.
# Exclude smaller cards (rtx_2080_ti = 11GB, v100 = 16GB → too small).
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gos_K50_N4096_seed44_matchJord_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N4096_matchJord"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + RANDOM % 20000))

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl

torchrun \
    --nproc_per_node=4 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m experiment9_imagenet_localization.run \
        --method gos \
        --K 50 \
        --N 4096 \
        --seed 44 \
        --epochs 30 \
        --batch_size 32 \
        --lr 5e-4 \
        --sample_floor 0.005 \
        --data_dir "/zfsauton/datasets/imagenet" \
        --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N4096_seed44_matchJord" \
        --num_workers 8 \
        --wandb
