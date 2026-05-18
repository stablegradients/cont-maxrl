#!/bin/bash
#SBATCH --job-name=gos_K50_N65536_seed43_fixedN
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gos_K50_N65536_seed43_fixedN_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N65536_fixedN"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + RANDOM % 20000))

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}

torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m experiment9_imagenet_localization.run \
        --method gos \
        --K 50 \
        --N 65536 \
        --seed 43 \
        --epochs 30 \
        --batch_size 16 \
        --lr 5e-4 \
        --data_dir "/zfsauton/datasets/imagenet" \
        --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N65536_seed43_fixedN" \
        --num_workers 8 \
        --wandb
