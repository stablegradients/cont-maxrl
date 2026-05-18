#!/bin/bash
#SBATCH --job-name=rloo_K50_N65536_seed43_4gpu
#SBATCH --partition=general
# A5000 or A6000 — exclude smaller cards (rtx_2080_ti=11G, v100=16G).
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/rloo_K50_N65536_seed43_4gpu_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="rloo_K50_N65536_4gpu_clamped"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + RANDOM % 20000))

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}
# torch.compile enabled (matches the exp_iou_ord_ce_clamped baseline for tighter parity)

torchrun \
    --nproc_per_node=4 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m experiment9_imagenet_localization.run \
        --method rloo \
        --K 50 \
        --N 65536 \
        --seed 43 \
        --epochs 30 \
        --batch_size 32 \
        --lr 5e-4 \
        --grad_clip 4.0 \
        --data_dir "/zfsauton/datasets/imagenet" \
        --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N65536_seed43_4gpu" \
        --num_workers 8 \
        --wandb
