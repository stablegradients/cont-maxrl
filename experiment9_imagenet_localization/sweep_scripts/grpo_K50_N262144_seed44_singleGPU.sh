#!/bin/bash
#SBATCH --job-name=grpo_K50_N262144_seed44_singleGPU
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/grpo_K50_N262144_seed44_singleGPU_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="grpo_K50_N262144_singleGPU_clamped"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}
# torch.compile enabled (matches the exp_iou_ord_ce_clamped baseline for tighter parity)

python -m experiment9_imagenet_localization.run \
    --method grpo \
    --K 50 \
    --N 262144 \
    --seed 44 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --grad_clip 4.0 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N262144_seed44_singleGPU" \
    --num_workers 8 \
    --wandb
