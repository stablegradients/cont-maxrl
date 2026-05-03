#!/bin/bash
#SBATCH --job-name=gos_K50_N65536_seed45_singleGPU
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gos_K50_N65536_seed45_singleGPU_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N65536_singleGPU"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl

# Speedup stack: bf16 autocast (~1.4×, gradient cos verified ≈ 1.0 vs fp32)
# + torch.compile mode='default' (~1.3×) + channels_last layout (~1.5×,
# baked into model.to() inside run.py)
export BF16=1
export DISABLE_COMPILE=1
# Compile is enabled by default; set DISABLE_COMPILE=1 to turn off.

python -m experiment9_imagenet_localization.run \
    --method gos \
    --K 50 \
    --N 65536 \
    --seed 45 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N65536_seed45_singleGPU" \
    --num_workers 8 \
    --wandb
