#!/bin/bash
#SBATCH --job-name=cosine_xl
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/cosine_xl_%j.out

set -eo pipefail
cd /zfsauton/scratch/shrinivr/cont-maxrl
source ~/.bashrc
conda activate cont-maxrl

mkdir -p /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/cosine_check_jord_xl

python -m experiment9_imagenet_localization.gradient_analysis cosine \
    --checkpoint /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed43/checkpoints/epoch_25.pt \
    --K 50 \
    --seed 43 \
    --ref_method exp_iou_ord_ce \
    --methods "gos" \
    --Gs "65536,131072,262144" \
    --n_images 50 \
    --n_trials 2 \
    --output_dir /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/cosine_check_jord_xl
