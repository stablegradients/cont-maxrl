#!/bin/bash
#SBATCH --job-name=epoch0_iou
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/scripts/logs/epoch0_iou_%j.out

set -eo pipefail
mkdir -p /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/scripts/logs
cd /zfsauton/scratch/shrinivr/cont-maxrl
source ~/.bashrc
conda activate cont-maxrl

python -u experiment9_imagenet_localization/scripts/compute_epoch0_iou.py
