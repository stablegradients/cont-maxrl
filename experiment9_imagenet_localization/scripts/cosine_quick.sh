#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=cos_quick
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/scripts/logs/cos_quick_%j.out

set -eo pipefail
mkdir -p /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/scripts/logs
cd /zfsauton/scratch/shrinivr/cont-maxrl
source ~/.bashrc
conda activate cont-maxrl

python -u experiment9_imagenet_localization/scripts/cosine_quick.py \
    --Gs 1024,16384,262144 \
    --n_images 8 \
    --seed 42
