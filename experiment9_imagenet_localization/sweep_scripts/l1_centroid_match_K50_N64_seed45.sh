#!/bin/bash
#SBATCH --job-name=j_3f9bb160
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/j_3f9bb160_%j.out

# `set -u` would be nice but breaks conda activation: some conda env-activate
# scripts (e.g. activate-binutils_linux-64.sh) reference unbound vars like
# ADDR2LINE and hard-fail under -u. We keep -e + -o pipefail for safety.
set -eo pipefail

# wandb reads credentials from ~/.netrc (written by `wandb login`), so no
# WANDB_API_KEY env export is needed. The earlier strict `${VAR:?...}` check
# was killing jobs before they started.
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="l1_centroid_match_K50_N64"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate cont-maxrl

python -m experiment9_imagenet_localization.run \
    --method l1_centroid_match \
    --K 50 \
    --N 64 \
    --seed 45 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/l1_centroid_match_K50_N64_seed45" \
    --num_workers 8 \
    --wandb
