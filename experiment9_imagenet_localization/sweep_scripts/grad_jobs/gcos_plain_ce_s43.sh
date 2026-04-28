#!/bin/bash
#SBATCH --job-name=gcos_plain_ce_s43
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gcos_plain_ce_s43_%j.out

set +u
cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
set -u
set -eo pipefail
PYBIN="${CONDA_PREFIX:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl}/bin/python"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/gradient_analysis"
echo "=== cosine  ref=plain_ce  plain_ce_K50_N64_seed43  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis cosine \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --K 50 \
    --seed 43 \
    --ref_method "plain_ce" \
    --methods "gos,binary_maxrl,grpo,rloo,reinforce,dg" \
    --Gs "4,8,16,32,64,128,256,512,1024,4096" \
    --n_images 200 \
    --n_trials 10 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/gradient_analysis"
echo "=== cosine  ref=plain_ce  plain_ce_K50_N64_seed43  epoch=10 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis cosine \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/checkpoints/epoch_10.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --K 50 \
    --seed 43 \
    --ref_method "plain_ce" \
    --methods "gos,binary_maxrl,grpo,rloo,reinforce,dg" \
    --Gs "4,8,16,32,64,128,256,512,1024,4096" \
    --n_images 200 \
    --n_trials 10 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/gradient_analysis"
echo "=== cosine  ref=plain_ce  plain_ce_K50_N64_seed43  epoch=25 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis cosine \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/checkpoints/epoch_25.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --K 50 \
    --seed 43 \
    --ref_method "plain_ce" \
    --methods "gos,binary_maxrl,grpo,rloo,reinforce,dg" \
    --Gs "4,8,16,32,64,128,256,512,1024,4096" \
    --n_images 200 \
    --n_trials 10 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/plain_ce_K50_N64_seed43/gradient_analysis"
