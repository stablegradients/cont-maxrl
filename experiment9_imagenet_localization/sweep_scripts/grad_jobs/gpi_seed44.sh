#!/bin/bash
#SBATCH --job-name=gpi_seed44
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gpi_seed44_%j.out

set +u
cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
set -u
set -eo pipefail
PYBIN="${CONDA_PREFIX:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl}/bin/python"

echo "### per_image  seed=44  Mon Apr 27 12:44:17 PM EDT 2026 ###"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/gradient_analysis"
echo "=== per_image  gos_K50_N64_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "gos" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/gradient_analysis"
echo "=== per_image  gos_K50_N64_seed44  epoch=10 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/checkpoints/epoch_10.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "gos" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/gradient_analysis"
echo "=== per_image  gos_K50_N64_seed44  epoch=25 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/checkpoints/epoch_25.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "gos" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N64_seed44/gradient_analysis"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/gradient_analysis"
echo "=== per_image  grpo_K50_N256_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "grpo" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/gradient_analysis"
echo "=== per_image  grpo_K50_N256_seed44  epoch=10 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/checkpoints/epoch_10.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "grpo" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/gradient_analysis"
echo "=== per_image  grpo_K50_N256_seed44  epoch=25 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/checkpoints/epoch_25.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "grpo" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/grpo_K50_N256_seed44/gradient_analysis"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/gradient_analysis"
echo "=== per_image  reinforce_K50_N256_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "reinforce" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/gradient_analysis"
echo "=== per_image  reinforce_K50_N256_seed44  epoch=10 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/checkpoints/epoch_10.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "reinforce" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/gradient_analysis"
echo "=== per_image  reinforce_K50_N256_seed44  epoch=25 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/checkpoints/epoch_25.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "reinforce" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/reinforce_K50_N256_seed44/gradient_analysis"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/gradient_analysis"
echo "=== per_image  rloo_K50_N256_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "rloo" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/gradient_analysis"
echo "=== per_image  rloo_K50_N256_seed44  epoch=10 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/checkpoints/epoch_10.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "rloo" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/gradient_analysis"
echo "=== per_image  rloo_K50_N256_seed44  epoch=25 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/checkpoints/epoch_25.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "rloo" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/rloo_K50_N256_seed44/gradient_analysis"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/binary_maxrl_K50_N64_seed44/gradient_analysis"
echo "=== per_image  binary_maxrl_K50_N64_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/binary_maxrl_K50_N64_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "binary_maxrl" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/binary_maxrl_K50_N64_seed44/gradient_analysis"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/dg_K50_N64_seed44/gradient_analysis"
echo "=== per_image  dg_K50_N64_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/dg_K50_N64_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "dg" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/dg_K50_N64_seed44/gradient_analysis"

mkdir -p "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/gradient_analysis"
echo "=== per_image  exp_iou_ord_ce_K50_N64_seed44  epoch=1 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/checkpoints/epoch_1.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "exp_iou_ord_ce" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/gradient_analysis"
echo "=== per_image  exp_iou_ord_ce_K50_N64_seed44  epoch=10 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/checkpoints/epoch_10.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "exp_iou_ord_ce" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/gradient_analysis"
echo "=== per_image  exp_iou_ord_ce_K50_N64_seed44  epoch=25 ==="
"${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \
    --checkpoint "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/checkpoints/epoch_25.pt" \
    --data_dir "/zfsauton/datasets/imagenet" \
    --method "exp_iou_ord_ce" \
    --K 50 \
    --seed 44 \
    --n_images 500 \
    --n_threshold_images 50 \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed44/gradient_analysis"
