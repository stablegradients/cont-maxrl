#!/bin/bash
#SBATCH --job-name=gcos_jord_s43
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gcos_jord_s43_%j.out

set +u
cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
set -u
set -eo pipefail
PYBIN="${CONDA_PREFIX:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl}/bin/python"

RUN_DIR=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_K50_N64_seed43
OUT_DIR="${RUN_DIR}/gradient_analysis"
mkdir -p "${OUT_DIR}"
COMMON_ARGS=(
    --data_dir "/zfsauton/datasets/imagenet"
    --K 50
    --seed 43
    --ref_method "exp_iou_ord_ce"
    --methods "gos,binary_maxrl,grpo,rloo,reinforce,dg"
    --Gs "4,8,16,32,64,128,256,512,1024"
    --n_images 64
    --n_trials 5
    --output_dir "${OUT_DIR}"
)

for E in 1 10 25; do
    CKPT="${RUN_DIR}/checkpoints/epoch_${E}.pt"
    if [[ ! -f "${CKPT}" ]]; then
        echo "[skip] missing ${CKPT}"
        continue
    fi
    echo "=== cosine ref=exp_iou_ord_ce  exp_iou_ord_ce_seed43  epoch=${E} ==="
    "${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis cosine \
        --checkpoint "${CKPT}" \
        "${COMMON_ARGS[@]}"
done
