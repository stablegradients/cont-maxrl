#!/usr/bin/env bash
#SBATCH --job-name=pie_reeval_ckpt
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# Re-eval an arbitrary (arm, step) verl ckpt on pie_val_200.
# Usage: sbatch reeval_ckpt.sh <arm> <step> [tag]
#   tag in {"", "bs64"} — selects which results dir to read.

ARM="${1:?arm required: maxrl_unnorm | grpo | rloo}"
STEP="${2:?step required}"
TAG="${3:-}"  # "" -> g16 (bs256); "bs64" -> g16_bs64

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -eo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
cd "${REPO_ROOT}"

set -a; source .env 2>/dev/null || true; set +a
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -n "${TAG}" ]]; then
  DIR_TAG="g16_${TAG}"
  GROUP_TAG="G16_${TAG^^}"
  SUBDIR="${ARM}_${TAG}"
else
  DIR_TAG="g16"
  GROUP_TAG="G16"
  SUBDIR="${ARM}"
fi
CKPT="/work/nvme/bgmo/sramasubramanian/cont-maxrl/results/qwen3_1.7b_instruct_pie_${ARM}_${DIR_TAG}_s0/global_step_${STEP}"
[ -d "${CKPT}/actor/huggingface" ] || { echo "FATAL: no HF snapshot at ${CKPT}/actor/huggingface"; exit 2; }

WANDB_GROUP="qwen3-1.7b-instruct_pie_${ARM}_${GROUP_TAG}"

$PYTHON "${REPO_ROOT}/experiment11_code_optim/scripts/reeval_step100_ckpt.py" \
    --ckpt "${CKPT}" --arm "${ARM}" --step "${STEP}" \
    --wandb_group "${WANDB_GROUP}" --output_subdir "${SUBDIR}"
