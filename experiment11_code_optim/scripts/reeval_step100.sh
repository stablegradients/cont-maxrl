#!/usr/bin/env bash
#SBATCH --job-name=pie_reeval_step100
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# Re-eval step-100 ckpts for all three arms (one ckpt per submission;
# the spray script below launches three jobs back to back).
# Usage: sbatch reeval_step100.sh <arm>
#   arm ∈ {maxrl_unnorm, grpo, rloo}

ARM="${1:?arm required: maxrl_unnorm | grpo | rloo}"

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

CKPT="/work/nvme/bgmo/sramasubramanian/cont-maxrl/results/qwen3_1.7b_instruct_pie_${ARM}_g16_s0/global_step_100"
[ -d "${CKPT}" ] || { echo "FATAL: ckpt missing: ${CKPT}"; exit 2; }

$PYTHON "${REPO_ROOT}/experiment11_code_optim/scripts/reeval_step100_ckpt.py" \
    --ckpt "${CKPT}" --arm "${ARM}" --step 100
