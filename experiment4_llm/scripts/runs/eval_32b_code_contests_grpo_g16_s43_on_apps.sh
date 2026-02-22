#!/usr/bin/env bash
#SBATCH --job-name=eval_32b_code_contests_grpo_g16_s43_on_apps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=32
#SBATCH --exclude=gpu20,gpu21,gpu22,gpu23
#SBATCH --output=results/32b_code_contests_grpo_g16_s43/eval_apps/slurm_%j.out
#SBATCH --error=results/32b_code_contests_grpo_g16_s43/eval_apps/slurm_%j.err
set -euo pipefail

# ============================================================
# Eval: 32b trained on CODE_CONTESTS | eval on APPS
# grpo | G=16 | seed=43
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# Conda CUDA (not system CUDA)
export CUDA_HOME="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Caches
export HF_HOME="/zfsauton/scratch/shrinivr/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

# Project must be importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_DIR="${REPO_ROOT}/results/32b_code_contests_grpo_g16_s43"
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_apps"
mkdir -p "${OUTPUT_DIR}"

N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 8)
echo "=== Eval: 32b_code_contests_grpo_g16_s43_on_apps ==="
echo "GPUs: ${N_GPUS}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Benchmark: apps (apps_test.parquet)"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --data_file "${REPO_ROOT}/data/apps_test.parquet" \
  --output_dir "${OUTPUT_DIR}" \
  --n_completions 200 \
  --temperature 0.8 \
  --max_tokens 2048 \
  --tensor_parallel_size ${N_GPUS} \
  --k_values 1 2 5 10 20 50 100

echo ""
echo "=== Evaluation complete: 32b_code_contests_grpo_g16_s43_on_apps ==="
