#!/usr/bin/env bash
#SBATCH --job-name=eval_1.5b_code_contests_maxrl_unnorm_g32_s42_on_livecodebench
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --time=47:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=120
#SBATCH --nodes=1
#SBATCH --exclude=gpu20,gpu21,gpu22,gpu23
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/results/1.5b_code_contests_maxrl_unnorm_g32_s42/eval_livecodebench/slurm_%j.out
#SBATCH --error=/zfsauton/scratch/shrinivr/cont-maxrl/results/1.5b_code_contests_maxrl_unnorm_g32_s42/eval_livecodebench/slurm_%j.err
set -euo pipefail

# ============================================================
# Eval: 1.5b trained on CODE_CONTESTS | eval on LIVECODEBENCH
# maxrl_unnorm | G=32 | seed=42
# ============================================================

REPO_ROOT="/zfsauton/scratch/shrinivr/cont-maxrl"
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

CHECKPOINT_DIR="${REPO_ROOT}/results/1.5b_code_contests_maxrl_unnorm_g32_s42"
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_livecodebench"
mkdir -p "${OUTPUT_DIR}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
echo "=== Eval: 1.5b_code_contests_maxrl_unnorm_g32_s42_on_livecodebench ==="
echo "GPUs: ${N_GPUS}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Benchmark: livecodebench (livecodebench_test.parquet)"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --data_file "${REPO_ROOT}/data/livecodebench_test.parquet" \
  --output_dir "${OUTPUT_DIR}" \
  --n_completions 200 \
  --temperature 0.8 \
  --max_tokens 1024 \
  --tensor_parallel_size ${N_GPUS} \
  --k_values 1 2 5 10 20 50 100

echo ""
echo "=== Evaluation complete: 1.5b_code_contests_maxrl_unnorm_g32_s42_on_livecodebench ==="
