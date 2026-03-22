#!/usr/bin/env bash
#SBATCH --job-name=qwen3_4b_think_lcbv5
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-4B thinking mode on LCB v5 (Oct 2024 - Jan 2025)
# Matches Qwen3 technical report methodology:
#   - v5-only dataset (167 problems)
#   - Relaxed system prompt (no output restriction)
#   - top_k=20, T=0.6, top_p=0.95
#   - 32k output budget
# Paper reference: arxiv 2505.09388, page 13
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

export REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

MODEL="Qwen/Qwen3-4B"
DATA_FILE="${REPO_ROOT}/data/livecodebench_v5_test.parquet"
OUTPUT_BASE="${REPO_ROOT}/results/qwen3_4b_thinking_lcb"

# Official LCB system prompt
SYSTEM_PROMPT="You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

echo "=== Qwen3-4B Thinking on LCB v5 (163 problems) ==="
echo "Model: ${MODEL} (enable_thinking=True)"
echo "Dataset: v5-only (163 problems)"
echo "Config: T=0.6, top_p=0.95, top_k=20, max_tokens=32768"
echo ""

# ==================================================================
# 1. Sampling (T=0.6, n=10) — matches paper config
# ==================================================================
SAMPLING_DIR="${OUTPUT_BASE}/sampling"
if [[ -f "${SAMPLING_DIR}/eval_results.json" ]]; then
  echo ">>> Sampling: already done, skipping."
else
  echo ">>> Sampling (T=0.6, top_k=20, n=10)"
  mkdir -p "${SAMPLING_DIR}"
  $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
    --checkpoint_dir "${MODEL}" \
    --data_file "${DATA_FILE}" \
    --output_dir "${SAMPLING_DIR}" \
    --n_completions 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --max_tokens 32768 \
    --enable_thinking \
    --system_prompt "${SYSTEM_PROMPT}" \
    --gpu_memory_utilization 0.93 \
    --max_num_seqs 48 \
    --max_model_len 33792 \
    --tensor_parallel_size 4 \
    --timeout_per_test 6 \
    --n_workers 256 \
    --k_values 1 10 \
    2>&1 | tee "${SAMPLING_DIR}/eval.log"
fi

# ==================================================================
# 2. Greedy (T=0, n=1) — clean pass@1 reference
# ==================================================================
GREEDY_DIR="${OUTPUT_BASE}/greedy"
if [[ -f "${GREEDY_DIR}/eval_results.json" ]]; then
  echo ">>> Greedy: already done, skipping."
else
  echo ">>> Greedy (T=0, n=1)"
  mkdir -p "${GREEDY_DIR}"
  $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
    --checkpoint_dir "${MODEL}" \
    --data_file "${DATA_FILE}" \
    --output_dir "${GREEDY_DIR}" \
    --n_completions 1 \
    --temperature 0.0 \
    --top_p 1.0 \
    --top_k -1 \
    --max_tokens 32768 \
    --enable_thinking \
    --system_prompt "${SYSTEM_PROMPT}" \
    --gpu_memory_utilization 0.93 \
    --max_num_seqs 48 \
    --max_model_len 33792 \
    --tensor_parallel_size 4 \
    --timeout_per_test 6 \
    --n_workers 256 \
    --k_values 1 \
    2>&1 | tee "${GREEDY_DIR}/eval.log"
fi

echo ""
echo "=== Qwen3-4B Thinking LCB v5 eval complete ==="
echo "Results: ${OUTPUT_BASE}/"
