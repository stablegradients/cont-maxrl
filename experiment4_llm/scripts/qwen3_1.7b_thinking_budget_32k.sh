#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_think_32k
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
# Thinking budget eval: Qwen3-1.7B (instruct) on LiveCodeBench
# Single config: max_tokens=32768
# ============================================================

# Activate conda environment (before set -u, conda scripts use unset vars)
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL)
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# Caches
export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

# Project must be importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# verl/training uses vLLM v0 engine; eval must match
export VLLM_USE_V1=0
# Disable torch.compile (inductor backend requires triton, which is not installed)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

MODEL="Qwen/Qwen3-1.7B"
DATA_FILE="${REPO_ROOT}/data/livecodebench_test.parquet"
OUTPUT_BASE="${REPO_ROOT}/results/qwen3_1.7b_thinking_budget_sweep"
MAX_TOKENS=32768
OUTPUT_DIR="${OUTPUT_BASE}/max_tokens_${MAX_TOKENS}"

echo "=== Qwen3-1.7B Thinking Budget Eval: 32k ==="
echo "Model: ${MODEL} (enable_thinking=True)"
echo "max_tokens: ${MAX_TOKENS}"
echo "Eval config: 10 completions, T=0.6, top_p=0.95, gpu_mem=0.93"
echo ""

# Skip if already done
if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
  echo "Already done, exiting."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

# max_num_seqs: at 32k context on GH200 (95GB usable), ~24 concurrent seqs
# Conservative to avoid KV cache preemptions
$PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
  --checkpoint_dir "${MODEL}" \
  --data_file "${DATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --n_completions 10 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens "${MAX_TOKENS}" \
  --enable_thinking \
  --gpu_memory_utilization 0.93 \
  --max_num_seqs 96 \
  --max_model_len "${MAX_TOKENS}" \
  --tensor_parallel_size 4 \
  --timeout_per_test 5 \
  --n_workers 64 \
  --k_values 1 10 \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "=== 32k eval complete ==="
echo "Results: ${OUTPUT_DIR}/eval_results.json"
