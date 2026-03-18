#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_base_evalplus
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-1.7B-Base on HumanEval+ and MBPP+ (via evalplus)
# Base (pretrained) model — no chat template, completion only
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
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

export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

MODEL="Qwen/Qwen3-1.7B-Base"
OUTPUT_BASE="${REPO_ROOT}/results/qwen3_1.7b_base_evalplus"
mkdir -p "${OUTPUT_BASE}"

echo "=== Qwen3-1.7B-Base EvalPlus (HumanEval+ & MBPP+) ==="
echo "Model: ${MODEL}"
echo ""

# Patch evalplus default max_new_tokens from 768 -> 1024
EVALPLUS_BASE="$($PYTHON -c 'import evalplus.provider.base; print(evalplus.provider.base.__file__)')"
sed -i 's/max_new_tokens: int = 768/max_new_tokens: int = 1024/' "${EVALPLUS_BASE}"
echo "Patched evalplus max_new_tokens to 1024"

# ------------------------------------------------------------------
# 1. HumanEval+ — greedy (pass@1)
# ------------------------------------------------------------------
echo ">>> HumanEval+ (greedy, pass@1)"
$PYTHON -m evalplus.codegen \
  "${MODEL}" humaneval \
  --root "${OUTPUT_BASE}" \
  --greedy \
  --backend vllm \
  --tp 4 \
  --force_base_prompt \
  --trust_remote_code \
  --dtype bfloat16 \
  --attn_implementation eager

echo ">>> Evaluating HumanEval+"
$PYTHON -m evalplus.evaluate \
  --dataset humaneval \
  --samples "${OUTPUT_BASE}/humaneval/Qwen--Qwen3-1.7B-Base_vllm_temp_0.0.jsonl" \
  2>&1 | tee "${OUTPUT_BASE}/humaneval_results.txt"

echo ""

# ------------------------------------------------------------------
# 2. HumanEval+ — sampling (pass@10)
# ------------------------------------------------------------------
echo ">>> HumanEval+ (T=0.6, n=10, pass@10)"
$PYTHON -m evalplus.codegen \
  "${MODEL}" humaneval \
  --root "${OUTPUT_BASE}/sampling" \
  --n_samples 10 \
  --temperature 0.6 \
  --backend vllm \
  --tp 4 \
  --force_base_prompt \
  --trust_remote_code \
  --dtype bfloat16 \
  --attn_implementation eager

echo ">>> Evaluating HumanEval+ (sampling)"
$PYTHON -m evalplus.evaluate \
  --dataset humaneval \
  --samples "${OUTPUT_BASE}/sampling/humaneval/Qwen--Qwen3-1.7B-Base_vllm_temp_0.6.jsonl" \
  2>&1 | tee "${OUTPUT_BASE}/humaneval_sampling_results.txt"

echo ""

# ------------------------------------------------------------------
# 3. MBPP+ — greedy (pass@1)
# ------------------------------------------------------------------
echo ">>> MBPP+ (greedy, pass@1)"
$PYTHON -m evalplus.codegen \
  "${MODEL}" mbpp \
  --root "${OUTPUT_BASE}" \
  --greedy \
  --backend vllm \
  --tp 4 \
  --force_base_prompt \
  --trust_remote_code \
  --dtype bfloat16 \
  --attn_implementation eager

echo ">>> Evaluating MBPP+"
$PYTHON -m evalplus.evaluate \
  --dataset mbpp \
  --samples "${OUTPUT_BASE}/mbpp/Qwen--Qwen3-1.7B-Base_vllm_temp_0.0.jsonl" \
  2>&1 | tee "${OUTPUT_BASE}/mbpp_results.txt"

echo ""

# ------------------------------------------------------------------
# 4. MBPP+ — sampling (pass@10)
# ------------------------------------------------------------------
echo ">>> MBPP+ (T=0.6, n=10, pass@10)"
$PYTHON -m evalplus.codegen \
  "${MODEL}" mbpp \
  --root "${OUTPUT_BASE}/sampling" \
  --n_samples 10 \
  --temperature 0.6 \
  --backend vllm \
  --tp 4 \
  --force_base_prompt \
  --trust_remote_code \
  --dtype bfloat16 \
  --attn_implementation eager

echo ">>> Evaluating MBPP+ (sampling)"
$PYTHON -m evalplus.evaluate \
  --dataset mbpp \
  --samples "${OUTPUT_BASE}/sampling/mbpp/Qwen--Qwen3-1.7B-Base_vllm_temp_0.6.jsonl" \
  2>&1 | tee "${OUTPUT_BASE}/mbpp_sampling_results.txt"

echo ""
echo "=== All evaluations complete ==="
echo "Results in: ${OUTPUT_BASE}/"
echo "  humaneval_results.txt       (greedy pass@1)"
echo "  humaneval_sampling_results.txt (T=0.6 pass@10)"
echo "  mbpp_results.txt            (greedy pass@1)"
echo "  mbpp_sampling_results.txt   (T=0.6 pass@10)"
