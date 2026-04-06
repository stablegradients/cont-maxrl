#!/usr/bin/env bash
#SBATCH --job-name=reeval_grpo_g16
#SBATCH --account=bgtv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# Re-evaluate GRPO G=16 checkpoints on HE+ and MBPP+ (fixed prompts)

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

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_USE_V1=0
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TMPDIR="/tmp/${USER}_reeval_tmp"
mkdir -p "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

EVAL_STEPS="260 540"

$PYTHON -m experiment4_llm.src.eval_checkpoints \
  --results_dirs "${REPO_ROOT}/results/qwen3_1.7b_base_deepcoder_grpo_g16_s42" \
  --run_names "GRPO_G16" \
  --eval_files "${REPO_ROOT}/data/evalplus_humaneval_test.parquet" "${REPO_ROOT}/data/evalplus_mbpp_test.parquet" \
  --eval_names humaneval_plus mbpp_plus \
  --base_model Qwen/Qwen3-1.7B-Base \
  --steps ${EVAL_STEPS} \
  --n_completions 200 \
  --temperature 0.6 \
  --max_tokens 4096 \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.90 \
  --timeout_per_test 6 \
  --n_workers 128 \
  --k_values 1 2 5 10 20 50 100 200 \
  --plot_output "${REPO_ROOT}/results/reeval_plots/pass_at_k_grpo.png"

echo "=== GRPO G=16 re-evaluation complete ==="
