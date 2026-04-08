#!/usr/bin/env bash
#SBATCH --job-name=reeval_dg_g16
#SBATCH --account=bgtv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=16:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgtv/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgtv/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# Re-evaluate DG (Delightful Gradient) G=16 checkpoints (all 3 benchmarks, 200 completions)

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
RESULTS_DIR="/projects/bgtv/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_dg_g16_s0"
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
export WANDB_API_KEY="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"
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TMPDIR="/tmp/${USER}_reeval_tmp"
mkdir -p "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

$PYTHON -m experiment4_llm.src.eval_checkpoints \
  --results_dirs "${RESULTS_DIR}" \
  --run_names "DG_G16" \
  --eval_files \
    "${REPO_ROOT}/data/livecodebench_v5_test.parquet" \
    "${REPO_ROOT}/data/evalplus_humaneval_test.parquet" \
    "${REPO_ROOT}/data/evalplus_mbpp_test.parquet" \
  --eval_names livecodebench_v5 humaneval_plus mbpp_plus \
  --base_model Qwen/Qwen3-1.7B-Base \
  --n_completions 256 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k -1 \
  --max_tokens 4096 \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.95 \
  --timeout_per_test 6 \
  --n_workers 256 \
  --k_values 1 4 16 64 256 \
  --plot_output "${RESULTS_DIR}/reeval_plots/pass_at_k.png" \
  --wandb_run_ids nnatsrwr \
  --wandb_project cont-maxrl-code \
  --wandb_entity stablegradients

echo "=== DG G=16 re-evaluation complete ==="
