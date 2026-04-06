#!/usr/bin/env bash
#SBATCH --job-name=reeval_evalplus
#SBATCH --account=bgtv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Re-evaluate G=16 checkpoints on HumanEval+ and MBPP+
# with fixed functional prompt template.
#
# Evaluates both GRPO and MaxRL at insitu_eval epoch boundaries.
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# Conda CUDA
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
export TMPDIR="/tmp/${USER}_reeval_tmp"
mkdir -p "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Eval steps: match the insitu_eval epoch boundaries from the G=16 training scripts
# steps_per_epoch=83, so eval at: 83, 166, 249, 332, 415, 498, 540 (last ckpt for GRPO)
# Also include early steps for the full learning curve
EVAL_STEPS="20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 400 420 440 460 480 500 520 540"

GRPO_DIR="${REPO_ROOT}/results/qwen3_1.7b_base_deepcoder_grpo_g16_s42"
MAXRL_DIR="${REPO_ROOT}/results/qwen3_1.7b_base_deepcoder_maxrl_g16_s42"

HE_PARQUET="${REPO_ROOT}/data/evalplus_humaneval_test.parquet"
MBPP_PARQUET="${REPO_ROOT}/data/evalplus_mbpp_test.parquet"

echo "=== Re-evaluation: HE+ & MBPP+ with fixed prompts ==="
echo "GRPO dir: ${GRPO_DIR}"
echo "MaxRL dir: ${MAXRL_DIR}"
echo "Steps: ${EVAL_STEPS}"
echo ""

$PYTHON -m experiment4_llm.src.eval_checkpoints \
  --results_dirs "${GRPO_DIR}" "${MAXRL_DIR}" \
  --run_names "GRPO_G16" "MaxRL_G16" \
  --eval_files "${HE_PARQUET}" "${MBPP_PARQUET}" \
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
  --plot_output "${REPO_ROOT}/results/reeval_plots/pass_at_k.png"

echo ""
echo "=== Re-evaluation complete ==="
echo "Plots: ${REPO_ROOT}/results/reeval_plots/"
echo "Metrics: ${GRPO_DIR}/reeval_*/  and  ${MAXRL_DIR}/reeval_*/"
