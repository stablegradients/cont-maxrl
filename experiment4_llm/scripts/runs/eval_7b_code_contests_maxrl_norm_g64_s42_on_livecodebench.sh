#!/usr/bin/env bash
#SBATCH --job-name=eval_7b_code_contests_maxrl_norm_g64_s42_on_livecodebench
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Eval: 7b trained on CODE_CONTESTS | eval on LIVECODEBENCH
# maxrl_norm | G=64 | seed=42
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
# Single-node uses NVLink/shared memory, no libfabric needed
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# W&B
export WANDB_API_KEY="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"
export WANDB_ENTITY="stablegradients"

# Caches
export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

# Project must be importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_DIR="${REPO_ROOT}/results/7b_code_contests_maxrl_norm_g64_s42"
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_livecodebench"
mkdir -p "${OUTPUT_DIR}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
echo "=== Eval: 7b_code_contests_maxrl_norm_g64_s42_on_livecodebench ==="
echo "GPUs: ${N_GPUS}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Benchmark: livecodebench (livecodebench_test.parquet)"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --data_file "${REPO_ROOT}/data/livecodebench_test.parquet" \
  --output_dir "${OUTPUT_DIR}" \
  --n_completions 200 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k -1 \
  --max_tokens 1024 \
  --tensor_parallel_size ${N_GPUS} \
  --k_values 1 2 5 10 20 50 100 200

echo ""
echo "=== Evaluation complete: 7b_code_contests_maxrl_norm_g64_s42_on_livecodebench ==="
