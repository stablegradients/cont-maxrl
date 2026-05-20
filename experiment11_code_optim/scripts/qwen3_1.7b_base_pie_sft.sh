#!/usr/bin/env bash
#SBATCH --job-name=q3_1.7b_pie_sft
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# PIE cold-start SFT (spec v9 §6) — Qwen3-1.7B-Base on train_hq_only.jsonl
# Output is the actor checkpoint consumed by the RL launch via
# $PIE_SFT_CKPT (see qwen3_1.7b_base_pie_{maxrl,grpo,rloo}_g16.sh).
#
# Usage: sbatch qwen3_1.7b_base_pie_sft.sh [SEED]
# ============================================================

SEED=${1:-0}

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/work/nvme/bgtu/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

PIE_DATA_ROOT="/work/nvme/bgmo/sramasubramanian/pie_data"
TRAIN_JSONL="${PIE_DATA_ROOT}/raw/train_hq_only.jsonl"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B-Base}"

OUTPUT_DIR="${STORAGE_ROOT}/sft_ckpts/qwen3_1.7b_pie_sft_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL, single-node)
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# W&B (env-only)
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in submission env}"
export WANDB_ENTITY="${WANDB_ENTITY:-stablegradients}"
export WANDB_PROJECT="${WANDB_PROJECT:-cont-maxrl-pie}"
export WANDB_RUN_GROUP="qwen3-1.7b-base_pie_sft"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_pie_sft_tmp"
mkdir -p "${TORCH_HOME}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# FSDP environment (HF Trainer reads these)
export FSDP_TRANSFORMER_CLS_TO_WRAP="Qwen3DecoderLayer"
export ACCELERATE_MIXED_PRECISION="bf16"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}

echo "=== SFT: Qwen3-1.7B Base | PIE cold-start | seed=${SEED} ==="
echo "Base model:    ${BASE_MODEL}"
echo "Train data:    ${TRAIN_JSONL}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "GPUs:          ${N_GPUS}"

# torchrun for multi-GPU FSDP
torchrun --nproc_per_node=${N_GPUS} --master_port=29501 \
    "${REPO_ROOT}/experiment11_code_optim/sft/cold_start_sft.py" \
    --train_jsonl "${TRAIN_JSONL}" \
    --base_model "${BASE_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --max_length 8192 \
    --seed ${SEED}

echo ""
echo "=== SFT complete: seed=${SEED} ==="
echo "Final checkpoint: ${OUTPUT_DIR}/final"
echo "Wire as: export PIE_SFT_CKPT=${OUTPUT_DIR}/final"
