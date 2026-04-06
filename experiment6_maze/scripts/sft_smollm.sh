#!/usr/bin/env bash
#SBATCH --job-name=qwen2_3m_maze_sft
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export WANDB_API_KEY="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"
export WANDB_ENTITY="stablegradients"
export WANDB_MODE="offline"
export WANDB_PROJECT="cont-maxrl-maze"

MODEL_INIT="${REPO_ROOT}/checkpoints/maze17x17_qwen2_3m_init"
DATA_DIR="${REPO_ROOT}/data/maze_17x17"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/maze17x17_qwen2_3m_sft"

echo "=== Step 1: Generate SFT data (1M train, 256 val) ==="
$PYTHON -m experiment6_maze.src.sft_data_prep \
  --size 17 \
  --n_train 1000000 \
  --n_val 256 \
  --train_seed_start 0 \
  --val_seed_start 1100000 \
  --output_dir "${DATA_DIR}"

echo "=== Step 2: SFT training — Qwen2 3.94M from scratch ==="
$PYTHON -m experiment6_maze.src.sft \
  --model_path "${MODEL_INIT}" \
  --train_data "${DATA_DIR}/sft_train.parquet" \
  --val_data "${DATA_DIR}/sft_val.parquet" \
  --output_dir "${OUTPUT_DIR}" \
  --learning_rate 5e-4 \
  --batch_size 32 \
  --micro_batch_size 8 \
  --max_steps 1500 \
  --max_length 512 \
  --warmup_ratio 0.0 \
  --eval_steps 50 \
  --save_steps 500 \
  --use_generative_eval \
  --eval_samples 50 \
  --n_samples_per_prompt 8 \
  --eval_temperature 1.0 \
  --use_wandb \
  --project_name "cont-maxrl-maze" \
  --experiment_name "qwen2_3m_sft"

echo "=== SFT complete ==="
