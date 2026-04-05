#!/usr/bin/env bash
#SBATCH --job-name=smollm_maze_sft
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
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
export WANDB_API_KEY="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"
export WANDB_ENTITY="stablegradients"
export WANDB_PROJECT="cont-maxrl-maze"

echo "=== SFT: SmolLM2-135M on 17x17 mazes ==="

$PYTHON "${REPO_ROOT}/experiment6_maze/src/sft.py" \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --train_file "${REPO_ROOT}/data/maze_17x17/sft_train.jsonl" \
  --val_file "${REPO_ROOT}/data/maze_17x17/sft_val.jsonl" \
  --output_dir "${REPO_ROOT}/checkpoints/smollm2_135m_maze17x17_sft" \
  --batch_size 64 \
  --lr 5e-5 \
  --epochs 3 \
  --warmup_steps 100 \
  --max_length 896 \
  --save_steps 500 \
  --eval_steps 500

echo "=== SFT complete ==="
