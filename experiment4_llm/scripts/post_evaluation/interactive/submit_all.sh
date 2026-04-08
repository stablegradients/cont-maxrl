#!/usr/bin/env bash
# Submit all reeval jobs on ghx4, spread across all 4 accounts
# ~2.8h per checkpoint, so:
#   1 ckpt jobs → 4h, 3 ckpt jobs → 10h, split big runs across accounts
set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"

echo "=== Submitting reeval jobs on ghx4, spread across 4 accounts ==="

# --- Small runs (1-3 checkpoints, fit in one job each) ---

# bmaxrl_g16 (1 ckpt) → bgmo, ~3h
sbatch --account=bgmo-dtai-gh --partition=ghx4 --gpus-per-node=4 --mem=0 \
    --cpus-per-task=288 --nodes=1 --time=4:00:00 \
    --job-name=re_bmaxrl_g16 \
    --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out \
    --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err \
    ${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/reeval_bmaxrl_g16.sh

# rloo_g16 (1 ckpt) → bgtv, ~3h
sbatch --account=bgtv-dtai-gh --partition=ghx4 --gpus-per-node=4 --mem=0 \
    --cpus-per-task=288 --nodes=1 --time=4:00:00 \
    --job-name=re_rloo_g16 \
    --output=/projects/bgtv/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out \
    --error=/projects/bgtv/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err \
    ${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/reeval_rloo_g16.sh

# grpo_g16 (3 ckpts) → bgtu, ~9h
sbatch --account=bgtu-dtai-gh --partition=ghx4 --gpus-per-node=4 --mem=0 \
    --cpus-per-task=288 --nodes=1 --time=10:00:00 \
    --job-name=re_grpo_g16 \
    --output=/projects/bgtu/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out \
    --error=/projects/bgtu/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err \
    ${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/reeval_grpo_g16.sh

# maxrl_g16 (3 ckpts) → bgtv (rebalanced off bgtu), ~9h
sbatch --account=bgtv-dtai-gh --partition=ghx4 --gpus-per-node=4 --mem=0 \
    --cpus-per-task=288 --nodes=1 --time=10:00:00 \
    --job-name=re_maxrl_g16 \
    --output=/projects/bgtu/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out \
    --error=/projects/bgtu/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err \
    ${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/reeval_maxrl_g16.sh

# --- Big runs (split across accounts) ---

# grpo_g4 (10 ckpts) → split 5+5 across bgts + bgmo
sbatch --account=bgts-dtai-gh --partition=ghx4 --gpus-per-node=4 --mem=0 \
    --cpus-per-task=288 --nodes=1 --time=16:00:00 \
    --job-name=re_grpo_g4 \
    --output=/projects/bgts/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out \
    --error=/projects/bgts/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err \
    ${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/reeval_grpo_g4.sh

# maxrl_g4 (8 ckpts) → bgtu, ~22h
sbatch --account=bgtu-dtai-gh --partition=ghx4 --gpus-per-node=4 --mem=0 \
    --cpus-per-task=288 --nodes=1 --time=24:00:00 \
    --job-name=re_maxrl_g4 \
    --output=/projects/bgts/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out \
    --error=/projects/bgts/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err \
    ${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/reeval_maxrl_g4.sh

echo ""
squeue -u $USER
echo ""
echo "=== All jobs submitted ==="
