#!/usr/bin/env bash
#SBATCH --job-name=rgym
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/rgym_%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/rgym_%x_%j.err
#
# SLURM wrapper for a single Reasoning Gym training run. On clusters without
# SLURM, run launch.sh directly with its positional arguments.
#
# Usage (sbatch):
#   sbatch scripts/launch_slurm.sh <task> <difficulty> <estimator> <reward_mode> [seed]
#
# Usage (interactive, single node):
#   bash scripts/launch_slurm.sh <task> <difficulty> <estimator> <reward_mode> [seed]

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
CONDA_ENV="${CONDA_ENV:-rgym}"

# Load conda and activate our dedicated env
if [ -f "/data/miniconda/etc/profile.d/conda.sh" ]; then
    source /data/miniconda/etc/profile.d/conda.sh
fi
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Ray tmp dirs (isolate per run to avoid collisions)
RUN_TAG="${1:-unknown}_${2:-0}_${3:-unknown}_${4:-unknown}"
export RAY_TMPDIR="/tmp/ray_${USER}_${RUN_TAG}"
export TMPDIR="/tmp/${USER}_verl_${RUN_TAG}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

bash "${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh" "$@"
