#!/usr/bin/env bash
# Run all tests for experiment10_reasoning_gym.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
CONDA_ENV="${CONDA_ENV:-rgym}"

if [ -f "/data/miniconda/etc/profile.d/conda.sh" ]; then
    source /data/miniconda/etc/profile.d/conda.sh
fi
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
exec python -m pytest experiment10_reasoning_gym/tests/ "$@"
