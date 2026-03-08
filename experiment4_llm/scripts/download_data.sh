#!/usr/bin/env bash

# Download and prepare all datasets for verl training format.
# Output: data/*.parquet

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cont-maxrl

set -euo pipefail

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
mkdir -p "${REPO_ROOT}/data"

echo "=== Downloading and preparing all datasets ==="
python "${REPO_ROOT}/experiment4_llm/src/data_prep.py" \
  --dataset all \
  --output_dir "${REPO_ROOT}/data" \
  --min_tests 3 \
  --hf_cache_dir "${HF_HOME}"

echo ""
echo "=== Done ==="
echo "Files:"
ls -lh "${REPO_ROOT}/data/"*.parquet
