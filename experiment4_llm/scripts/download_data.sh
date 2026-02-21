#!/usr/bin/env bash
set -euo pipefail

# Download and prepare APPS dataset for verl training format.
# Output: data/apps_train.jsonl, data/apps_test.jsonl

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cont-maxrl

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
mkdir -p "${REPO_ROOT}/data"

echo "=== Downloading and preparing APPS dataset ==="
python "${REPO_ROOT}/experiment4_llm/src/data_prep.py" \
  --output_dir "${REPO_ROOT}/data" \
  --min_tests 3 \
  --difficulties introductory interview \
  --hf_cache_dir "${HF_HOME}"

echo ""
echo "=== Done ==="
echo "Train: $(wc -l < "${REPO_ROOT}/data/apps_train.jsonl") problems"
echo "Test:  $(wc -l < "${REPO_ROOT}/data/apps_test.jsonl") problems"
