#!/usr/bin/env bash
set -euo pipefail

# Pre-download model weights so training doesn't block on HF downloads.
# Downloads to HF_HOME cache -- shared across all runs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
eval "$(conda shell.bash hook)"
conda activate cont-maxrl

export HF_HOME="${REPO_ROOT}/.cache/huggingface"

echo "=== Downloading model weights ==="
echo "Cache dir: ${HF_HOME}"

MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen/Qwen2.5-Coder-32B-Instruct"
)

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- ${model} ---"
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = '${model}'
print(f'  Downloading tokenizer: {model_id}')
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'  Downloading model weights: {model_id}')
AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='cpu',  # download only, don't load to GPU
)
print(f'  Done: {model_id}')
"
done

echo ""
echo "=== All models downloaded ==="
