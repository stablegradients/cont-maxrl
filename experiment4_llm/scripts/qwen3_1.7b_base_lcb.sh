#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_base_lcb
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-1.7B-Base on LiveCodeBench v5
# Base (pretrained) model — plain text prompts, no chat template
# Greedy (pass@1) + Sampling T=0.6 (pass@10)
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

export REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "=== Qwen3-1.7B-Base on LiveCodeBench v5 ==="
echo "Greedy (pass@1) + Sampling T=0.6 n=10 (pass@10)"
echo ""

$PYTHON -u << 'PYEOF'
import json
import os
import gc
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from experiment4_llm.src.evaluate import load_problems, evaluate_completions

MODEL = "Qwen/Qwen3-1.7B-Base"
REPO_ROOT = os.environ["REPO_ROOT"]
DATA_FILE = f"{REPO_ROOT}/data/livecodebench_test.parquet"
OUTPUT_BASE = Path(REPO_ROOT) / "results" / "qwen3_1.7b_base_lcb"
MAX_TOKENS = 1024
N_WORKERS = 256

# ---- Load problems ----
print(f"Loading problems from {DATA_FILE}")
problems = load_problems(DATA_FILE)
print(f"  {len(problems)} problems loaded")

# ---- Format prompts (plain text for base model) ----
formatted_prompts = []
for problem in problems:
    parts = [msg["content"] for msg in problem["prompt"]]
    text = "\n\n".join(parts) + "\n\n```python\n"
    formatted_prompts.append(text)

# ---- Init model ----
print("Loading model...")
llm = LLM(
    model=MODEL,
    tokenizer=MODEL,
    trust_remote_code=True,
    gpu_memory_utilization=0.93,
    tensor_parallel_size=4,
    max_num_seqs=256,
    max_model_len=2048,
    dtype="bfloat16",
    enforce_eager=True,
)

STOP_TOKENS = ["```\n", "\n```"]

# ==================================================================
# 1. Greedy (pass@1)
# ==================================================================
greedy_dir = OUTPUT_BASE / "greedy"
greedy_dir.mkdir(parents=True, exist_ok=True)
completions_path = greedy_dir / "completions.json"

if (greedy_dir / "eval_results.json").exists():
    print("\n>>> Greedy: already done, skipping.")
else:
    print("\n>>> Greedy (T=0, n=1)")
    if completions_path.exists():
        print(f"  Loading cached completions from {completions_path}")
        with open(completions_path) as f:
            completions = json.load(f)
    else:
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stop=STOP_TOKENS,
        )
        outputs = llm.generate(formatted_prompts, sampling_params)
        completions = [[o.text for o in output.outputs] for output in outputs]
        with open(completions_path, "w") as f:
            json.dump(completions, f)
        print(f"  Completions saved to {completions_path}")

    print("  Evaluating...")
    results = evaluate_completions(
        completions=completions,
        problems=problems,
        k_values=[1],
        timeout_per_test=5,
        n_workers=N_WORKERS,
    )
    with open(greedy_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n  === Greedy Results ===")
    for k, v in results["pass_at_k"].items():
        if v is not None:
            print(f"  pass@{k}: {v:.4f} ({v*100:.1f}%)")
    s = results["summary"]
    print(f"  Mean output length (chars): {s['mean_output_length']:.0f}")

# ==================================================================
# 2. Sampling T=0.6, n=10 (pass@1, pass@10)
# ==================================================================
sampling_dir = OUTPUT_BASE / "sampling"
sampling_dir.mkdir(parents=True, exist_ok=True)
completions_path = sampling_dir / "completions.json"

if (sampling_dir / "eval_results.json").exists():
    print("\n>>> Sampling: already done, skipping.")
else:
    print("\n>>> Sampling (T=0.6, n=10)")
    if completions_path.exists():
        print(f"  Loading cached completions from {completions_path}")
        with open(completions_path) as f:
            completions = json.load(f)
    else:
        sampling_params = SamplingParams(
            n=10,
            temperature=0.6,
            top_p=0.95,
            max_tokens=MAX_TOKENS,
            stop=STOP_TOKENS,
        )
        outputs = llm.generate(formatted_prompts, sampling_params)
        completions = [[o.text for o in output.outputs] for output in outputs]
        with open(completions_path, "w") as f:
            json.dump(completions, f)
        print(f"  Completions saved to {completions_path}")

    print("  Evaluating...")
    results = evaluate_completions(
        completions=completions,
        problems=problems,
        k_values=[1, 10],
        timeout_per_test=5,
        n_workers=N_WORKERS,
    )
    with open(sampling_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n  === Sampling Results ===")
    for k, v in results["pass_at_k"].items():
        if v is not None:
            print(f"  pass@{k}: {v:.4f} ({v*100:.1f}%)")
    s = results["summary"]
    print(f"  Mean output length (chars): {s['mean_output_length']:.0f}")

print("\n=== All done ===")
print(f"Results: {OUTPUT_BASE}/greedy/eval_results.json")
print(f"         {OUTPUT_BASE}/sampling/eval_results.json")

PYEOF

echo ""
echo "=== Qwen3-1.7B-Base LCB v5 eval complete ==="
