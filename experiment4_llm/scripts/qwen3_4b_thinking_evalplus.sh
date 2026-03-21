#!/usr/bin/env bash
#SBATCH --job-name=qwen3_4b_think_evalplus
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-4B (instruct, thinking mode) on HumanEval+ and MBPP+
# 32k output budget, T=0.6 for sampling, greedy for pass@1
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

export REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
export PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
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

OUTPUT_BASE="${REPO_ROOT}/results/qwen3_4b_thinking_evalplus"
mkdir -p "${OUTPUT_BASE}"

echo "=== Qwen3-4B Thinking Mode EvalPlus (HumanEval+ & MBPP+) ==="
echo "Model: Qwen/Qwen3-4B (enable_thinking=True)"
echo "Output budget: 32768 tokens"
echo ""

$PYTHON -u << 'PYEOF'
import json
import os
import re
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"
OUTPUT_BASE = Path(os.environ["REPO_ROOT"]) / "results" / "qwen3_4b_thinking_evalplus"
MAX_TOKENS = 32768


def extract_code_from_thinking(response: str) -> str:
    """Strip thinking tags and extract code from model response."""
    text = response.strip()
    # Strip special tokens
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    # Strip thinking
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    # Extract from markdown code block
    text_lower = text.lower()
    if "```python" in text_lower:
        idx = text_lower.index("```python")
        after = text[idx + len("```python"):]
        if after.startswith("3") and (len(after) == 1 or after[1] in "\n\r \t"):
            after = after[1:]
        code = after.split("```")[0]
        return code.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            lines = code.split("\n")
            if lines and lines[0].strip().lower() in ("python", "python3", "py", ""):
                lines = lines[1:]
            return "\n".join(lines).strip()
    return text


def generate_and_evaluate(dataset_name, problems, tokenizer, llm, temperature, n_samples, tag):
    """Generate completions, extract code, save as evalplus jsonl, run evaluate."""
    output_dir = OUTPUT_BASE / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_file = output_dir / f"{dataset_name}_samples.jsonl"

    if samples_file.exists():
        print(f"  Cached: {samples_file}")
    else:
        # Build chat prompts
        prompts = []
        task_ids = []
        for task_id, problem in problems.items():
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. Complete the given function."},
                {"role": "user", "content": problem["prompt"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(text)
            task_ids.append(task_id)

        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            top_k=20 if temperature > 0 else -1,
            max_tokens=MAX_TOKENS,
        )

        print(f"  Generating {n_samples} completions for {len(prompts)} problems...")
        outputs = llm.generate(prompts, sampling_params)

        # Save as evalplus jsonl
        with open(samples_file, "w") as f:
            for task_id, output in zip(task_ids, outputs):
                for completion in output.outputs:
                    code = extract_code_from_thinking(completion.text)
                    sample = {
                        "task_id": task_id,
                        "solution": code,
                    }
                    f.write(json.dumps(sample) + "\n")

        print(f"  Saved to {samples_file}")

    # Run evalplus evaluate
    print(f"  Evaluating {dataset_name}...")
    results_file = output_dir / f"{dataset_name}_results.txt"
    import subprocess
    result = subprocess.run(
        [
            os.environ.get("PYTHON", "python"), "-m", "evalplus.evaluate",
            "--dataset", dataset_name,
            "--samples", str(samples_file),
        ],
        capture_output=True, text=True,
    )
    output_text = result.stdout + result.stderr
    with open(results_file, "w") as f:
        f.write(output_text)
    print(output_text)
    return output_text


# ---- Setup ----
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("Loading model...")
llm = LLM(
    model=MODEL,
    tokenizer=MODEL,
    trust_remote_code=True,
    gpu_memory_utilization=0.93,
    tensor_parallel_size=4,
    max_num_seqs=48,
    max_model_len=33792,  # 32k output + ~1k input headroom
    dtype="bfloat16",
    enforce_eager=True,
)

# ---- Load datasets ----
print("Loading HumanEval+ and MBPP+...")
humaneval = get_human_eval_plus()
mbpp = get_mbpp_plus()

# ---- 1. HumanEval+ greedy (pass@1) ----
print("\n>>> HumanEval+ (greedy, pass@1)")
generate_and_evaluate("humaneval", humaneval, tokenizer, llm, temperature=0.0, n_samples=1, tag="greedy")

# ---- 2. HumanEval+ sampling (pass@10) ----
print("\n>>> HumanEval+ (T=0.6, n=10, pass@10)")
generate_and_evaluate("humaneval", humaneval, tokenizer, llm, temperature=0.6, n_samples=10, tag="sampling")

# ---- 3. MBPP+ greedy (pass@1) ----
print("\n>>> MBPP+ (greedy, pass@1)")
generate_and_evaluate("mbpp", mbpp, tokenizer, llm, temperature=0.0, n_samples=1, tag="greedy")

# ---- 4. MBPP+ sampling (pass@10) ----
print("\n>>> MBPP+ (T=0.6, n=10, pass@10)")
generate_and_evaluate("mbpp", mbpp, tokenizer, llm, temperature=0.6, n_samples=10, tag="sampling")

print("\n=== All evaluations complete ===")

PYEOF

echo ""
echo "=== Qwen3-4B Thinking EvalPlus complete ==="
echo "Results: ${OUTPUT_BASE}/"
