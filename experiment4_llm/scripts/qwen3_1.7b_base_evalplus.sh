#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_base_evalplus
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-1.7B-Base on HumanEval+ and MBPP+ (standalone Python)
# Base (pretrained) model — plain text prompts, no chat template
# max_new_tokens=1024
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

OUTPUT_BASE="${REPO_ROOT}/results/qwen3_1.7b_base_evalplus"
mkdir -p "${OUTPUT_BASE}"

echo "=== Qwen3-1.7B-Base EvalPlus (HumanEval+ & MBPP+) ==="
echo "Model: Qwen/Qwen3-1.7B-Base (base, completion mode)"
echo "max_new_tokens=1024"
echo ""

$PYTHON -u << 'PYEOF'
import json
import os
import subprocess
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-1.7B-Base"
OUTPUT_BASE = Path(os.environ["REPO_ROOT"]) / "results" / "qwen3_1.7b_base_evalplus"
MAX_TOKENS = 1024


def generate_and_evaluate(dataset_name, problems, llm, temperature, n_samples, tag):
    """Generate completions using base model (completion mode), save as evalplus jsonl, evaluate."""
    output_dir = OUTPUT_BASE / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_file = output_dir / f"{dataset_name}_samples.jsonl"

    if samples_file.exists():
        print(f"  Cached: {samples_file}")
    else:
        # Base model: use the raw prompt as prefix (completion mode)
        prompts = []
        task_ids = []
        for task_id, problem in problems.items():
            prompts.append(problem["prompt"])
            task_ids.append(task_id)

        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            top_k=20 if temperature > 0 else -1,
            max_tokens=MAX_TOKENS,
            stop=["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("],
        )

        print(f"  Generating {n_samples} completions for {len(prompts)} problems...")
        outputs = llm.generate(prompts, sampling_params)

        # Save as evalplus jsonl with "solution" key (full standalone code)
        with open(samples_file, "w") as f:
            for task_id, problem, output in zip(task_ids, problems.values(), outputs):
                for completion in output.outputs:
                    # Prepend the prompt to make it a complete solution
                    full_code = problem["prompt"] + completion.text
                    sample = {
                        "task_id": task_id,
                        "solution": full_code,
                    }
                    f.write(json.dumps(sample) + "\n")

        print(f"  Saved to {samples_file}")

    # Run evalplus evaluate
    print(f"  Evaluating {dataset_name}...")
    results_file = output_dir / f"{dataset_name}_results.txt"
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

# ---- Load datasets ----
print("Loading HumanEval+ and MBPP+...")
humaneval = get_human_eval_plus()
mbpp = get_mbpp_plus()

# ---- 1. HumanEval+ greedy (pass@1) ----
print("\n>>> HumanEval+ (greedy, pass@1)")
generate_and_evaluate("humaneval", humaneval, llm, temperature=0.0, n_samples=1, tag="greedy")

# ---- 2. HumanEval+ sampling (pass@10) ----
print("\n>>> HumanEval+ (T=0.6, n=10, pass@10)")
generate_and_evaluate("humaneval", humaneval, llm, temperature=0.6, n_samples=10, tag="sampling")

# ---- 3. MBPP+ greedy (pass@1) ----
print("\n>>> MBPP+ (greedy, pass@1)")
generate_and_evaluate("mbpp", mbpp, llm, temperature=0.0, n_samples=1, tag="greedy")

# ---- 4. MBPP+ sampling (pass@10) ----
print("\n>>> MBPP+ (T=0.6, n=10, pass@10)")
generate_and_evaluate("mbpp", mbpp, llm, temperature=0.6, n_samples=10, tag="sampling")

print("\n=== All evaluations complete ===")

PYEOF

echo ""
echo "=== Qwen3-1.7B-Base EvalPlus complete ==="
echo "Results: ${OUTPUT_BASE}/"
