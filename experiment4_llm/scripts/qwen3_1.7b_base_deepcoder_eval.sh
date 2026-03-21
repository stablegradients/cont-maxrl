#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_base_eval
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Eval: Qwen3-1.7B-Base (trained on DeepCoder) on:
#   1. LCB v5 (167 problems, 200 completions, pass@k)
#   2. EvalPlus HumanEval / HumanEval+ (greedy + sampling)
#   3. EvalPlus MBPP / MBPP+ (greedy + sampling)
#
# Usage: sbatch qwen3_1.7b_base_deepcoder_eval.sh <results_dir>
#   e.g. sbatch qwen3_1.7b_base_deepcoder_eval.sh results/qwen3_1.7b_base_deepcoder_grpo
#        sbatch qwen3_1.7b_base_deepcoder_eval.sh results/qwen3_1.7b_base_deepcoder_maxrl
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <results_dir>"
  exit 1
fi

RESULTS_DIR="$1"

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

if [[ "${RESULTS_DIR}" != /* ]]; then
  RESULTS_DIR="${REPO_ROOT}/${RESULTS_DIR}"
fi

# Conda CUDA
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

BASE_MODEL="Qwen/Qwen3-1.7B-Base"
LCB_DATA="${REPO_ROOT}/data/livecodebench_v5_test.parquet"
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}

# Derive method name from results_dir for wandb
METHOD_NAME=$(basename "${RESULTS_DIR}")
echo "=== Qwen3-1.7B-Base Post-Training Eval ==="
echo "Results dir: ${RESULTS_DIR}"
echo "Method: ${METHOD_NAME}"
echo "Benchmarks: LCB v5 (167), HumanEval/HE+, MBPP/MBPP+"
echo "GPUs: ${N_GPUS}"
echo ""

# ==================================================================
# Phase 1: Merge FSDP checkpoints to HF format
# ==================================================================
echo "--- Phase 1: Merge FSDP -> HF ---"

CKPT_DIRS=()
while IFS= read -r d; do
  CKPT_DIRS+=("$d")
done < <(
  for d in "${RESULTS_DIR}"/global_step_*/actor; do
    [[ -d "$d" ]] && echo "$d"
  done | sort -t/ -k1 -V
)

for ACTOR_DIR in "${CKPT_DIRS[@]}"; do
  CKPT_DIR="$(dirname "${ACTOR_DIR}")"
  STEP_NAME="$(basename "${CKPT_DIR}")"
  HF_DIR="${CKPT_DIR}/actor_hf"

  if [[ -d "${HF_DIR}" ]] && [[ -f "${HF_DIR}/config.json" ]]; then
    echo "  ${STEP_NAME}: HF checkpoint exists, skipping."
  else
    echo "  ${STEP_NAME}: merging..."
    $PYTHON "${REPO_ROOT}/vendor/verl/scripts/model_merger.py" \
      --backend fsdp \
      --hf_model_path "${BASE_MODEL}" \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${HF_DIR}"
  fi
done
echo ""

# ==================================================================
# Phase 2: Build work queue — base model + all checkpoints
# ==================================================================
declare -a ALL_MODEL_PATHS=()
declare -a ALL_STEP_NAMES=()

ALL_MODEL_PATHS+=("${BASE_MODEL}")
ALL_STEP_NAMES+=("step_0_base")

for ACTOR_DIR in "${CKPT_DIRS[@]}"; do
  CKPT_DIR="$(dirname "${ACTOR_DIR}")"
  STEP_NAME="$(basename "${CKPT_DIR}")"
  HF_DIR="${CKPT_DIR}/actor_hf"
  ALL_MODEL_PATHS+=("${HF_DIR}")
  ALL_STEP_NAMES+=("${STEP_NAME}")
done

echo "Models to evaluate: ${#ALL_MODEL_PATHS[@]}"
echo ""

# ==================================================================
# Phase 3: Evaluate each checkpoint
# ==================================================================
for idx in "${!ALL_MODEL_PATHS[@]}"; do
  MODEL_PATH="${ALL_MODEL_PATHS[$idx]}"
  STEP_NAME="${ALL_STEP_NAMES[$idx]}"

  echo "============================================================"
  echo ">>> Evaluating: ${STEP_NAME}"
  echo ">>> Model: ${MODEL_PATH}"
  echo "============================================================"

  # ------ 3a. LCB v5 (200 completions, pass@k) ------
  LCB_OUT="${RESULTS_DIR}/eval_lcbv5/${STEP_NAME}"
  if [[ -f "${LCB_OUT}/eval_results.json" ]]; then
    echo "  LCB v5: already done, skipping."
  else
    echo "  LCB v5: generating 200 completions..."
    mkdir -p "${LCB_OUT}"
    $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
      --checkpoint_dir "${MODEL_PATH}" \
      --data_file "${LCB_DATA}" \
      --output_dir "${LCB_OUT}" \
      --n_completions 200 \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 4096 \
      --tensor_parallel_size "${N_GPUS}" \
      --gpu_memory_utilization 0.93 \
      --max_num_seqs 256 \
      --timeout_per_test 6 \
      --n_workers 256 \
      --k_values 1 2 5 10 20 50 100 200 \
      2>&1 | tee "${LCB_OUT}/eval.log"
    echo "  LCB v5: done."
  fi

  # ------ 3b. EvalPlus (HumanEval, HumanEval+, MBPP, MBPP+) ------
  EVALPLUS_OUT="${RESULTS_DIR}/eval_evalplus/${STEP_NAME}"
  if [[ -f "${EVALPLUS_OUT}/humaneval_results.txt" ]] && [[ -f "${EVALPLUS_OUT}/mbpp_results.txt" ]]; then
    echo "  EvalPlus: already done, skipping."
  else
    echo "  EvalPlus: generating completions..."
    mkdir -p "${EVALPLUS_OUT}"

    $PYTHON -u << PYEOF
import json
import os
import subprocess
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from vllm import LLM, SamplingParams

MODEL = "${MODEL_PATH}"
OUTPUT_DIR = Path("${EVALPLUS_OUT}")
MAX_TOKENS = 1024

print("  Loading model for EvalPlus...")
llm = LLM(
    model=MODEL,
    tokenizer="${BASE_MODEL}",
    trust_remote_code=True,
    gpu_memory_utilization=0.93,
    tensor_parallel_size=${N_GPUS},
    max_num_seqs=256,
    max_model_len=2048,
    dtype="bfloat16",
    enforce_eager=True,
)

STOP_TOKENS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]

def generate_and_evaluate(dataset_name, problems, temperature, n_samples, tag):
    """Generate completions in base-model completion mode, evaluate with evalplus."""
    tag_dir = OUTPUT_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    samples_file = tag_dir / f"{dataset_name}_samples.jsonl"

    if not samples_file.exists():
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
            stop=STOP_TOKENS,
        )

        print(f"    Generating {n_samples} completions for {len(prompts)} {dataset_name} problems...")
        outputs = llm.generate(prompts, sampling_params)

        with open(samples_file, "w") as f:
            for task_id, problem, output in zip(task_ids, problems.values(), outputs):
                for completion in output.outputs:
                    full_code = problem["prompt"] + completion.text
                    f.write(json.dumps({"task_id": task_id, "solution": full_code}) + "\n")
        print(f"    Saved to {samples_file}")
    else:
        print(f"    Cached: {samples_file}")

    # evalplus evaluate — reports HE/HE+ or MBPP/MBPP+ in output
    results_file = tag_dir / f"{dataset_name}_results.txt"
    print(f"    Evaluating {dataset_name} ({tag})...")
    result = subprocess.run(
        ["${PYTHON}", "-m", "evalplus.evaluate",
         "--dataset", dataset_name,
         "--samples", str(samples_file)],
        capture_output=True, text=True,
    )
    output_text = result.stdout + result.stderr
    with open(results_file, "w") as f:
        f.write(output_text)
    print(output_text)

# Load datasets
humaneval = get_human_eval_plus()
mbpp = get_mbpp_plus()

# HumanEval greedy (pass@1)
generate_and_evaluate("humaneval", humaneval, temperature=0.0, n_samples=1, tag="greedy")
# HumanEval sampling (pass@10)
generate_and_evaluate("humaneval", humaneval, temperature=0.6, n_samples=10, tag="sampling")
# MBPP greedy (pass@1)
generate_and_evaluate("mbpp", mbpp, temperature=0.0, n_samples=1, tag="greedy")
# MBPP sampling (pass@10)
generate_and_evaluate("mbpp", mbpp, temperature=0.6, n_samples=10, tag="sampling")

print("  EvalPlus: done.")
PYEOF

    # Copy summary results to top-level for easy access
    for tag in greedy sampling; do
      for bench in humaneval mbpp; do
        src="${EVALPLUS_OUT}/${tag}/${bench}_results.txt"
        if [[ -f "${src}" ]]; then
          cp "${src}" "${EVALPLUS_OUT}/${bench}_${tag}_results.txt"
        fi
      done
    done
  fi

  echo ""
done

# ==================================================================
# Phase 4: Summary table
# ==================================================================
echo "============================================================"
echo "Summary: ${METHOD_NAME}"
echo "============================================================"

$PYTHON -u << 'PYEOF'
import json, os, re, sys

results_dir = os.environ.get("RESULTS_DIR", sys.argv[1] if len(sys.argv) > 1 else ".")

# LCB v5 results
print("\n--- LCB v5 (167 problems) ---")
print(f"{'Step':<25} {'pass@1':>8} {'pass@10':>8} {'pass@100':>8} {'mean_score':>10}")
print("-" * 65)

lcb_dir = os.path.join(results_dir, "eval_lcbv5")
if os.path.isdir(lcb_dir):
    for entry in sorted(os.listdir(lcb_dir)):
        rf = os.path.join(lcb_dir, entry, "eval_results.json")
        if not os.path.isfile(rf):
            continue
        with open(rf) as f:
            d = json.load(f)
        pk = d.get("pass_at_k", {})
        s = d.get("summary", {})
        p1 = pk.get("1", pk.get(1))
        p10 = pk.get("10", pk.get(10))
        p100 = pk.get("100", pk.get(100))
        ms = s.get("mean_score")
        fmt = lambda v: f"{v*100:.1f}%" if v is not None else "N/A"
        print(f"{entry:<25} {fmt(p1):>8} {fmt(p10):>8} {fmt(p100):>8} {fmt(ms):>10}")

# EvalPlus results
print("\n--- EvalPlus ---")
ep_dir = os.path.join(results_dir, "eval_evalplus")
if os.path.isdir(ep_dir):
    for entry in sorted(os.listdir(ep_dir)):
        step_dir = os.path.join(ep_dir, entry)
        if not os.path.isdir(step_dir):
            continue
        print(f"\n  {entry}:")
        for tag in ["greedy", "sampling"]:
            for bench in ["humaneval", "mbpp"]:
                rf = os.path.join(step_dir, tag, f"{bench}_results.txt")
                if os.path.isfile(rf):
                    with open(rf) as f:
                        text = f.read()
                    # Extract pass@k lines
                    for line in text.split("\n"):
                        if "pass@" in line.lower() or "base" in line.lower() or "plus" in line.lower():
                            print(f"    [{bench} {tag}] {line.strip()}")

print("")
PYEOF

echo ""
echo "=== All evaluations complete ==="
