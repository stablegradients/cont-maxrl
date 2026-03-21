#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_think_sweep_v5
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
# Thinking budget sweep: Qwen3-1.7B (instruct) on LCB v5
# FIXED to match Qwen3 technical report methodology:
#   1. v5-only dataset (167 problems, not cumulative release_v5)
#   2. Relaxed system prompt (no "Output ONLY..." restriction)
#   3. top_k=20
# Sweep: 4k -> 8k -> 16k -> 32k (sequential, TP=4 each)
# Paper reference: arxiv 2505.09388, page 13
# ============================================================

# Activate conda environment (before set -u, conda scripts use unset vars)
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL)
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# Caches
export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

# Project must be importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# verl/training uses vLLM v0 engine; eval must match
export VLLM_USE_V1=0
# Disable torch.compile (inductor backend requires triton, which is not installed)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

MODEL="Qwen/Qwen3-1.7B"
DATA_FILE="${REPO_ROOT}/data/livecodebench_v5_test.parquet"
OUTPUT_BASE="${REPO_ROOT}/results/qwen3_1.7b_thinking_budget_sweep"

# Official LCB system prompt
SYSTEM_PROMPT="You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

# Thinking budget sweep: 4k -> 8k -> 16k -> 32k (sequential phases, TP=4 each)
MAX_TOKENS_LIST=(4096 8192 16384 32768)

echo "=== Qwen3-1.7B Thinking Budget Sweep on LCB (official prompt, 876 problems) ==="
echo "Model: ${MODEL} (enable_thinking=True)"
echo "Dataset: release_v5 (876 problems)"
echo "Fixes: official LCB prompt+starter code, class Solution sandbox, top_k=20"
echo "Sweep: max_tokens = ${MAX_TOKENS_LIST[*]} (sequential, TP=4)"
echo "Eval config: 10 completions, T=0.6, top_p=0.95, top_k=20, gpu_mem=0.93"
echo ""

mkdir -p "${OUTPUT_BASE}"

# ------------------------------------------------------------------
# Sequential phases: each budget uses all 4 GPUs (TP=4)
# ------------------------------------------------------------------
for MT in "${MAX_TOKENS_LIST[@]}"; do
  OUTPUT_DIR="${OUTPUT_BASE}/max_tokens_${MT}"

  if [[ -f "${OUTPUT_DIR}/eval_results.json" ]]; then
    echo ">>> Phase max_tokens=${MT}: already done, skipping."
    continue
  fi

  echo ">>> Phase max_tokens=${MT} (TP=4, all GPUs)"
  mkdir -p "${OUTPUT_DIR}"

  $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
    --checkpoint_dir "${MODEL}" \
    --data_file "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_completions 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20 \
    --max_tokens "${MT}" \
    --enable_thinking \
    --system_prompt "${SYSTEM_PROMPT}" \
    --gpu_memory_utilization 0.93 \
    --max_num_seqs 256 \
    --max_model_len "${MT}" \
    --tensor_parallel_size 4 \
    --timeout_per_test 6 \
    --n_workers 256 \
    --k_values 1 10 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

  echo ">>> Phase max_tokens=${MT} complete."
  echo ""
done

echo ""
echo "All phases complete."
echo ""

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
echo "============================================================"
echo "Summary: Thinking Budget Sweep (Qwen3-1.7B, LCB v5)"
echo "============================================================"
printf "%-15s  %10s  %10s  %10s  %10s  %10s\n" \
  "max_tokens" "pass@1" "pass@10" "mean_score" "uniq_rate" "out_len"
printf "%-15s  %10s  %10s  %10s  %10s  %10s\n" \
  "----------" "------" "-------" "----------" "---------" "-------"

for MT in "${MAX_TOKENS_LIST[@]}"; do
  RESULT_FILE="${OUTPUT_BASE}/max_tokens_${MT}/eval_results.json"
  if [[ -f "${RESULT_FILE}" ]]; then
    read -r P1 P10 MS UR OL < <($PYTHON -c "
import json, sys
d = json.load(open('${RESULT_FILE}'))
pk = d['pass_at_k']
s = d.get('summary', {})
fmt = lambda k: f\"{pk[str(k)]:.4f}\" if str(k) in pk and pk[str(k)] is not None else 'N/A'
fmtv = lambda v: f\"{v:.4f}\" if v is not None else 'N/A'
fmti = lambda v: f\"{v:.0f}\" if v is not None else 'N/A'
print(fmt(1), fmt(10), fmtv(s.get('mean_score')), fmtv(s.get('mean_unique_solution_rate')), fmti(s.get('mean_output_length')))
" 2>/dev/null || echo "ERR ERR ERR ERR ERR")
    printf "%-15s  %10s  %10s  %10s  %10s  %10s\n" "${MT}" "${P1}" "${P10}" "${MS}" "${UR}" "${OL}"
  else
    printf "%-15s  %10s  %10s  %10s  %10s  %10s\n" "${MT}" "MISSING" "MISSING" "MISSING" "MISSING" "MISSING"
  fi
done

# ------------------------------------------------------------------
# Generate comparison plot
# ------------------------------------------------------------------
PLOT_FILE="${OUTPUT_BASE}/thinking_budget_sweep.pdf"
echo ""
echo "Generating plot: ${PLOT_FILE}"

$PYTHON -c "
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

output_base = '${OUTPUT_BASE}'
max_tokens_list = [4096, 8192, 16384, 32768]

# Collect results
data = {}
for mt in max_tokens_list:
    result_file = os.path.join(output_base, f'max_tokens_{mt}', 'eval_results.json')
    if os.path.isfile(result_file):
        with open(result_file) as f:
            data[mt] = json.load(f)

if not data:
    print('No eval results found, skipping plot.')
    sys.exit(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: pass@k curves for each budget
ax = axes[0]
k_values = [1, 10]
for mt in sorted(data.keys()):
    pk = data[mt]['pass_at_k']
    ks, vals = [], []
    for k in k_values:
        v = pk.get(str(k))
        if v is not None:
            ks.append(k)
            vals.append(v)
    ax.plot(ks, vals, marker='o', markersize=5, label=f'max_tokens={mt}')
ax.set_xlabel('k')
ax.set_ylabel('pass@k')
ax.set_title('pass@k vs Thinking Budget')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Panel 2: pass@1 bar chart
ax = axes[1]
budgets = sorted(data.keys())
p1_vals = []
for mt in budgets:
    v = data[mt]['pass_at_k'].get('1')
    p1_vals.append(v if v is not None else 0)
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2'][:len(budgets)]
bars = ax.bar([str(b) for b in budgets], p1_vals, color=colors)
for bar, val in zip(bars, p1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
ax.set_xlabel('max_tokens (thinking budget)')
ax.set_ylabel('pass@1')
ax.set_title('pass@1 vs Thinking Budget')
ax.set_ylim(0, max(p1_vals) * 1.3 if p1_vals else 1)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: mean output length & mean score
ax = axes[2]
lengths = [data[mt]['summary'].get('mean_output_length', 0) for mt in budgets]
scores = [data[mt]['summary'].get('mean_score', 0) for mt in budgets]
x = np.arange(len(budgets))
width = 0.35
ax2 = ax.twinx()
bars1 = ax.bar(x - width/2, lengths, width, label='Mean output length (chars)', color='#4C72B0', alpha=0.7)
bars2 = ax2.bar(x + width/2, scores, width, label='Mean score', color='#55A868', alpha=0.7)
ax.set_xlabel('max_tokens (thinking budget)')
ax.set_ylabel('Mean Output Length (chars)')
ax2.set_ylabel('Mean Score')
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in budgets])
ax.set_title('Output Length & Score vs Budget')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Qwen3-1.7B Thinking Budget Sweep on LCB v5 (paper methodology)', fontsize=14)
plt.tight_layout()
plt.savefig('${PLOT_FILE}', dpi=150)
print(f'Plot saved to ${PLOT_FILE}')
"

echo ""
echo "=== Thinking budget sweep complete ==="
echo "Results: ${OUTPUT_BASE}/"
echo "  max_tokens_4096/eval_results.json"
echo "  max_tokens_8192/eval_results.json"
echo "  max_tokens_16384/eval_results.json"
echo "  max_tokens_32768/eval_results.json"
echo "  thinking_budget_sweep.pdf"
