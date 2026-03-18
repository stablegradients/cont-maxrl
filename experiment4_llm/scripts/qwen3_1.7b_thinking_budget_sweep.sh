#!/usr/bin/env bash
#SBATCH --job-name=qwen3_1.7b_think_sweep
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
# Thinking budget sweep: Qwen3-1.7B (instruct) on LiveCodeBench
# No training -- just inference with enable_thinking=True
# Sweep over max_tokens: 4096, 8192, 16384
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
DATA_FILE="${REPO_ROOT}/data/livecodebench_test.parquet"
OUTPUT_BASE="${REPO_ROOT}/results/qwen3_1.7b_thinking_budget_sweep"
N_GPUS_TOTAL=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}

# Thinking budget sweep: max_tokens controls how much the model can think + code
MAX_TOKENS_LIST=(4096 8192 16384)

echo "=== Qwen3-1.7B Thinking Budget Sweep on LiveCodeBench ==="
echo "Model: ${MODEL} (enable_thinking=True)"
echo "GPUs: ${N_GPUS_TOTAL}"
echo "Sweep: max_tokens = ${MAX_TOKENS_LIST[*]}"
echo "Eval config: 10 completions, T=0.6, top_p=0.95, gpu_mem=0.93"
echo ""

# Create per-GPU tmp dirs
for gpu_id in $(seq 0 $((N_GPUS_TOTAL - 1))); do
  mkdir -p "/tmp/${USER}_verl_tmp_gpu${gpu_id}"
done

# ------------------------------------------------------------------
# Helper: run eval for one (max_tokens) config on one GPU
# ------------------------------------------------------------------
run_eval_on_gpu() {
  local GPU_ID="$1"
  local MAX_TOKENS="$2"
  local OUTPUT_DIR="${OUTPUT_BASE}/max_tokens_${MAX_TOKENS}"
  local LOG_FILE="${OUTPUT_DIR}/eval.log"

  mkdir -p "${OUTPUT_DIR}"

  echo "  GPU ${GPU_ID}: max_tokens=${MAX_TOKENS} -> ${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  TMPDIR="/tmp/${USER}_verl_tmp_gpu${GPU_ID}" \
  $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
    --checkpoint_dir "${MODEL}" \
    --data_file "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_completions 10 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens "${MAX_TOKENS}" \
    --enable_thinking \
    --gpu_memory_utilization 0.93 \
    --max_num_seqs 256 \
    --max_model_len "${MAX_TOKENS}" \
    --tensor_parallel_size 1 \
    --timeout_per_test 5 \
    --n_workers 64 \
    --k_values 1 10 \
    > "${LOG_FILE}" 2>&1

  echo "  GPU ${GPU_ID}: max_tokens=${MAX_TOKENS} done (see ${LOG_FILE})"
}

# ------------------------------------------------------------------
# Dispatch: run up to N_GPUS_TOTAL configs in parallel
# ------------------------------------------------------------------
mkdir -p "${OUTPUT_BASE}"

declare -A GPU_PIDS=()
work_idx=0

while [[ ${work_idx} -lt ${#MAX_TOKENS_LIST[@]} ]]; do
  for gpu_id in $(seq 0 $((N_GPUS_TOTAL - 1))); do
    if [[ ${work_idx} -ge ${#MAX_TOKENS_LIST[@]} ]]; then
      break
    fi

    MT="${MAX_TOKENS_LIST[${work_idx}]}"

    # Skip if already done
    if [[ -f "${OUTPUT_BASE}/max_tokens_${MT}/eval_results.json" ]]; then
      echo "max_tokens=${MT}: already done, skipping."
      work_idx=$((work_idx + 1))
      continue
    fi

    run_eval_on_gpu "${gpu_id}" "${MT}" &
    GPU_PIDS[${gpu_id}]=$!
    work_idx=$((work_idx + 1))
  done

  # Wait for this batch to finish
  for pid in "${GPU_PIDS[@]}"; do
    wait "${pid}" || echo "  WARNING: a GPU eval process failed (exit $?)"
  done
  GPU_PIDS=()
done

echo ""
echo "All evaluations complete."
echo ""

# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
echo "============================================================"
echo "Summary: Thinking Budget Sweep (Qwen3-1.7B, LiveCodeBench)"
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
max_tokens_list = [4096, 8192, 16384]

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
bars = ax.bar([str(b) for b in budgets], p1_vals, color=['#4C72B0', '#55A868', '#C44E52'])
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

fig.suptitle('Qwen3-1.7B Thinking Budget Sweep on LiveCodeBench (no training)', fontsize=14)
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
echo "  thinking_budget_sweep.pdf"
