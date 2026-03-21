#!/usr/bin/env bash
#SBATCH --job-name=pilot_eval
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1

# ============================================================
# Pilot Eval: convert FSDP→HF, eval on LCB with 4 GPUs in parallel
# Each GPU runs an independent vLLM instance (TP=1) on a different checkpoint.
# Usage: sbatch pilot_eval.sh <results_dir>
#   e.g. sbatch pilot_eval.sh results/pilot_grpo
# ============================================================

# Activate conda environment (before set -u, conda scripts use unset vars)
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <results_dir>"
  echo "  e.g. sbatch $0 results/pilot_grpo"
  exit 1
fi

RESULTS_DIR="$1"

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# Make results_dir absolute if relative
if [[ "${RESULTS_DIR}" != /* ]]; then
  RESULTS_DIR="${REPO_ROOT}/${RESULTS_DIR}"
fi

# SLURM log dir
EVAL_LOG_DIR="${RESULTS_DIR}/eval_livecodebench"
mkdir -p "${EVAL_LOG_DIR}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL)
# Single-node uses NVLink/shared memory, no libfabric needed
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

# verl/training uses vLLM v0 engine; eval must match (v1 requires triton)
export VLLM_USE_V1=0
# Disable torch.compile (inductor backend requires triton, which is not installed)
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

DATA_FILE="${REPO_ROOT}/data/livecodebench_test.parquet"
BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
N_GPUS_TOTAL=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}

echo "=== Pilot Eval on LiveCodeBench ==="
echo "Results dir: ${RESULTS_DIR}"
echo "GPUs: ${N_GPUS_TOTAL} (each runs independent TP=1 vLLM instance)"
echo ""

# ------------------------------------------------------------------
# Helper: run eval for one checkpoint on one GPU
# ------------------------------------------------------------------
run_eval_on_gpu() {
  local GPU_ID="$1"
  local CHECKPOINT_DIR="$2"
  local OUTPUT_DIR="$3"
  local LOG_FILE="${OUTPUT_DIR}/eval.log"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  TMPDIR="/tmp/${USER}_verl_tmp_gpu${GPU_ID}" \
  $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --data_file "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_completions 200 \
    --temperature 0.8 \
    --max_tokens 1024 \
    --tensor_parallel_size 1 \
    --timeout_per_test 6 \
    --n_workers 64 \
    --k_values 1 2 5 10 20 50 100 200 \
    > "${LOG_FILE}" 2>&1

  echo "  GPU ${GPU_ID}: $(basename "${CHECKPOINT_DIR}") done (see ${LOG_FILE})"
}

# Create per-GPU tmp dirs
for gpu_id in $(seq 0 $((N_GPUS_TOTAL - 1))); do
  mkdir -p "/tmp/${USER}_verl_tmp_gpu${gpu_id}"
done

# ------------------------------------------------------------------
# Phase 1: Merge all FSDP checkpoints to HF (sequential, CPU-only)
# ------------------------------------------------------------------
echo "--- Phase 1: Merge FSDP → HF for all checkpoints ---"

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

# ------------------------------------------------------------------
# Phase 2: Build work queue of (checkpoint_dir, output_dir) pairs
# ------------------------------------------------------------------
declare -a WORK_CKPTS=()
declare -a WORK_OUTDIRS=()
declare -a WORK_STEPS=()

# Step 0: base model
BASE_EVAL_DIR="${RESULTS_DIR}/eval_livecodebench/step_0_base"
mkdir -p "${BASE_EVAL_DIR}"
if [[ -f "${BASE_EVAL_DIR}/eval_results.json" ]]; then
  echo "Step 0 (base): already done, skipping."
else
  WORK_CKPTS+=("${BASE_MODEL}")
  WORK_OUTDIRS+=("${BASE_EVAL_DIR}")
  WORK_STEPS+=(0)
fi

# Training checkpoints
for ACTOR_DIR in "${CKPT_DIRS[@]}"; do
  CKPT_DIR="$(dirname "${ACTOR_DIR}")"
  STEP_NAME="$(basename "${CKPT_DIR}")"
  STEP_NUM="${STEP_NAME#global_step_}"
  HF_DIR="${CKPT_DIR}/actor_hf"
  STEP_EVAL_DIR="${RESULTS_DIR}/eval_livecodebench/${STEP_NAME}"
  mkdir -p "${STEP_EVAL_DIR}"

  if [[ -f "${STEP_EVAL_DIR}/eval_results.json" ]]; then
    echo "${STEP_NAME}: already done, skipping."
  else
    WORK_CKPTS+=("${HF_DIR}")
    WORK_OUTDIRS+=("${STEP_EVAL_DIR}")
    WORK_STEPS+=("${STEP_NUM}")
  fi
done

TOTAL_WORK=${#WORK_CKPTS[@]}
echo ""
echo "--- Phase 2: Evaluate ${TOTAL_WORK} checkpoints on ${N_GPUS_TOTAL} GPUs in parallel ---"

# ------------------------------------------------------------------
# Phase 3: Dispatch work to GPUs, N_GPUS_TOTAL at a time
# ------------------------------------------------------------------
declare -A GPU_PIDS=()
work_idx=0

while [[ ${work_idx} -lt ${TOTAL_WORK} ]]; do
  # Launch up to N_GPUS_TOTAL parallel jobs
  for gpu_id in $(seq 0 $((N_GPUS_TOTAL - 1))); do
    if [[ ${work_idx} -ge ${TOTAL_WORK} ]]; then
      break
    fi

    echo "  Launching step ${WORK_STEPS[${work_idx}]} on GPU ${gpu_id}"
    run_eval_on_gpu "${gpu_id}" "${WORK_CKPTS[${work_idx}]}" "${WORK_OUTDIRS[${work_idx}]}" &
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
# Collect all steps for summary (including previously completed ones)
# ------------------------------------------------------------------
declare -a EVAL_STEPS=()
declare -a EVAL_DIRS=()

EVAL_STEPS+=(0)
EVAL_DIRS+=("${RESULTS_DIR}/eval_livecodebench/step_0_base")

for ACTOR_DIR in "${CKPT_DIRS[@]}"; do
  CKPT_DIR="$(dirname "${ACTOR_DIR}")"
  STEP_NAME="$(basename "${CKPT_DIR}")"
  STEP_NUM="${STEP_NAME#global_step_}"
  EVAL_STEPS+=("${STEP_NUM}")
  EVAL_DIRS+=("${RESULTS_DIR}/eval_livecodebench/${STEP_NAME}")
done

# ------------------------------------------------------------------
# Print summary table
# ------------------------------------------------------------------
echo "============================================================"
echo "Summary: pass@k on LiveCodeBench"
echo "============================================================"
printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s\n" "Step" "pass@1" "pass@10" "pass@100" "mean_score" "uniq_rate" "out_len"
printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s\n" "----" "------" "-------" "--------" "----------" "---------" "-------"

for i in "${!EVAL_STEPS[@]}"; do
  STEP="${EVAL_STEPS[$i]}"
  EDIR="${EVAL_DIRS[$i]}"
  RESULT_FILE="${EDIR}/eval_results.json"
  if [[ -f "${RESULT_FILE}" ]]; then
    read -r P1 P10 P100 MS UR OL < <($PYTHON -c "
import json, sys
d = json.load(open('${RESULT_FILE}'))
pk = d['pass_at_k']
s = d.get('summary', {})
fmt = lambda k: f\"{pk[str(k)]:.4f}\" if str(k) in pk and pk[str(k)] is not None else 'N/A'
fmtv = lambda v: f\"{v:.4f}\" if v is not None else 'N/A'
fmti = lambda v: f\"{v:.0f}\" if v is not None else 'N/A'
print(fmt(1), fmt(10), fmt(100), fmtv(s.get('mean_score')), fmtv(s.get('mean_unique_solution_rate')), fmti(s.get('mean_output_length')))
" 2>/dev/null || echo "ERR ERR ERR ERR ERR ERR")
    printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s\n" "${STEP}" "${P1}" "${P10}" "${P100}" "${MS}" "${UR}" "${OL}"
  else
    printf "%-20s  %10s  %10s  %10s  %10s  %10s  %10s\n" "${STEP}" "MISSING" "MISSING" "MISSING" "MISSING" "MISSING" "MISSING"
  fi
done

# ------------------------------------------------------------------
# Generate eval metrics plot
# ------------------------------------------------------------------
PLOT_FILE="${RESULTS_DIR}/eval_livecodebench/pass_at_k_curve.pdf"
echo ""
echo "Generating plot: ${PLOT_FILE}"

$PYTHON -c "
import json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

eval_dir = '${RESULTS_DIR}/eval_livecodebench'
results_dir_name = os.path.basename('${RESULTS_DIR}')

# Collect results from all step directories
steps = []
pass_at = {1: [], 5: [], 10: [], 100: []}
mean_scores = []
unique_rates = []
output_lengths = []

for entry in sorted(os.listdir(eval_dir)):
    result_file = os.path.join(eval_dir, entry, 'eval_results.json')
    if not os.path.isfile(result_file):
        continue
    if 'step_0_base' in entry:
        step = 0
    elif entry.startswith('global_step_'):
        step = int(entry.split('_')[-1])
    else:
        continue
    with open(result_file) as f:
        data = json.load(f)
    pk = data.get('pass_at_k', {})
    sm = data.get('summary', {})
    steps.append(step)
    for k in pass_at:
        val = pk.get(str(k))
        pass_at[k].append(float(val) if val is not None else None)
    mean_scores.append(sm.get('mean_score'))
    unique_rates.append(sm.get('mean_unique_solution_rate'))
    output_lengths.append(sm.get('mean_output_length'))

if not steps:
    print('No eval results found, skipping plot.')
    sys.exit(0)

# Sort by step
order = sorted(range(len(steps)), key=lambda i: steps[i])
steps = [steps[i] for i in order]
for k in pass_at:
    pass_at[k] = [pass_at[k][i] for i in order]
mean_scores = [mean_scores[i] for i in order]
unique_rates = [unique_rates[i] for i in order]
output_lengths = [output_lengths[i] for i in order]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: pass@k
ax = axes[0, 0]
for k in [1, 5, 10, 100]:
    vals = pass_at[k]
    valid = [(s, v) for s, v in zip(steps, vals) if v is not None]
    if valid:
        ax.plot([s for s, v in valid], [v for s, v in valid],
                marker='o', markersize=4, label=f'pass@{k}')
ax.set_xlabel('Training Step')
ax.set_ylabel('Pass Rate')
ax.set_title('pass@k')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Panel 2: Mean score (fraction of tests passed)
ax = axes[0, 1]
valid = [(s, v) for s, v in zip(steps, mean_scores) if v is not None]
if valid:
    ax.plot([s for s, v in valid], [v for s, v in valid],
            marker='o', markersize=4, color='tab:green')
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Fraction of Tests Passed')
ax.set_title('Mean Score (continuous)')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Panel 3: Unique solution rate
ax = axes[1, 0]
valid = [(s, v) for s, v in zip(steps, unique_rates) if v is not None]
if valid:
    ax.plot([s for s, v in valid], [v for s, v in valid],
            marker='o', markersize=4, color='tab:orange')
ax.set_xlabel('Training Step')
ax.set_ylabel('Unique Solution Rate')
ax.set_title('Diversity (unique solutions / total)')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# Panel 4: Output length
ax = axes[1, 1]
valid = [(s, v) for s, v in zip(steps, output_lengths) if v is not None]
if valid:
    ax.plot([s for s, v in valid], [v for s, v in valid],
            marker='o', markersize=4, color='tab:red')
ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Output Length (chars)')
ax.set_title('Output Length')
ax.grid(True, alpha=0.3)

fig.suptitle(f'LiveCodeBench Eval Metrics ({results_dir_name})', fontsize=14)
plt.tight_layout()
plt.savefig('${PLOT_FILE}', dpi=150)
print(f'Plot saved to ${PLOT_FILE}')
"

echo ""
echo "=== Pilot evaluation complete ==="
