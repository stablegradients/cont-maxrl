#!/usr/bin/env bash
#SBATCH --job-name=pilot_eval
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=860G
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
set -euo pipefail

# ============================================================
# Pilot Eval: iterate over checkpoints, convert FSDP→HF, eval on LCB
# Usage: sbatch pilot_eval.sh <results_dir>
#   e.g. sbatch pilot_eval.sh results/pilot_grpo
# ============================================================

# Activate conda environment
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

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

# Caches
export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TMPDIR}"

# Project must be importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
DATA_FILE="${REPO_ROOT}/data/livecodebench_test.parquet"
BASE_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"

echo "=== Pilot Eval on LiveCodeBench ==="
echo "Results dir: ${RESULTS_DIR}"
echo "GPUs: ${N_GPUS}"
echo ""

# Collect (step, checkpoint_path) pairs for the summary table
declare -a EVAL_STEPS=()
declare -a EVAL_DIRS=()

# ------------------------------------------------------------------
# Step 0: evaluate base model
# ------------------------------------------------------------------
echo "--- Step 0: base model (${BASE_MODEL}) ---"
BASE_EVAL_DIR="${RESULTS_DIR}/eval_livecodebench/step_0_base"
mkdir -p "${BASE_EVAL_DIR}"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
  --checkpoint_dir "${BASE_MODEL}" \
  --data_file "${DATA_FILE}" \
  --output_dir "${BASE_EVAL_DIR}" \
  --n_completions 200 \
  --temperature 0.8 \
  --max_tokens 1024 \
  --tensor_parallel_size ${N_GPUS} \
  --timeout_per_test 1 \
  --n_workers 256 \
  --k_values 1 2 5 10 20 50 100 200

EVAL_STEPS+=(0)
EVAL_DIRS+=("${BASE_EVAL_DIR}")
echo ""

# ------------------------------------------------------------------
# Iterate over training checkpoints (sorted numerically)
# ------------------------------------------------------------------
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
  STEP_NAME="$(basename "${CKPT_DIR}")"          # e.g. global_step_100
  STEP_NUM="${STEP_NAME#global_step_}"            # e.g. 100
  HF_DIR="${CKPT_DIR}/actor_hf"
  STEP_EVAL_DIR="${RESULTS_DIR}/eval_livecodebench/${STEP_NAME}"
  mkdir -p "${STEP_EVAL_DIR}"

  echo "--- ${STEP_NAME}: merge FSDP → HF ---"
  if [[ -d "${HF_DIR}" ]] && [[ -f "${HF_DIR}/config.json" ]]; then
    echo "  HF checkpoint already exists, skipping merge."
  else
    $PYTHON "${REPO_ROOT}/vendor/verl/scripts/model_merger.py" \
      --backend fsdp \
      --hf_model_path "${BASE_MODEL}" \
      --local_dir "${ACTOR_DIR}" \
      --target_dir "${HF_DIR}"
  fi

  echo "--- ${STEP_NAME}: evaluate on LiveCodeBench ---"
  if [[ -f "${STEP_EVAL_DIR}/eval_results.json" ]]; then
    echo "  Eval results already exist, skipping."
  else
    $PYTHON "${REPO_ROOT}/experiment4_llm/src/evaluate.py" \
      --checkpoint_dir "${HF_DIR}" \
      --data_file "${DATA_FILE}" \
      --output_dir "${STEP_EVAL_DIR}" \
      --n_completions 200 \
      --temperature 0.8 \
      --max_tokens 1024 \
      --tensor_parallel_size ${N_GPUS} \
      --timeout_per_test 1 \
      --n_workers 256 \
      --k_values 1 2 5 10 20 50 100 200
  fi

  EVAL_STEPS+=("${STEP_NUM}")
  EVAL_DIRS+=("${STEP_EVAL_DIR}")
  echo ""
done

# ------------------------------------------------------------------
# Print summary table
# ------------------------------------------------------------------
echo "============================================================"
echo "Summary: pass@k on LiveCodeBench"
echo "============================================================"
printf "%-20s  %10s  %10s  %10s\n" "Step" "pass@1" "pass@10" "pass@100"
printf "%-20s  %10s  %10s  %10s\n" "----" "------" "-------" "--------"

for i in "${!EVAL_STEPS[@]}"; do
  STEP="${EVAL_STEPS[$i]}"
  EDIR="${EVAL_DIRS[$i]}"
  RESULT_FILE="${EDIR}/eval_results.json"
  if [[ -f "${RESULT_FILE}" ]]; then
    # eval_results.json has {"pass_at_k": {"1": 0.xx, "10": 0.xx, ...}, ...}
    read -r P1 P10 P100 < <($PYTHON -c "
import json, sys
d = json.load(open('${RESULT_FILE}'))
pk = d['pass_at_k']
fmt = lambda k: f\"{pk[str(k)]:.4f}\" if str(k) in pk and pk[str(k)] is not None else 'N/A'
print(fmt(1), fmt(10), fmt(100))
" 2>/dev/null || echo "ERR ERR ERR")
    printf "%-20s  %10s  %10s  %10s\n" "${STEP}" "${P1}" "${P10}" "${P100}"
  else
    printf "%-20s  %10s  %10s  %10s\n" "${STEP}" "MISSING" "MISSING" "MISSING"
  fi
done

echo ""
echo "=== Pilot evaluation complete ==="
