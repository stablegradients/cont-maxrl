#!/usr/bin/env bash
# Driver: 15-job sweep covering 6 Qwen3 baselines × {sampled, greedy} × {chat, raw+1shot for Base}
#
# Sweep matrix:
#   3 Base models   × (chat n=16 T=1.0) × (raw_oneshot n=16 T=1.0) × (raw_oneshot n=1 T=0.0)   = 9 jobs
#   3 Instruct      × (chat n=16 T=1.0) × (chat n=1 T=0.0)                                      = 6 jobs
# Total: 15.
#
# Submit with:
#   bash eval_qwen3_baselines.sh             # dry-run: prints sbatch commands
#   bash eval_qwen3_baselines.sh --submit    # actually submits
#
# Per-job allocation: ghx4 whole-node (4 GPUs) for 1h30m. Per-model run-time ~30-60 min.

set -eo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
ONESHOT="${REPO}/experiment11_code_optim/scripts/baseline_eval_oneshot_preamble.txt"
JOB_SCRIPT="${REPO}/experiment11_code_optim/scripts/eval_qwen3_baseline_one.sh"
SUBMIT=0
if [ "${1:-}" = "--submit" ]; then SUBMIT=1; fi

ACCOUNTS=(bgmo bgtu bgts bgtv bgyb bgyc bgyd)
ACCT_IDX=0

submit_one() {
  local MODEL_ID="$1"
  local TEMPLATE_MODE="$2"
  local N_COMPLETIONS="$3"
  local TEMPERATURE="$4"
  local OUTPUT_TAG="$5"
  local ACCT="${ACCOUNTS[$((ACCT_IDX % ${#ACCOUNTS[@]}))]}-dtai-gh"
  ACCT_IDX=$((ACCT_IDX + 1))

  # EOS list is derived inside the inner script from MODEL_ID's suffix
  # (-Base → 151643, else → 151645,151643). Avoids the sbatch --export
  # comma-in-value parsing issue.
  local EXPORT_VARS="ALL,MODEL_ID=${MODEL_ID},TEMPLATE_MODE=${TEMPLATE_MODE},N_COMPLETIONS=${N_COMPLETIONS},TEMPERATURE=${TEMPERATURE},OUTPUT_TAG=${OUTPUT_TAG}"
  if [ "${TEMPLATE_MODE}" = "raw_oneshot" ]; then
    EXPORT_VARS="${EXPORT_VARS},ONESHOT_FILE=${ONESHOT}"
  fi

  local CMD="sbatch --account=${ACCT} --export=${EXPORT_VARS} --job-name=qwen3eval_${OUTPUT_TAG} ${JOB_SCRIPT}"
  if [ $SUBMIT -eq 1 ]; then
    echo "+ ${CMD}"
    eval "${CMD}"
  else
    echo "${CMD}"
  fi
}

echo "=== Qwen3 baseline PIE eval sweep ==="
echo "submit-mode: $([ $SUBMIT -eq 1 ] && echo YES || echo DRY-RUN)"
echo

# --- Base models (3 sizes) × 3 variants each = 9 jobs
for SIZE in 0.6B 1.7B 4B; do
  MODEL="Qwen/Qwen3-${SIZE}-Base"
  # B1: chat template, sampled
  submit_one "${MODEL}" chat        16 1.0 "base_${SIZE}_chat_n16_t1"
  # B2: raw + 1-shot, sampled
  submit_one "${MODEL}" raw_oneshot 16 1.0 "base_${SIZE}_oneshot_n16_t1"
  # Greedy anchor (B2 path is most informative)
  submit_one "${MODEL}" raw_oneshot 1  0.0 "base_${SIZE}_oneshot_n1_t0"
done

# --- Instruct models (3 sizes) × 2 variants each = 6 jobs
for SIZE in 0.6B 1.7B 4B; do
  MODEL="Qwen/Qwen3-${SIZE}"
  submit_one "${MODEL}" chat 16 1.0 "instruct_${SIZE}_chat_n16_t1"
  submit_one "${MODEL}" chat 1  0.0 "instruct_${SIZE}_chat_n1_t0"
done

echo
echo "=== sweep size: 15 jobs ==="
