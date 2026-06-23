#!/usr/bin/env bash
# ============================================================================
# Spray the full-batch gem5 launcher across all accounts for one estimator
# (or all three). The launcher's election makes the FIRST copy to start the
# sole survivor and self-cancels the rest, so over-spraying only speeds pickup.
#
#   Usage: spray_fullbatch.sh <grpo|gos|rloo|all> [SEED] [SEGMENTS]
#   e.g.   spray_fullbatch.sh all          # 3 estimators x 7 accounts = 21 jobs
#          spray_fullbatch.sh gos 0 8      # gos only, seed 0, 8-segment budget
# ============================================================================
set -euo pipefail
WHICH="${1:?usage: spray_fullbatch.sh <grpo|gos|rloo|all> [SEED] [SEGMENTS]}"
SEED="${2:-0}"
SEG="${3:-8}"
REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
SELF="${REPO_ROOT}/experiment11_code_optim/scripts/fullbatch_gem5/run_fullbatch_gem5.sh"
ACCTS=(bgtu bgtv bgyb bgyc bgyd bgts bgmo)

case "${WHICH}" in
  all) ADVS=(grpo gos rloo);;
  grpo|gos|rloo) ADVS=("${WHICH}");;
  *) echo "FATAL: first arg must be grpo|gos|rloo|all"; exit 2;;
esac

# WANDB key must exist (the launcher hard-fails without it); confirm here too.
if [ -f "${REPO_ROOT}/.env" ]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
: "${WANDB_API_KEY:?WANDB_API_KEY must be set (source .env) before spraying}"

for adv in "${ADVS[@]}"; do
  JOBNAME="pie_fb_${adv}_s${SEED}"
  echo "=== spraying ${adv} (seed=${SEED}, segments=${SEG}) as ${JOBNAME} across ${#ACCTS[@]} accounts ==="
  for a in "${ACCTS[@]}"; do
    JID=$(sbatch --parsable --account="${a}-dtai-gh" --job-name="${JOBNAME}" \
            "${SELF}" "${adv}" "${SEED}" "${SEG}" 2>&1) \
      && echo "  ${a}-dtai-gh -> job ${JID}" \
      || echo "  ${a}-dtai-gh -> SUBMIT FAILED: ${JID}"
  done
done
echo "=== sprayed. The first copy of each estimator to START running wins; siblings self-cancel/are reaped. ==="
echo "    watch:  experiment11_code_optim/scripts/fullbatch_gem5/status.sh"
