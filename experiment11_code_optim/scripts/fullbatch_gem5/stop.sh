#!/usr/bin/env bash
# Cleanly stop a full-batch run: cancels the running survivor AND its already-
# queued afterany resume child (same job name), then clears the durable lock.
#   Usage: stop.sh <grpo|gos|rloo|all> [SEED]
set -uo pipefail
WHICH="${1:?usage: stop.sh <grpo|gos|rloo|all> [SEED]}"
SEED="${2:-0}"
STORAGE_ROOT="/work/nvme/bgtv/sramasubramanian/cont-maxrl"
case "${WHICH}" in all) ADVS=(grpo gos rloo);; grpo|gos|rloo) ADVS=("${WHICH}");; *) echo "bad adv"; exit 2;; esac
for adv in "${ADVS[@]}"; do
  JOBNAME="pie_fb_${adv}_s${SEED}"
  echo "scancel -u ${USER} --name=${JOBNAME} (running + pending resume children)"
  scancel -u "${USER}" --name="${JOBNAME}" 2>/dev/null || true
  rm -rf "${STORAGE_ROOT}/locks/qwen3_1.7b_pie_gem5_fb_${adv}_g16_bs64_s${SEED}.lock" 2>/dev/null || true
done
echo "stopped: ${ADVS[*]} (seed=${SEED}). Re-spray to resume from the last full-state ckpt."
