#!/usr/bin/env bash
# Status of the full-batch gem5 runs: queue state, elected survivor + lock,
# latest checkpointed step, and a recent reward line per estimator.
#   Usage: status.sh [SEED]
set -uo pipefail
SEED="${1:-0}"
STORAGE_ROOT="/work/nvme/bgtv/sramasubramanian/cont-maxrl"
LOGDIR="/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs"
TOTAL=2570

echo "===== queue (pie_fb_*) ====="
squeue -u "${USER}" -o "%.10i %.16j %.12a %.2t %.11M %.11L %R" 2>/dev/null | awk 'NR==1 || /pie_fb_/'

for adv in grpo gos rloo; do
  RUN_TAG="qwen3_1.7b_pie_gem5_fb_${adv}_g16_bs64_s${SEED}"
  OUT="${STORAGE_ROOT}/results/${RUN_TAG}"
  STEP="$(cat "${OUT}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
  OWNER="$(cat "${STORAGE_ROOT}/locks/${RUN_TAG}.lock/owner" 2>/dev/null || echo '-')"
  PCT=$(awk -v s="${STEP}" -v t="${TOTAL}" 'BEGIN{printf "%.1f", (t>0? 100.0*s/t:0)}')
  printf "\n----- %-4s | step %s/%s (%s%%) | lock owner=%s -----\n" "${adv}" "${STEP}" "${TOTAL}" "${PCT}" "${OWNER}"
  # newest console log for the elected job
  LOG=$(ls -t "${LOGDIR}/pie_fb_${adv}_s${SEED}_"*.out 2>/dev/null | head -1)
  if [ -n "${LOG:-}" ]; then
    echo "  log: ${LOG}"
    grep -aoE "training/global_step:[0-9]+|critic/score/mean:[0-9.eE+-]+|critic/score/mean_speedup:[0-9.eE+-]+|timing_s/step:[0-9.]+" "${LOG}" 2>/dev/null \
      | tail -4 | paste -sd' ' -
  fi
done
echo ""
echo "(stop a run cleanly with stop.sh <adv|all> ${SEED})"
