#!/usr/bin/env bash
# ============================================================================
# Spray the n=1024 best_reward@k test eval across BOTH ghx4 and ghx4-interactive
# x 7 accounts (14 combos, round-robin) to grab nodes fastest. For each arm, pin
# its LATEST complete carveout snapshot and fan N_SHARDS 1-GPU cells. Shards are
# sized to finish well under the 2h interactive cap, so a cell runs on either
# partition. Cells are idempotent -> re-running fills only missing shards.
#   Usage: spray_besteval.sh <N_SHARDS> [arms...]     (default 16 grpo gos rloo)
#   Pin steps with STEP_grpo=500 STEP_gos=300 STEP_rloo=500 env vars.
# ============================================================================
set -euo pipefail
N_SHARDS="${1:-16}"; shift || true
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(grpo gos rloo)
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
CELL="${REPO}/experiment11_code_optim/eval/run_besteval_cell.sh"
STORAGE=/work/nvme/bgtv/sramasubramanian/cont-maxrl
ACCTS=(bgtu bgtv bgyb bgyc bgyd bgts bgmo)
PARTS=(ghx4 ghx4-interactive)
[ -f "${REPO}/.env" ] && { set -a; source "${REPO}/.env"; set +a; }

COMBOS=(); for p in "${PARTS[@]}"; do for a in "${ACCTS[@]}"; do COMBOS+=("${a}|${p}"); done; done

latest_step() {
  local d="${STORAGE}/results/qwen3_1.7b_pie_gem5_fb_${1}_g16_bs64_s0/actor_ckpts" best=0 n
  for s in "$d"/step_*; do [ -f "$s/.carveout_complete" ] || continue; n="${s##*step_}"; [ "$n" -gt "$best" ] && best="$n"; done
  echo "$best"
}

ci=0; RUNS=""
for arm in "${ARMS[@]}"; do
  if [ "${arm}" = "base" ]; then
    STEP=0                                   # untrained init; cell ignores STEP for base -> base_step0
  else
    pin="STEP_${arm}"; STEP="${!pin:-$(latest_step "$arm")}"
    [ "${STEP}" = "0" ] && { echo "SKIP ${arm}: no complete snapshot"; continue; }
  fi
  RUNS="${RUNS} ${arm}_step${STEP}"
  echo "=== ${arm}: step ${STEP} | ${N_SHARDS} shards across ${#COMBOS[@]} (acct x partition) combos ==="
  for sh in $(seq 0 $((N_SHARDS-1))); do
    combo="${COMBOS[$((ci % ${#COMBOS[@]}))]}"; ci=$((ci+1))
    acct="${combo%|*}"; part="${combo#*|}"
    jid=$(sbatch --parsable --account="${acct}-dtai-gh" --partition="${part}" --time=2:00:00 \
            --job-name="besteval_${arm}_s${STEP}" \
            "${CELL}" "${arm}" "${STEP}" "${sh}" "${N_SHARDS}" 2>&1) \
      && echo "  ${arm} sh ${sh}/${N_SHARDS} -> ${acct}/${part} job ${jid}" \
      || echo "  ${arm} sh ${sh} SUBMIT FAILED: ${jid}"
  done
done
echo ""
echo "watch:     squeue -u ${USER} -h -n besteval_grpo_s500,besteval_gos_s300,besteval_rloo_s500 | wc -l"
echo "aggregate: python -m experiment11_code_optim.eval.besteval_aggregate --eval_root ${STORAGE}/besteval --runs${RUNS}"
