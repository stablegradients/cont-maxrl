#!/usr/bin/env bash
# 1Hz EXTERNAL reaper for the full-batch spray. The launcher already self-reaps,
# but this kills losers faster + is a safety net against a double-election.
#
# For each estimator, the elected survivor is the job id in its durable lock's
# `owner` file (authoritative — same id the launcher itself trusts). Once that
# owner is known, cancel every OTHER job of the same name EXCEPT dependency-held
# ones (the afterany:self resume child the winner queues — must survive). An
# estimator is "resolved" once only its owner (+ any resume child) remains; the
# reaper exits when all three are resolved (or after a deadline).
#   Usage: reap_1hz.sh [SEED]
SEED="${1:-0}"
LOCKROOT="/work/nvme/bgtv/sramasubramanian/cont-maxrl/locks"
ADVS=(grpo gos rloo)
NAMES="pie_fb_grpo_s${SEED},pie_fb_gos_s${SEED},pie_fb_rloo_s${SEED}"
deadline=$(( $(date +%s) + 12*3600 ))    # stop after 12h even if nodes never free
declare -A DONE
echo "$(date +%T) reaper start (seed=${SEED}); watching ${NAMES}"
while :; do
  # one squeue per tick: id|state|dependency|name for all our spray jobs
  mapfile -t ROWS < <(squeue -u "${USER}" -h -n "${NAMES}" -o "%i|%t|%E|%j" 2>/dev/null)
  for adv in "${ADVS[@]}"; do
    [ -n "${DONE[$adv]:-}" ] && continue
    name="pie_fb_${adv}_s${SEED}"
    owner="$(cat "${LOCKROOT}/qwen3_1.7b_pie_gem5_fb_${adv}_g16_bs64_s${SEED}.lock/owner" 2>/dev/null || true)"
    mine=(); for r in "${ROWS[@]}"; do [ "${r##*|}" = "${name}" ] && mine+=("$r"); done
    if [ -z "${owner}" ]; then
      # no elected survivor yet. If <=1 of this name remains and none queued, done.
      [ "${#mine[@]}" -le 1 ] && DONE[$adv]=1
      continue
    fi
    extra=0
    for r in "${mine[@]}"; do
      jid="${r%%|*}"; rest="${r#*|}"; st="${rest%%|*}"; dep="${rest#*|}"; dep="${dep%%|*}"
      [ "${jid}" = "${owner}" ] && continue        # the survivor
      case "${dep}" in after*) continue;; esac     # afterany resume child: keep (squeue shows '(null)' for none)
      if scancel "${jid}" 2>/dev/null; then echo "$(date +%T) reaped ${adv} loser ${jid} (state=${st})"; fi
      extra=$((extra+1))
    done
    [ "${extra}" -eq 0 ] && DONE[$adv]=1            # only owner(+child) remains
  done
  if [ "${#DONE[@]}" -ge 3 ]; then echo "$(date +%T) all 3 estimators resolved — reaper done"; break; fi
  [ "$(date +%s)" -ge "${deadline}" ] && { echo "$(date +%T) deadline reached — reaper exiting"; break; }
  sleep 1
done