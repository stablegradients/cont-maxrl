#!/usr/bin/env bash
# Spray the partial-program re-time (fixed pipeline, native pre-screen): train +
# val + TEST sharded. DRY RUN unless --submit.
#   ./spray_retime.sh [--submit] [NTRAIN] [NVAL] [NTEST]
set -euo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
JOB="$REPO/experiment11_code_optim/precompute/retime_job.sh"
ACCTS=(bgtu bgtv bgyb bgyc bgyd bgts bgmo)
SUBMIT="${1:-}"; [ "$SUBMIT" = "--submit" ] && shift || SUBMIT=""
NTRAIN="${1:-48}"; NVAL="${2:-4}"; NTEST="${3:-8}"
i=0
emit() {  # split shard nshards
  local acct="${ACCTS[$((i % ${#ACCTS[@]}))]}"
  local cmd=(sbatch --account="${acct}-dtai-gh" --job-name="g5rt_${1}_${2}" "$JOB" "$1" "$2" "$3")
  if [ "$SUBMIT" = "--submit" ]; then echo "[$i] ${cmd[*]}"; "${cmd[@]}"; else echo "[dry $i] ${cmd[*]}"; fi
  i=$((i+1))
}
for s in $(seq 0 $((NTRAIN-1))); do emit train "$s" "$NTRAIN"; done
for s in $(seq 0 $((NVAL-1))); do emit val "$s" "$NVAL"; done
for s in $(seq 0 $((NTEST-1))); do emit test "$s" "$NTEST"; done
echo "total ${i} retime jobs$([ "$SUBMIT" = "--submit" ] || echo ' — DRY RUN, pass --submit')"
