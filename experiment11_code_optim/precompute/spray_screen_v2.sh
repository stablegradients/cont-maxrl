#!/usr/bin/env bash
# Spray the augmented native correctness screen (per-program compile+fails) across
# train/val/test shards. DRY RUN unless --submit.
#   ./spray_screen_v2.sh [--submit] [NTRAIN] [NVAL] [NTEST]
set -euo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
JOB="$REPO/experiment11_code_optim/precompute/screen_v2_job.sh"
ACCTS=(bgtu bgtv bgyb bgyc bgyd bgts bgmo)
SUBMIT="${1:-}"; [ "$SUBMIT" = "--submit" ] && shift || SUBMIT=""
NTRAIN="${1:-48}"; NVAL="${2:-4}"; NTEST="${3:-4}"
i=0
emit() {  # split shard nshards
  local acct="${ACCTS[$((i % ${#ACCTS[@]}))]}"
  local cmd=(sbatch --account="${acct}-dtai-gh" --job-name="g5scr2_${1}_${2}" "$JOB" "$1" "$2" "$3")
  if [ "$SUBMIT" = "--submit" ]; then echo "[$i] ${cmd[*]}"; "${cmd[@]}"; else echo "[dry $i] ${cmd[*]}"; fi
  i=$((i+1))
}
for s in $(seq 0 $((NTRAIN-1))); do emit train "$s" "$NTRAIN"; done
for s in $(seq 0 $((NVAL-1))); do emit val "$s" "$NVAL"; done
for s in $(seq 0 $((NTEST-1))); do emit test "$s" "$NTEST"; done
echo "total ${i} screen jobs$([ "$SUBMIT" = "--submit" ] || echo ' — DRY RUN, pass --submit')"
