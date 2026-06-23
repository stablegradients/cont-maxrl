#!/usr/bin/env bash
# Spray the dataset gem5 sweep: train sharded NSHARDS ways + val + test,
# round-robin over the 7 accounts. DRY RUN unless --submit.
#   ./spray_sweep.sh [--submit] [NSHARDS_TRAIN]
set -euo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
JOB="$REPO/experiment11_code_optim/precompute/sweep_job.sh"
ACCTS=(bgtu bgtv bgyb bgyc bgyd bgts bgmo)
SUBMIT="${1:-}"; [ "$SUBMIT" = "--submit" ] && shift || SUBMIT=""
NTRAIN="${1:-48}"        # 1-GPU jobs backfill easily -> shard fine for speed
NSMALL="${2:-4}"         # shards for val/test/synthetic_only
i=0
emit() {  # split shard nshards
  local acct="${ACCTS[$((i % ${#ACCTS[@]}))]}"
  local cmd=(sbatch --account="${acct}-dtai-gh" --job-name="g5sw_${1}_${2}" "$JOB" "$1" "$2" "$3")
  if [ "$SUBMIT" = "--submit" ]; then echo "[$i] ${cmd[*]}"; "${cmd[@]}"; else echo "[dry $i] ${cmd[*]}"; fi
  i=$((i+1))
}
# Disjoint program sets (train ⊂ train_with_synthetic, so synthetic_only =
# the 2,970 net-new programs only — train is never timed twice):
#   train (40,625) sharded NTRAIN ways; val + test + synthetic_only small.
for s in $(seq 0 $((NTRAIN-1))); do emit train "$s" "$NTRAIN"; done
for s in $(seq 0 $((NSMALL-1))); do emit val "$s" "$NSMALL"; done
for s in $(seq 0 $((NSMALL-1))); do emit test "$s" "$NSMALL"; done
for s in $(seq 0 $((NSMALL-1))); do emit synthetic_only "$s" "$NSMALL"; done
echo "total ${i} jobs$([ "$SUBMIT" = "--submit" ] || echo ' — DRY RUN, pass --submit')"
