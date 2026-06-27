#!/usr/bin/env bash
# Election reaper for the cross-account prerl spray. Each config is sprayed to
# several accounts under the shared job-name prerl_<config>. As soon as a config
# has a RUNNING winner, kill its still-PENDING siblings (free the other accounts'
# slots). Once a config's gen_<config>.json lands, kill any leftover siblings.
# If a config ever loses ALL jobs without producing output, self-heal by
# resubmitting. Only ever scancel PENDING jobs (never a RUNNING one mid-write).
set -uo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
OUT="${REPO}/experiment11_code_optim/results/prerl_analysis"
RUN="${REPO}/experiment11_code_optim/eval/run_prerl.sh"
CFGS="inst17b_P1 inst17b_P2 inst17b_P3 inst2507_P1 inst2507_P2 inst2507_P3 think2507_P1 think2507_P2"
ACCTS=(bgtu bgtv bgyb bgyc bgyd bgts bgmo)
U=$(whoami)
cd "$OUT"

for i in $(seq 1 300); do
  allgen=1
  for cfg in $CFGS; do
    name="prerl_${cfg}"; gen="gen_${cfg}.json"
    if [ -f "$gen" ]; then
      pend=$(squeue -u "$U" -h -n "$name" -t PENDING -o "%i" 2>/dev/null)
      [ -n "$pend" ] && { scancel $pend 2>/dev/null; echo "[$i $(date +%H:%M)] $cfg DONE -> killed pending $(echo $pend | tr '\n' ' ')"; }
      continue
    fi
    allgen=0
    run=$(squeue -u "$U" -h -n "$name" -t RUNNING -o "%i" 2>/dev/null)
    pend=$(squeue -u "$U" -h -n "$name" -t PENDING -o "%i" 2>/dev/null)
    if [ -n "$run" ]; then
      [ -n "$pend" ] && { scancel $pend 2>/dev/null; echo "[$i $(date +%H:%M)] $cfg elected $(echo $run | tr '\n' ' ')-> killed pending $(echo $pend | tr '\n' ' ')"; }
    elif [ -z "$pend" ]; then
      # no running, no pending, no output -> self-heal: resubmit to 2 accounts
      a1=${ACCTS[$((i % 7))]}; a2=${ACCTS[$(((i + 3) % 7))]}
      for a in "$a1" "$a2"; do
        sbatch --parsable --account=${a}-dtai-gh -J "$name" --export=ALL,ONLY_CONFIG=${cfg} "$RUN" >/dev/null 2>&1
      done
      echo "[$i $(date +%H:%M)] $cfg had NO jobs/output -> resubmitted to $a1,$a2"
    fi
  done
  [ "$allgen" = "1" ] && { echo "ALL 5 GEN FILES PRESENT"; break; }
  sleep 40
done
echo "ELECTION DONE -> $(ls gen_*.json 2>/dev/null | grep -v smoke | wc -l)/5 gen files"
ls -la "$OUT"/gen_*.json 2>/dev/null | grep -v smoke