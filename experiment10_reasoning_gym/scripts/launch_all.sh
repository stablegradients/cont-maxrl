#!/usr/bin/env bash
# Batch launcher for Sweep A + Sweep B (spec §6.3).
#
# Sweep A — Estimator comparison (GOS+continuous vs MaxRL+binary)
#   word_sorting    difficulties: 4, 6, 8, 10, 12
#   knights_knaves  difficulties: 3, 5, 7
#
# Sweep B — Reward-type ablation (GOS+continuous vs GOS+binary)
#   GOS+continuous runs are shared with Sweep A; only GOS+binary is new here.
#
# Set DRY_RUN=1 to just print the commands.
# Set SUBMIT=sbatch to submit via SLURM (default is direct bash invocation).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
SUBMIT="${SUBMIT:-bash}"                # or "sbatch"
DRY_RUN="${DRY_RUN:-0}"

WS_DIFFS=(4 6 8 10 12)
KK_DIFFS=(3 5 7)

run() {
    local task=$1
    local difficulty=$2
    local estimator=$3
    local reward_mode=$4
    local seed="${5:-42}"
    local script
    if [ "$SUBMIT" = "sbatch" ]; then
        script="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch_slurm.sh"
    else
        script="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh"
    fi
    local cmd="$SUBMIT $script $task $difficulty $estimator $reward_mode $seed"
    echo "[launch_all] $cmd"
    if [ "$DRY_RUN" = "1" ]; then
        return
    fi
    $cmd
}

echo "=========================================="
echo "SWEEP A: Estimator compare (GOS+cont, MaxRL+bin)"
echo "=========================================="
for d in "${WS_DIFFS[@]}"; do
    run word_sorting "$d" gos continuous
    run word_sorting "$d" maxrl binary
done
for d in "${KK_DIFFS[@]}"; do
    run knights_knaves "$d" gos continuous
    run knights_knaves "$d" maxrl binary
done

echo
echo "=========================================="
echo "SWEEP B: Reward ablation (additional GOS+binary runs)"
echo "=========================================="
for d in "${WS_DIFFS[@]}"; do
    run word_sorting "$d" gos binary
done
for d in "${KK_DIFFS[@]}"; do
    run knights_knaves "$d" gos binary
done
