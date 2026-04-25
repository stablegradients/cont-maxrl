#!/usr/bin/env bash
# Dispatcher for the 36-run WS + SR + GoL sweep on DeltaAI.
#
# 36 runs = 3 tasks x 3 difficulties x {gos,grpo} x {continuous,binary}
#
# Round-robins across 7 accounts (8 GPU cap each = 56 concurrent slots).
# Each run submits with chain_left=1 → current + 1 dependent = 2 chained jobs
# (94 wall-hours max, resume_mode=auto reuses the latest rolling checkpoint).
#
# Usage:
#   bash scripts/dispatch_sweep_deltaai.sh            # submit all 36 runs
#   DRY_RUN=1 bash scripts/dispatch_sweep_deltaai.sh  # print sbatch commands, no submit

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/projects/bgmo/sramasubramanian/cont-maxrl}"
SCRIPT="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch_slurm_deltaai.sh"
SLURM_LOG_DIR="${REPO_ROOT}/slurm_logs"
DISPATCH_LOG="${REPO_ROOT}/experiment10_reasoning_gym/logs/dispatch_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${SLURM_LOG_DIR}" "$(dirname ${DISPATCH_LOG})"

[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

SEED="${SEED:-42}"
CHAIN_LEFT="${CHAIN_LEFT:-1}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNTS=(bgmo-dtai-gh bgtv-dtai-gh bgtu-dtai-gh bgts-dtai-gh bgyb-dtai-gh bgyc-dtai-gh bgyd-dtai-gh)

# All 36 configs: "task diff estimator reward"
CONFIGS=(
    # word_sorting × {4,8,12} × {gos,grpo} × {continuous,binary}
    "word_sorting 4 gos continuous"
    "word_sorting 4 gos binary"
    "word_sorting 4 grpo continuous"
    "word_sorting 4 grpo binary"
    "word_sorting 8 gos continuous"
    "word_sorting 8 gos binary"
    "word_sorting 8 grpo continuous"
    "word_sorting 8 grpo binary"
    "word_sorting 12 gos continuous"
    "word_sorting 12 gos binary"
    "word_sorting 12 grpo continuous"
    "word_sorting 12 grpo binary"
    # sentence_reordering × {3,7,10} × {gos,grpo} × {continuous,binary}
    "sentence_reordering 3 gos continuous"
    "sentence_reordering 3 gos binary"
    "sentence_reordering 3 grpo continuous"
    "sentence_reordering 3 grpo binary"
    "sentence_reordering 7 gos continuous"
    "sentence_reordering 7 gos binary"
    "sentence_reordering 7 grpo continuous"
    "sentence_reordering 7 grpo binary"
    "sentence_reordering 10 gos continuous"
    "sentence_reordering 10 gos binary"
    "sentence_reordering 10 grpo continuous"
    "sentence_reordering 10 grpo binary"
    # game_of_life × {4,7,10} × {gos,grpo} × {continuous,binary}
    "game_of_life 4 gos continuous"
    "game_of_life 4 gos binary"
    "game_of_life 4 grpo continuous"
    "game_of_life 4 grpo binary"
    "game_of_life 7 gos continuous"
    "game_of_life 7 gos binary"
    "game_of_life 7 grpo continuous"
    "game_of_life 7 grpo binary"
    "game_of_life 10 gos continuous"
    "game_of_life 10 gos binary"
    "game_of_life 10 grpo continuous"
    "game_of_life 10 grpo binary"
)

echo "Dispatching ${#CONFIGS[@]} runs across ${#ACCOUNTS[@]} accounts (seed=${SEED}, chain_left=${CHAIN_LEFT})"
echo "Log: ${DISPATCH_LOG}"
echo "Dry run: ${DRY_RUN}"
echo

i=0
for spec in "${CONFIGS[@]}"; do
    set -- $spec; task=$1; diff=$2; est=$3; reward=$4
    acct="${ACCOUNTS[$((i % ${#ACCOUNTS[@]}))]}"
    tag="${task}_${est}_${reward}_diff${diff}_s${SEED}"   # matches W&B experiment name
    cmd="sbatch --account=${acct} --job-name=rgym_${tag} --parsable ${SCRIPT} ${task} ${diff} ${est} ${reward} ${SEED} ${CHAIN_LEFT}"
    if [ "${DRY_RUN}" = "1" ]; then
        printf "  [%2d/%d] %-60s acct=%s\n" "$((i+1))" "${#CONFIGS[@]}" "$tag" "$acct"
        echo "         $cmd"
    else
        jobid=$(eval $cmd 2>&1)
        if [[ "$jobid" =~ ^[0-9]+$ ]]; then
            printf "  [%2d/%d] OK   job=%s  %-60s acct=%s\n" "$((i+1))" "${#CONFIGS[@]}" "$jobid" "$tag" "$acct" | tee -a "${DISPATCH_LOG}"
        else
            printf "  [%2d/%d] FAIL          %-60s acct=%s  err=%s\n" "$((i+1))" "${#CONFIGS[@]}" "$tag" "$acct" "$jobid" | tee -a "${DISPATCH_LOG}"
        fi
    fi
    i=$((i+1))
done

echo
echo "=== done: ${i} dispatched ==="
if [ "${DRY_RUN}" != "1" ]; then
    echo "Check queue:  squeue -u \$USER --name=rgym_* -o '%.10i %.20j %.10a %.2t %.10M %.20R'"
fi
