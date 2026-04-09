#!/bin/bash
# Queue-based launcher: 2 concurrent 4-GPU DDP training runs on 8 GPUs.
# Usage: bash experiment7_depth/scripts/submit_all.sh [runs_dir]
set -euo pipefail

SCRIPT_DIR="${1:-$(dirname "$(realpath "$0")")/runs}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=${NUM_GPUS:-8}
GPUS_PER_RUN=${GPUS_PER_RUN:-4}
NUM_SLOTS=$((NUM_GPUS / GPUS_PER_RUN))

# Slot GPU assignments
declare -A SLOT_GPUS
SLOT_GPUS[0]="0,1,2,3"
SLOT_GPUS[1]="4,5,6,7"

JOB_FILES=("$SCRIPT_DIR"/zx7k_*.sh)
TOTAL_JOBS=${#JOB_FILES[@]}
echo "[$(date +%H:%M:%S)] Total jobs: $TOTAL_JOBS, $NUM_SLOTS slots x $GPUS_PER_RUN GPUs each"

declare -A SLOT_PIDS
for slot in $(seq 0 $((NUM_SLOTS - 1))); do
    SLOT_PIDS[$slot]=""
done

completed=0
next_job=0

while [[ $next_job -lt $TOTAL_JOBS ]] || [[ -n "${SLOT_PIDS[0]:-}" ]] || [[ -n "${SLOT_PIDS[1]:-}" ]]; do
    for slot in $(seq 0 $((NUM_SLOTS - 1))); do
        pid=${SLOT_PIDS[$slot]:-}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            # Count completion if a job just finished
            if [[ -n "$pid" ]]; then
                wait "$pid" 2>/dev/null || true
                completed=$((completed + 1))
                echo "[$(date +%H:%M:%S)] Completed $completed/$TOTAL_JOBS"
            fi
            SLOT_PIDS[$slot]=""
            if [[ $next_job -lt $TOTAL_JOBS ]]; then
                job="${JOB_FILES[$next_job]}"
                jobname=$(basename "$job" .sh)
                gpus=${SLOT_GPUS[$slot]}

                echo "[$(date +%H:%M:%S)] [slot $slot, GPUs $gpus] Launching $jobname ($((next_job + 1))/$TOTAL_JOBS)"
                CUDA_VISIBLE_DEVICES=$gpus bash "$job" > "$LOG_DIR/$jobname.log" 2>&1 &
                SLOT_PIDS[$slot]=$!
                next_job=$((next_job + 1))
            fi
        fi
    done
    sleep 5
done

wait
echo "[$(date +%H:%M:%S)] All $TOTAL_JOBS jobs complete."
