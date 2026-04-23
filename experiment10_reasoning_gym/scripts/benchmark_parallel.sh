#!/usr/bin/env bash
# Benchmark: run 4 copies of WS@4 GOS+continuous for 10 steps in parallel,
# each on 2 GPUs (CUDA_VISIBLE_DEVICES sharding). Measures effective
# parallelism vs the 8-GPU baseline.
#
# Usage:  bash scripts/benchmark_parallel.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
LAUNCH="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh"

mkdir -p "${REPO_ROOT}/logs/bench_parallel"

run_one() {
    local tag=$1
    local gpus=$2
    local portbase=$3
    export CUDA_VISIBLE_DEVICES=$gpus
    # Unique tmpdirs and ray ports so 4 Ray clusters can coexist on the node
    export RAY_TMPDIR="/tmp/ray_bench_${tag}"
    export TMPDIR="/tmp/tmp_bench_${tag}"
    mkdir -p "$RAY_TMPDIR" "$TMPDIR"
    export WANDB_MODE=offline
    bash "$LAUNCH" word_sorting 4 gos continuous 42 \
        trainer.total_training_steps=10 \
        trainer.total_epochs=1 \
        trainer.val_before_train=False \
        trainer.test_freq=100 \
        trainer.save_freq=-1 \
        trainer.n_gpus_per_node=2 \
        trainer.project_name=bench \
        trainer.experiment_name=bench_${tag} \
        > "${REPO_ROOT}/logs/bench_parallel/${tag}.log" 2>&1 &
}

echo "=== Launching 4 parallel runs, 2 GPUs each ==="
T_START=$(date +%s)

run_one gpu01 "0,1" 6391
run_one gpu23 "2,3" 6392
run_one gpu45 "4,5" 6393
run_one gpu67 "6,7" 6394

wait

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
echo "=== All runs done in ${ELAPSED}s ==="

echo
echo "=== per-run step times ==="
for tag in gpu01 gpu23 gpu45 gpu67; do
    LOG="${REPO_ROOT}/logs/bench_parallel/${tag}.log"
    LAST_STEP=$(grep "perf/time_per_step" "$LOG" 2>/dev/null | tail -1 | grep -oE "time_per_step:[0-9.]+" || echo "NONE")
    MFU=$(grep "perf/mfu/actor" "$LOG" 2>/dev/null | tail -1 | grep -oE "mfu/actor:[0-9.]+" || echo "NONE")
    COMPLETED=$(grep -E "Training Progress: 10/10|Training Progress: 100%" "$LOG" 2>/dev/null | tail -1 || echo "NOT FOUND")
    echo "  ${tag}: ${LAST_STEP}  ${MFU}  [${COMPLETED}]"
done
