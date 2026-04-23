#!/usr/bin/env bash
# Benchmark: 8 copies of WS@4 GOS+continuous in parallel, 1 GPU each.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
LAUNCH="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh"
mkdir -p "${REPO_ROOT}/logs/bench_1gpu8x"

run_one() {
    local tag=$1
    local gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu \
    RAY_TMPDIR="/tmp/ray_b1_${tag}" \
    TMPDIR="/tmp/tmp_b1_${tag}" \
    WANDB_MODE=offline \
    bash "$LAUNCH" word_sorting 4 gos continuous 42 \
        trainer.total_training_steps=10 \
        trainer.total_epochs=1 \
        trainer.val_before_train=False \
        trainer.test_freq=100 \
        trainer.save_freq=-1 \
        trainer.n_gpus_per_node=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.project_name=bench1 \
        trainer.experiment_name=b1_${tag} \
        > "${REPO_ROOT}/logs/bench_1gpu8x/${tag}.log" 2>&1 &
}

echo "=== Launching 8 parallel runs, 1 GPU each ==="
T_START=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
    mkdir -p "/tmp/ray_b1_gpu${i}" "/tmp/tmp_b1_gpu${i}"
    run_one gpu${i} ${i}
done
wait
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
echo "=== 8 runs done in ${ELAPSED}s ==="
echo
for i in 0 1 2 3 4 5 6 7; do
    LOG="${REPO_ROOT}/logs/bench_1gpu8x/gpu${i}.log"
    step=$(grep "perf/time_per_step" "$LOG" 2>/dev/null | tail -1 | grep -oE "time_per_step:[0-9.]+")
    mfu=$(grep "perf/mfu/actor" "$LOG" 2>/dev/null | tail -1 | grep -oE "mfu/actor:[0-9.]+")
    last=$(grep -oE "Training Progress: *[0-9]+/10" "$LOG" 2>/dev/null | tail -1)
    echo "  gpu${i}: ${step}  ${mfu}  [${last}]"
done
