#!/usr/bin/env bash
# Benchmark: 16 copies of WS@4 GOS+continuous, 2 runs pinned to each GPU.
# Uses NVIDIA MPS so both processes can share SMs concurrently; without MPS
# CUDA time-slices which usually zeroes out the gain.
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
LAUNCH="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh"
mkdir -p "${REPO_ROOT}/logs/bench_2per_gpu"

# Start MPS daemon
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo "=== Starting MPS control daemon ==="
nvidia-cuda-mps-control -d 2>&1 || echo "MPS may already be running"
sleep 2

run_one() {
    local tag=$1
    local gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu \
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log \
    RAY_TMPDIR="/tmp/ray_b2_${tag}" \
    TMPDIR="/tmp/tmp_b2_${tag}" \
    WANDB_MODE=offline \
    bash "$LAUNCH" word_sorting 4 gos continuous 42 \
        trainer.total_training_steps=10 \
        trainer.total_epochs=1 \
        trainer.val_before_train=False \
        trainer.test_freq=100 \
        trainer.save_freq=-1 \
        trainer.n_gpus_per_node=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        trainer.project_name=bench2 \
        trainer.experiment_name=b2_${tag} \
        > "${REPO_ROOT}/logs/bench_2per_gpu/${tag}.log" 2>&1 &
}

echo "=== Launching 16 parallel runs, 2 per GPU (MPS-shared) ==="
T_START=$(date +%s)
for i in 0 1 2 3 4 5 6 7; do
    mkdir -p "/tmp/ray_b2_gpu${i}a" "/tmp/tmp_b2_gpu${i}a" "/tmp/ray_b2_gpu${i}b" "/tmp/tmp_b2_gpu${i}b"
    run_one gpu${i}a ${i}
    sleep 1  # stagger slightly to avoid race on Ray port allocation
    run_one gpu${i}b ${i}
done
wait
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
echo "=== 16 runs done in ${ELAPSED}s ==="

echo "=== Stopping MPS daemon ==="
echo quit | nvidia-cuda-mps-control 2>&1 || true

echo
for i in 0 1 2 3 4 5 6 7; do
    for s in a b; do
        LOG="${REPO_ROOT}/logs/bench_2per_gpu/gpu${i}${s}.log"
        step=$(grep "perf/time_per_step" "$LOG" 2>/dev/null | tail -1 | grep -oE "time_per_step:[0-9.]+")
        mfu=$(grep "perf/mfu/actor" "$LOG" 2>/dev/null | tail -1 | grep -oE "mfu/actor:[0-9.]+")
        last=$(grep -oE "Training Progress: *[0-9]+/10" "$LOG" 2>/dev/null | tail -1)
        printf "  gpu%s%s: %-22s %-20s [%s]\n" "$i" "$s" "$step" "$mfu" "$last"
    done
done
