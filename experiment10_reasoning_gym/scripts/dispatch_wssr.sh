#!/usr/bin/env bash
# Dispatcher for the WS+SR sweep.
#
# 24 runs total:
#   word_sorting        diffs {4, 8, 12}
#   sentence_reordering diffs {3, 7, 10}
#   each diff x 4 combos (GOS/GRPO x binary/continuous)
#
# 8 runs concurrent (one per GPU), 3 waves in sequence.
# 20 epochs per run, full 512 response cap, single seed 42.
# Each trained checkpoint validates on all 3 diffs of its task.

set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
LAUNCH="${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh"
LOG_DIR="${REPO_ROOT}/logs/prod"
mkdir -p "$LOG_DIR"

WS_VAL="4,8,12"
SR_VAL="3,7,10"
SEED=42

launch_run() {
    local gpu=$1 task=$2 diff=$3 est=$4 reward=$5 val_diffs=$6
    local tag="${task}_${est}_${reward}_diff${diff}_s${SEED}"
    local log="$LOG_DIR/${tag}.log"
    echo "  [gpu=${gpu}] ${tag} -> ${log}"
    CUDA_VISIBLE_DEVICES=$gpu \
    VAL_DIFFS=$val_diffs \
    RAY_TMPDIR=/tmp/ray_prod_${tag} \
    TMPDIR=/tmp/tmp_prod_${tag} \
    PROJECT_NAME=Reasoning-GYM-GOS \
    TOTAL_EPOCHS=20 \
    SAVE_FREQ=780 \
    MAX_CKPT_KEEP=2 \
    TEST_FREQ=78 \
    N_GPUS=1 \
    bash "$LAUNCH" "$task" "$diff" "$est" "$reward" "$SEED" \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        > "$log" 2>&1 &
    mkdir -p "/tmp/ray_prod_${tag}" "/tmp/tmp_prod_${tag}"
}

run_wave() {
    local wave_name=$1; shift
    local specs=("$@")
    local t_start=$(date +%s)
    echo "=== $wave_name (${#specs[@]} runs) ==="
    local gpu=0
    for spec in "${specs[@]}"; do
        # spec fields: task diff est reward val_diffs
        launch_run "$gpu" $spec
        gpu=$((gpu + 1))
    done
    wait
    local t_end=$(date +%s)
    echo "=== $wave_name done in $((t_end - t_start))s ==="
    echo
}

# Each wave is a list of 8 runs, one per GPU 0..7.
# Each spec: "task diff estimator reward val_diffs"

WAVE1=(
    "word_sorting 4 gos   continuous $WS_VAL"
    "word_sorting 4 gos   binary     $WS_VAL"
    "word_sorting 4 grpo  continuous $WS_VAL"
    "word_sorting 4 grpo  binary     $WS_VAL"
    "word_sorting 8 gos   continuous $WS_VAL"
    "word_sorting 8 gos   binary     $WS_VAL"
    "word_sorting 8 grpo  continuous $WS_VAL"
    "word_sorting 8 grpo  binary     $WS_VAL"
)

WAVE2=(
    "word_sorting 12 gos  continuous $WS_VAL"
    "word_sorting 12 gos  binary     $WS_VAL"
    "word_sorting 12 grpo continuous $WS_VAL"
    "word_sorting 12 grpo binary     $WS_VAL"
    "sentence_reordering 3 gos   continuous $SR_VAL"
    "sentence_reordering 3 gos   binary     $SR_VAL"
    "sentence_reordering 3 grpo  continuous $SR_VAL"
    "sentence_reordering 3 grpo  binary     $SR_VAL"
)

WAVE3=(
    "sentence_reordering 7  gos   continuous $SR_VAL"
    "sentence_reordering 7  gos   binary     $SR_VAL"
    "sentence_reordering 7  grpo  continuous $SR_VAL"
    "sentence_reordering 7  grpo  binary     $SR_VAL"
    "sentence_reordering 10 gos   continuous $SR_VAL"
    "sentence_reordering 10 gos   binary     $SR_VAL"
    "sentence_reordering 10 grpo  continuous $SR_VAL"
    "sentence_reordering 10 grpo  binary     $SR_VAL"
)

T_ALL=$(date +%s)
run_wave "WAVE 1" "${WAVE1[@]}"
run_wave "WAVE 2" "${WAVE2[@]}"
run_wave "WAVE 3" "${WAVE3[@]}"
T_END=$(date +%s)
echo "=== ALL WAVES DONE in $((T_END - T_ALL))s ==="
