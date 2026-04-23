#!/usr/bin/env bash
# Launch a single Reasoning Gym training run.
#
# Usage:
#   bash launch.sh <task> <difficulty> <estimator> <reward_mode> [seed]
#
#   <task>         word_sorting | knights_knaves | sentence_reordering | game_of_life
#   <difficulty>   task-specific integer (num_words / n_people / grid_size)
#   <estimator>    gos | maxrl | rloo | grpo   (mapped to verl names below)
#   <reward_mode>  continuous | binary
#   [seed]         optional seed (default 42)
#
# Spec name -> verl adv_estimator mapping (see experiment10 README):
#   gos       -> maxrl_unnorm   (ordinal cross-entropy, our estimator)
#   maxrl     -> binary_maxrl   (Tajwar et al. binary MaxRL)
#   rloo      -> rloo           (leave-one-out baseline)
#   grpo      -> grpo           (Z-score normalization baseline)

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <task> <difficulty> <estimator> <reward_mode> [seed]" >&2
    exit 1
fi

TASK=$1
DIFFICULTY=$2
ESTIMATOR_SHORT=$3
REWARD_MODE=$4
SEED="${5:-42}"
# Consume positional args so "$@" only forwards extra hydra overrides.
shift 4
if [ "$#" -gt 0 ] && [[ "$1" != *"="* ]]; then
    # SEED was provided as the 5th positional — shift it too.
    shift
fi

# --- Map short estimator names to verl's registered names ---
case "$ESTIMATOR_SHORT" in
    gos|maxrl_unnorm)    ADV_ESTIMATOR=maxrl_unnorm ;;
    maxrl|binary_maxrl)  ADV_ESTIMATOR=binary_maxrl ;;
    rloo)                ADV_ESTIMATOR=rloo ;;
    grpo)                ADV_ESTIMATOR=grpo ;;
    *)
        echo "ERROR: unknown estimator '$ESTIMATOR_SHORT'." >&2
        echo "  Valid: gos | maxrl | rloo | grpo" >&2
        exit 1 ;;
esac

# --- Reward mode -> compute_score function name in reward_manager.py ---
case "$REWARD_MODE" in
    continuous) REWARD_FN_NAME=compute_score_continuous ;;
    binary)     REWARD_FN_NAME=compute_score_binary ;;
    *)
        echo "ERROR: unknown reward_mode '$REWARD_MODE'. Use continuous or binary." >&2
        exit 1 ;;
esac

# --- Per-task response/prompt lengths (spec §6.2) ---
case "$TASK" in
    knights_knaves)
        MAX_PROMPT_LENGTH=1024
        MAX_RESPONSE_LENGTH=2048
        ;;
    word_sorting|sentence_reordering)
        MAX_PROMPT_LENGTH=512
        MAX_RESPONSE_LENGTH=512
        ;;
    game_of_life)
        MAX_PROMPT_LENGTH=1024
        MAX_RESPONSE_LENGTH=1024
        ;;
    *)
        echo "ERROR: unknown task '$TASK'." >&2
        exit 1 ;;
esac

# --- Paths ---
REPO_ROOT="${REPO_ROOT:-/root/shrinivas/cont-maxrl}"
DATA_DIR="${REPO_ROOT}/experiment10_reasoning_gym/data/${TASK}_diff${DIFFICULTY}"
TRAIN_DATA="${DATA_DIR}/train.parquet"

# Cross-difficulty validation: VAL_DIFFS is a comma-separated list of diffs
# (e.g. "4,8,12"). Defaults to just the training diff. Each resolves to
# experiment10_reasoning_gym/data/{task}_diff{N}/test.parquet.
VAL_DIFFS="${VAL_DIFFS:-${DIFFICULTY}}"
VAL_PATHS=()
for d in $(echo "$VAL_DIFFS" | tr ',' ' '); do
    VAL_PATHS+=("${REPO_ROOT}/experiment10_reasoning_gym/data/${TASK}_diff${d}/test.parquet")
done
# Hydra list syntax: data.val_files=[path1,path2,path3]
VAL_FILES_LIST=$(IFS=,; echo "[${VAL_PATHS[*]}]")

if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: missing train parquet: $TRAIN_DATA" >&2
    exit 1
fi
for p in "${VAL_PATHS[@]}"; do
    if [ ! -f "$p" ]; then
        echo "ERROR: missing val parquet: $p" >&2
        exit 1
    fi
done

REWARD_FN_PATH="${REPO_ROOT}/experiment10_reasoning_gym/src/reward_manager.py"

PROJECT_NAME="${PROJECT_NAME:-reasoning_gym_${TASK}}"
# Include task in experiment name so WS/SR runs don't collide when sharing a project.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${TASK}_${ESTIMATOR_SHORT}_${REWARD_MODE}_diff${DIFFICULTY}_s${SEED}}"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"

# --- Env ---
# Load project-wide secrets (WANDB_API_KEY, HF_TOKEN, ...) if present.
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export VERL_REWARD_WORKERS="${VERL_REWARD_WORKERS:-64}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
# verl calls wandb.init(...) without an `entity` kwarg; respect WANDB_ENTITY
# from .env (or caller) instead of defaulting to the logged-in user's workspace.
export WANDB_ENTITY="${WANDB_ENTITY:-stablegradients}"
export WANDB_RUN_GROUP="${PROJECT_NAME}"
export WANDB_MODE="${WANDB_MODE:-online}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set — wandb will prompt or fail." >&2
fi

N_GPUS="${N_GPUS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
SAVE_FREQ="${SAVE_FREQ:-780}"
MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
TEST_FREQ="${TEST_FREQ:-25}"

PYTHON="${PYTHON:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-${REPO_ROOT}/experiment4_llm/src/train.py}"

echo "=== Reasoning Gym Train ==="
echo "  task=${TASK} difficulty=${DIFFICULTY}"
echo "  adv_estimator=${ADV_ESTIMATOR} (short=${ESTIMATOR_SHORT})"
echo "  reward_mode=${REWARD_MODE} (fn=${REWARD_FN_NAME})"
echo "  seed=${SEED}"
echo "  prompt/response len = ${MAX_PROMPT_LENGTH} / ${MAX_RESPONSE_LENGTH}"
echo "  W&B project=${PROJECT_NAME} experiment=${EXPERIMENT_NAME}"
echo "  checkpoints -> ${OUTPUT_DIR}"
echo

exec $PYTHON -W ignore "${TRAIN_ENTRY}" \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_FILES_LIST}" \
    data.train_batch_size=256 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=${REWARD_FN_NAME} \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    "trainer.logger=[console,wandb]" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT_KEEP} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    "$@"
