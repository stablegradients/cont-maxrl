#!/usr/bin/env bash
#SBATCH --job-name=pie_fb_gem5
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:50:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# ============================================================================
# FULL-BATCH PIE gem5-speedup RL — single self-managing launcher.
#
#   Qwen3-1.7B | bysrc train set (16,452 src programs) | reward = gem5 SPEEDUP
#   ratio (src_ticks / rollout_ticks) on K=1 RANDOM group-sampled case;
#   correctness gated on ALL usable cases; rollout zeroed if >3x src ticks.
#   B=64  G=16  -> 1024 rollouts/step ; lr=1e-6 ; pure on-policy PG (no KL/
#   entropy/ref, clip inert via old_log_prob recompute) ; adv_estimator = arg.
#   ~257 steps/epoch x 10 = 2570 steps. NO in-training eval (eval is post-hoc).
#
# Self-management (no babysitting daemon needed):
#   * ELECTION  — spray this script to N accounts; the FIRST to start running
#                 claims a durable lock on bgtv and becomes the sole survivor;
#                 every other copy self-cancels on startup. Stale (dead-owner)
#                 locks are auto-reclaimed, so resume jobs re-take ownership.
#   * REAP      — the survivor scancels the rest of its spray cohort (same job
#                 name) so no still-pending sibling can later revive the run.
#   * RESUME    — the survivor submits ONE afterany:self resume child UPFRONT
#                 (so it survives a walltime kill), gated on work-remaining and
#                 a segment budget. resume_mode=auto picks up the rolling
#                 full-state ckpt. The tail child that sees "done" exits at once.
#
#   Usage:  sbatch --account=<acct>-dtai-gh --job-name=pie_fb_<adv>_s<seed> \
#                  run_fullbatch_gem5.sh <grpo|gos|rloo> [SEED] [SEGMENTS_LEFT]
#   Normally invoked via spray_fullbatch.sh (sets --job-name/--account/cohort).
# ============================================================================

ADV="${1:?usage: run_fullbatch_gem5.sh <grpo|gos|rloo> [SEED] [SEGMENTS_LEFT]}"
SEED="${2:-0}"
SEGMENTS_LEFT="${3:-8}"          # self-resubmit budget (8 x 48h = 384h headroom)
case "${ADV}" in grpo|gos|rloo) ;; *) echo "FATAL: adv must be grpo|gos|rloo (got ${ADV})"; exit 2;; esac

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/work/nvme/bgtv/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
SELF="${REPO_ROOT}/experiment11_code_optim/scripts/fullbatch_gem5/run_fullbatch_gem5.sh"

# Model is selected via env so this ONE launcher serves both 1.7B and 4B. MODEL_TAG
# MUST be resolved BEFORE RUN_TAG (RUN_TAG derives OUTPUT_DIR / lock / carveout /
# PIE_RUN_ID). With the old hardcoded '1.7b' tag a 4B run would resume INTO and
# overwrite the existing 1.7B run. Defaults reproduce the original 1.7B tag, so a
# 1.7B resume (no PIE_MODEL_* set) is bit-for-bit unaffected.
MODEL_PATH="${PIE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
MODEL_TAG="${PIE_MODEL_TAG:-qwen3_1.7b}"

RUN_TAG="${MODEL_TAG}_pie_gem5_fb_${ADV}_g16_bs64_s${SEED}"
OUTPUT_DIR="${STORAGE_ROOT}/results/${RUN_TAG}"
ACTOR_CKPT_DIR="${OUTPUT_DIR}/actor_ckpts"
LOCKROOT="${STORAGE_ROOT}/locks"
LOCKDIR="${LOCKROOT}/${RUN_TAG}.lock"
JOBNAME="${SLURM_JOB_NAME:-pie_fb_${MODEL_TAG}_${ADV}_s${SEED}}"
PIE_TOTAL_STEPS="${PIE_TOTAL_STEPS:-2570}"   # hard step cap == "done" threshold (1.7B: ==10.0 epochs)
mkdir -p "${OUTPUT_DIR}" "${ACTOR_CKPT_DIR}" "${LOCKROOT}" "${REPO_ROOT}/slurm_logs"

# ---------------------------------------------------------------------------
# ELECTION: claim the durable lock or self-cancel. First-to-run wins; a lock
# whose owner job is no longer in squeue is treated as stale and reclaimed
# (so an afterany resume child re-takes ownership from its dead parent).
# ---------------------------------------------------------------------------
owner_alive() { [ -n "$1" ] && squeue -j "$1" -h -o '%i' 2>/dev/null | grep -qw "$1"; }
claim_lock() {
  local tries=0 empties=0
  while :; do
    if mkdir "${LOCKDIR}" 2>/dev/null; then echo "${SLURM_JOB_ID}" > "${LOCKDIR}/owner"; return 0; fi
    local own; own="$(cat "${LOCKDIR}/owner" 2>/dev/null || true)"
    [ "${own}" = "${SLURM_JOB_ID}" ] && return 0
    if [ -z "${own}" ]; then
      # Lock dir exists but owner not yet written — the winner is mid-claim
      # (microsecond window between its mkdir and `echo > owner`). WAIT and
      # re-read; do NOT delete, or two jobs could both end up "winning". Only
      # if owner stays empty ~20s (winner died between mkdir and echo) reclaim.
      empties=$((empties+1)); [ ${empties} -gt 20 ] && { rm -rf "${LOCKDIR}" 2>/dev/null || true; empties=0; }
      sleep 1; continue
    fi
    if owner_alive "${own}"; then return 1; fi          # a live owner holds it -> we lose
    rm -rf "${LOCKDIR}" 2>/dev/null || true             # non-empty owner, dead -> reclaim
    tries=$((tries+1)); [ ${tries} -gt 30 ] && return 1
    sleep 1
  done
}
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if claim_lock; then
    echo "[election] WON ${RUN_TAG} as job ${SLURM_JOB_ID} on $(hostname)"
    # Release the lock on ANY exit (clean, crash, or SIGTERM/walltime) so the
    # resume child can re-claim without waiting for stale-detection.
    trap 'rm -rf "${LOCKDIR}" 2>/dev/null || true' EXIT
    trap 'rm -rf "${LOCKDIR}" 2>/dev/null || true; exit 143' TERM
    # REAP the rest of the spray cohort (same job name, not me).
    for j in $(squeue -u "${USER}" -h -n "${JOBNAME}" -o '%i' 2>/dev/null | grep -vw "${SLURM_JOB_ID}"); do
      echo "[reap] scancel sibling ${j}"; scancel "${j}" 2>/dev/null || true
    done
  else
    echo "[election] live owner $(cat "${LOCKDIR}/owner" 2>/dev/null) holds ${RUN_TAG}; self-cancelling ${SLURM_JOB_ID}"
    scancel "${SLURM_JOB_ID}" 2>/dev/null || true
    exit 0
  fi
fi

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -euo pipefail
export PYTHONNOUSERSITE=1
if [ -f "${REPO_ROOT}/.env" ]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi

# ---------------------------------------------------------------------------
# Data: bysrc train parquet (gem5 ground_truth: problem_id/src_id/split/
# usable_case_ids/gem5_src_per_tc_ticks). Val parquet only kept for config
# validity; in-training eval is DISABLED (eval_steps=[]).
# ---------------------------------------------------------------------------
PIE_PARQUET_ROOT="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet"
PIE_TRAIN_PARQUET="${PIE_TRAIN_PARQUET:-${PIE_PARQUET_ROOT}/pie_gem5_train.parquet}"
PIE_VAL_PARQUET="${PIE_VAL_PARQUET:-${PIE_PARQUET_ROOT}/pie_gem5_val_200.parquet}"
export PIE_TEST_CASE_DIR="${PIE_TEST_CASE_DIR:-/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases}"
[ -f "${PIE_TRAIN_PARQUET}" ] || { echo "FATAL: missing ${PIE_TRAIN_PARQUET}"; exit 2; }
# MODEL_PATH / MODEL_TAG resolved near the top (before RUN_TAG); do not redefine.

# Keep-all actor (hf_model) snapshots every 100 steps; rolling full-state every
# save_freq for cheap resume. carveout dir survives the rolling-1 deletion.
export PIE_ACTOR_CARVEOUT_DIR="${ACTOR_CKPT_DIR}"
export PIE_CARVEOUT_EVERY="${PIE_CARVEOUT_EVERY:-100}"   # permanent hf snapshot cadence (4B run uses 250)
# Rolling full-state cadence (1 kept, for resume). At ~7min/step a crash/walltime
# cut loses <= SAVE_FREQ steps; 20 -> <=~2.3h lost, save I/O ~12s is negligible.
# MUST divide PIE_CARVEOUT_EVERY (100) so carve-out lands on a save step.
SAVE_FREQ="${PIE_SAVE_FREQ:-20}"
KEEP_CKPT="${PIE_KEEP_CKPT:-1}"   # rolling full-state ckpts kept; 2 => a prior ckpt always survives a torn save (resume safety)
# carve-out must land on a save step, else the permanent snapshot is never taken.
if [ $(( PIE_CARVEOUT_EVERY % SAVE_FREQ )) -ne 0 ]; then
  echo "FATAL: PIE_CARVEOUT_EVERY (${PIE_CARVEOUT_EVERY}) must be a multiple of SAVE_FREQ (${SAVE_FREQ})"; exit 2
fi
# Rollout/actor shapes (env-overridable; defaults = the 1.7B run). The 4B run sets
# longer responses (16384), a matching max_model_len, and micro_batch=1 for memory
# headroom on the 4x-longer responses. max_model_len MUST be >= max_prompt(2560)+max_resp.
MAX_RESP="${PIE_MAX_RESP:-4096}"
MAX_MODEL_LEN="${PIE_MAX_MODEL_LEN:-6656}"
MICRO_BS="${PIE_MICRO_BS:-2}"
GPU_MEM_UTIL="${PIE_GPU_MEM_UTIL:-0.5}"
# remove_padding=True processes ACTUAL tokens (not the 18944 pad) -> ~2.5x faster
# actor + far less GPU mem; essential at 16k (stress: rmpad=False spills over HBM).
# Default False = the 1.7B run's behavior (backward-compat); the 4B run sets True.
REMOVE_PADDING="${PIE_REMOVE_PADDING:-False}"
if [ "${MAX_MODEL_LEN}" -lt $(( 2560 + MAX_RESP )) ]; then
  echo "FATAL: PIE_MAX_MODEL_LEN (${MAX_MODEL_LEN}) < max_prompt(2560)+max_resp(${MAX_RESP})"; exit 2
fi

export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET || true; unset NCCL_NET_PLUGIN || true; export NCCL_PLUGIN_P2P=0

: "${WANDB_API_KEY:?WANDB_API_KEY must be set in submission env (never hardcode)}"
export WANDB_ENTITY="${WANDB_ENTITY:-stablegradients}"
export WANDB_RUN_GROUP="${MODEL_TAG}_pie-gem5-fullbatch_${ADV}_G16_BS64"
EXPERIMENT_NAME="${WANDB_RUN_GROUP}_s${SEED}"

export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"

# ---- gem5 reward engine (reads ENV ONLY — verl's batch reward path passes no
#      reward_kwargs). K=1 RANDOM group-sampled case == the full-batch spec. ----
export PIE_MODE=1
export PIE_REWARD_IMPL=gem5
export PIE_GEM5_REWARD_KIND=ratio      # r = src_ticks / rollout_ticks (same case)
export PIE_GEM5_REWARD_K=1             # ONE gem5-timed case per prompt/step
export PIE_GEM5_CASE_SELECT="${PIE_GEM5_CASE_SELECT:-sample}"   # 'sample'=random usable case (1.7B); 4B run uses 'biggest' (heaviest, matches eval)
export PIE_GEM5_REWARD_SAMPLE=group    # same case for all 16 rollouts of a prompt (irrelevant under case_select=biggest, which is deterministic)
export PIE_GEM5_REWARD_SEED=${SEED}
export PIE_GEM5_BUDGET_MULT=3.0        # zero any rollout beyond 3x src ticks
export PIE_GEM5_REWARD_WORKERS="${PIE_GEM5_REWARD_WORKERS:-256}"   # parallel native+gem5 across the 288-core node
export PIE_NAT_TIMEOUT="${PIE_NAT_TIMEOUT:-10}"    # per-CASE native correctness timeout (s); not the speed judge. measured slowest src case ~0.32s
export PIE_GEM5_REWARD_NATIVE_BUDGET="${PIE_GEM5_REWARD_NATIVE_BUDGET:-120}"   # per-ROLLOUT cumulative native budget (s); measured slowest src total ~3.5s
export PIE_GEM5_REWARD_DEADLINE="${PIE_GEM5_REWARD_DEADLINE:-3000}"  # hard backstop per reward batch (s)
export TMPDIR="/tmp/${USER}_fbg5_${SLURM_JOB_ID:-$$}"
export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"; mkdir -p "${PIE_COMPILE_SCRATCH}"

export INSITU_EVAL_K_VALUES='[1,4,16,64]'
export INSITU_EVAL_BEST_AT_K=1
export PIE_RUN_ID="${RUN_TAG}"
export PIE_ALGO="${ADV}"
export PIE_INIT_CKPT="${MODEL_PATH}"
export VLLM_USE_V1=0
export RAY_OBJECT_STORE_MEMORY=32000000000
export RAY_TMPDIR="/tmp/ray_${USER}_${SLURM_JOB_ID:-$$}"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
STEPS_DONE="$(cat "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
echo "=== FULL-BATCH gem5 RL | ${MODEL_TAG} | ${ADV} | G=16 B=64 | lr=1e-6 | seed=${SEED} ==="
echo "    reward=ratio K=1 case_select=${PIE_GEM5_CASE_SELECT} budget=3x workers=${PIE_GEM5_REWARD_WORKERS} nat_to=${PIE_NAT_TIMEOUT}s nat_budget=${PIE_GEM5_REWARD_NATIVE_BUDGET}s"
echo "    model=${MODEL_PATH} max_resp=${MAX_RESP} max_model_len=${MAX_MODEL_LEN} micro_bs=${MICRO_BS} gpu_util=${GPU_MEM_UTIL}"
echo "    OUTPUT_DIR=${OUTPUT_DIR}"
echo "    steps_done=${STEPS_DONE}/${PIE_TOTAL_STEPS}  save_freq=${SAVE_FREQ} keep_ckpt=${KEEP_CKPT} carveout_every=${PIE_CARVEOUT_EVERY}  segments_left=${SEGMENTS_LEFT}  GPUs=${N_GPUS}"

# ---------------------------------------------------------------------------
# UPFRONT resume child (winner only): survives a walltime kill. Submitted only
# if work remains AND segment budget is left. The "done" check costs us at most
# one harmless tail job that starts, sees STEPS_DONE>=total, and exits.
# ---------------------------------------------------------------------------
NEXT_SEG=$((SEGMENTS_LEFT - 1))
if [ "${STEPS_DONE}" -ge "${PIE_TOTAL_STEPS}" ]; then
  echo "[chain] already at ${STEPS_DONE} >= ${PIE_TOTAL_STEPS} steps; no train, no resubmit"
  exit 0
elif [ "${NEXT_SEG}" -gt 0 ]; then
  CHILD=$(sbatch --parsable --account="${SLURM_JOB_ACCOUNT}" --job-name="${JOBNAME}" \
            --dependency=afterany:${SLURM_JOB_ID} "${SELF}" "${ADV}" "${SEED}" "${NEXT_SEG}" 2>/dev/null || true)
  echo "[chain] queued resume child ${CHILD:-<none>} (afterany:${SLURM_JOB_ID}, segments_left=${NEXT_SEG})"
else
  echo "[chain] segment budget exhausted; this is the last segment"
fi

set +e
$PYTHON "${REPO_ROOT}/experiment11_code_optim/train.py" \
  +data.seed=${SEED} \
  algorithm.adv_estimator=${ADV} \
  algorithm.filter_overlong_responses=False \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${PIE_TRAIN_PARQUET}" \
  data.val_files="${PIE_VAL_PARQUET}" \
  data.train_batch_size=64 \
  data.max_prompt_length=2560 \
  data.max_response_length=${MAX_RESP} \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING} \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS} \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=100 \
  actor_rollout_ref.actor.clip_ratio_high=100 \
  actor_rollout_ref.actor.grad_clip=2.0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS} \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment11_code_optim/reward/gem5_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=${PIE_TOTAL_STEPS} \
  trainer.total_epochs=11 \
  trainer.project_name=cont-maxrl-pie \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "trainer.logger=['console','wandb']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  "+actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=${KEEP_CKPT} \
  trainer.val_before_train=False \
  "insitu_eval.eval_steps=[]" \
  insitu_eval.data_file="${PIE_VAL_PARQUET}" \
  insitu_eval.output_dir="${OUTPUT_DIR}/insitu_eval" \
  insitu_eval.label="disabled" \
  ray_init.num_cpus=280

TRAIN_EXIT=$?
set -e
echo "=== segment done: ${ADV} seed=${SEED} exit=${TRAIN_EXIT} steps_now=$(cat "${OUTPUT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?')/${PIE_TOTAL_STEPS} ==="
exit ${TRAIN_EXIT}
