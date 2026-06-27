#!/usr/bin/env bash
#SBATCH --job-name=q3_1.7b_pie_maxrl_g16
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-1.7B (Base) | PIE (instruction-count percentile) | MaxRL_unnorm (GOS) | G=16 | bgmo
# Spec v9 §3.3 estimator-only: sole varying knob vs sibling scripts is
#   algorithm.adv_estimator (+ run-name strings). All other hparams are
#   transcribed from experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_maxrl_g16.sh
#   and verified at build time (see PARITY_CONFIG.md).
# Self-chaining: resubmits with --dependency=afterany
# Usage: sbatch qwen3_1.7b_base_pie_maxrl_g16.sh [SEED] [CHAIN_LEFT]
# ============================================================

SEED=${1:-0}
CHAIN_LEFT=${2:-4}

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/work/nvme/bgtu/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# ============================================================
# PIE-specific paths (verified/staged at §14 step 4)
# ============================================================
PIE_DATA_ROOT="/work/nvme/bgmo/sramasubramanian/pie_data"
PIE_REFERENCE_TABLE_DIR="${PIE_REFERENCE_TABLE_DIR:-${PIE_DATA_ROOT}/reference_table}"
PIE_TEST_CASE_DIR="${PIE_TEST_CASE_DIR:-/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases}"
PIE_TRAIN_PARQUET="${PIE_TRAIN_PARQUET:-${PIE_DATA_ROOT}/parquet/pie_train.parquet}"
PIE_VAL_PARQUET="${PIE_VAL_PARQUET:-${PIE_DATA_ROOT}/parquet/pie_val_200.parquet}"
# §8.6 env-var fallback for insitu_eval (which does NOT pass reward_kwargs):
export PIE_REFERENCE_TABLE_DIR
export PIE_TEST_CASE_DIR

# ============================================================
# Epoch arithmetic (spec §3.1a): COMPUTED from N_pie_train, NOT transcribed.
# These MUST be set in submission env; do NOT default to DeepCoder's 83/1000/996.
# The §10 isolation guard re-asserts at training time:
#   total_training_steps == 10 * save_freq
#   save_freq == ceil(N_pie_train / 256)
# ============================================================
# Defaults computed for the §9-GO 1-epoch full-train SFT checkpoint:
#   N_pie_train = 73,893 (post-token-filter, see precompute/build_pie_parquet.py)
#   PIE_STEPS_PER_EPOCH = ceil(73893 / 256) = 289
#   PIE_TOTAL_STEPS     = 3 epochs * 289 = 867   (was 10*spe in spec §3.1a; reduced
#                                                 after §9 GO cleared cleanly)
#   PIE_EVAL_STEPS      = [k*spe for k in 1..3] = [289, 578, 867]
PIE_STEPS_PER_EPOCH="${PIE_STEPS_PER_EPOCH:-289}"
PIE_TOTAL_STEPS="${PIE_TOTAL_STEPS:-867}"
PIE_EVAL_STEPS="${PIE_EVAL_STEPS:-[289,578,867]}"

# ============================================================
# Model init (spec §6): SFT cold-start by default ("hq").
# If PIE_SFT_CKPT unset, falls back to base model with loud warning
# (spec sft.mode=none; expected to fail §9 GO/NO-GO).
# ============================================================
PIE_SFT_CKPT="${PIE_SFT_CKPT:-/work/nvme/bgtu/sramasubramanian/cont-maxrl/sft_ckpts/qwen3_1.7b_pie_sft_1ep_fulltrain_chat_s0/final}"
MODEL_PATH="${PIE_SFT_CKPT}"
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "FATAL: PIE_SFT_CKPT directory missing: ${MODEL_PATH}"
    echo "       Train SFT first via scripts/qwen3_1.7b_base_pie_sft.sh or override PIE_SFT_CKPT."
    exit 2
fi

OUTPUT_DIR="${STORAGE_ROOT}/results/qwen3_1.7b_pie_maxrl_unnorm_g16_s${SEED}"
ACTOR_CKPT_DIR="${OUTPUT_DIR}/actor_ckpts"
mkdir -p "${OUTPUT_DIR}" "${ACTOR_CKPT_DIR}"
# §8.3 actor carve-out target: train.py installs a hook that hard-links
# every verl save into this dir under step_<N>/. Zero space cost.
export PIE_ACTOR_CARVEOUT_DIR="${ACTOR_CKPT_DIR}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL, single-node)
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# ============================================================
# W&B (spec §3.5: env-only; never hardcode the key — see SECURITY_NOTE.md)
# §3.6: group excludes seed; experiment_name includes it.
# ============================================================
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in submission env (spec §3.5 / SECURITY_NOTE.md)}"
export WANDB_ENTITY="${WANDB_ENTITY:-stablegradients}"
export WANDB_RUN_GROUP="qwen3-1.7b-base_pie_maxrl_unnorm_G16"
EXPERIMENT_NAME="${WANDB_RUN_GROUP}_s${SEED}"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export VERL_REWARD_WORKERS=64

# F6: must precede the python launch so train.py's sys.modules shim fires
# and insitu_eval's hardcoded experiment4_llm.src.reward_function import
# routes to PIE's reward fn. Without it, insitu_eval silently scores
# every rollout as 0 (spec §8 silent-failure trap; DEFERRED_FIXES.md "F6").
export PIE_MODE=1

# Skip the ~3-min/step actor old_log_prob recompute by using vLLM rollout
# log_probs directly. Safe under on-policy ppo_epochs=1 (our setup).
# Requires vendored verl patches in vllm_rollout_spmd.py + ray_trainer.py.
export PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1

# Insitu eval metrics: report pass@k AND best@k at K∈{1,4,16,64,256}.
# insitu_eval.py reads these env vars; ray_trainer.py logs both to W&B
# under insitu_eval/<label>/{pass_at_k, best_at_k}.
export INSITU_EVAL_K_VALUES='[1,4,16,64,256]'
export INSITU_EVAL_BEST_AT_K=1

# Carve-out sidecar metadata (consumed by guards/actor_carveout.py).
export PIE_RUN_ID="qwen3_1.7b_pie_maxrl_unnorm_g16_s${SEED}"
export PIE_ALGO="maxrl_unnorm"
export PIE_INIT_CKPT="${PIE_SFT_CKPT}"

# ============================================================
# §13 GO probe config — required for the reward fn to hit GATE B targets.
# Without these the K=5 cap is off (long-tail problems blow the step budget),
# PCH isn't used (~2.65× slower per-compile), and compile tmpdirs may land on
# Lustre (~10× metadata overhead). PCH is rebuilt once per job (~2s).
# ============================================================
export PIE_MEASUREMENT_MAX_TESTS=5
export PIE_COMPILE_SCRATCH="${TMPDIR}"
mkdir -p "${PIE_COMPILE_SCRATCH}"
# §perf-backend: switch reward measurement to HW retired-instructions.
# Reference table built under same backend; MANIFEST.json check at preload
# will refuse a units-mismatched table.
export PIE_BACKEND=perf
export PIE_REFERENCE_TABLE_DIR="${PIE_DATA_ROOT}/reference_table_perf"
export PIE_REFERENCE_TABLE_PICKLE="${PIE_DATA_ROOT}/reference_table_perf.pkl"
PIE_PCH_INCLUDE_PATH=$(${PYTHON} -c "import sys; sys.path.insert(0, '${REPO_ROOT}'); from experiment11_code_optim.measurement.core import setup_pch; print(setup_pch())")
export PIE_PCH_INCLUDE_PATH
echo "PIE reward config: BACKEND=${PIE_BACKEND} MAX_TESTS=${PIE_MEASUREMENT_MAX_TESTS} PCH=${PIE_PCH_INCLUDE_PATH} SCRATCH=${PIE_COMPILE_SCRATCH}"
echo "PIE reference table: ${PIE_REFERENCE_TABLE_DIR}"
export VLLM_USE_V1=0
export RAY_OBJECT_STORE_MEMORY=32000000000
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
echo "=== Train: Qwen3-1.7B Base | PIE | MaxRL_unnorm (GOS) | G=16 | seed=${SEED} | chain_left=${CHAIN_LEFT} ==="
echo "GPUs: ${N_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Model init: ${MODEL_PATH}"
echo "Epoch arithmetic: steps_per_epoch=${PIE_STEPS_PER_EPOCH} total=${PIE_TOTAL_STEPS} eval_steps=${PIE_EVAL_STEPS}"

# ============================================================
# Upfront chaining: submit next link NOW so it survives SIGKILL
# ============================================================
NEXT_CHAIN=$((CHAIN_LEFT - 1))
if [[ ${NEXT_CHAIN} -gt 0 ]]; then
    NEXT_JOB=$(sbatch \
        --account="${SLURM_JOB_ACCOUNT}" \
        --dependency=afterany:${SLURM_JOB_ID} \
        "${REPO_ROOT}/experiment11_code_optim/scripts/qwen3_1.7b_base_pie_maxrl_g16.sh" \
        "${SEED}" "${NEXT_CHAIN}")
    echo "Upfront-chained next job: ${NEXT_JOB} (chains_left=${NEXT_CHAIN})"
else
    echo "Last chain link (chain_left=${CHAIN_LEFT})"
fi

set +e
$PYTHON "${REPO_ROOT}/experiment11_code_optim/train.py" \
  +data.seed=${SEED} \
  algorithm.adv_estimator=gos \
  algorithm.filter_overlong_responses=False \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${PIE_TRAIN_PARQUET}" \
  data.val_files="${PIE_TRAIN_PARQUET}" \
  data.train_batch_size=256 \
  data.max_prompt_length=2560 \
  data.max_response_length=4096 \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.val_kwargs.n=64 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment11_code_optim/reward/pie_reward_fn.py" \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.reference_table_dir="${PIE_REFERENCE_TABLE_DIR}" \
  +custom_reward_function.reward_kwargs.test_case_dir="${PIE_TEST_CASE_DIR}" \
  trainer.total_training_steps=${PIE_TOTAL_STEPS} \
  trainer.total_epochs=3 \
  trainer.project_name=cont-maxrl-pie \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "trainer.logger=['console','wandb']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  "+actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train=False \
  insitu_eval.data_file="${PIE_VAL_PARQUET}" \
  insitu_eval.n_completions=200 \
  insitu_eval.eval_batch_size=32 \
  insitu_eval.steps_per_epoch=${PIE_STEPS_PER_EPOCH} \
  "insitu_eval.eval_steps=${PIE_EVAL_STEPS}" \
  insitu_eval.output_dir="${OUTPUT_DIR}/insitu_eval" \
  ray_init.num_cpus=280

TRAIN_EXIT=$?
set -e

# ============================================================
# Save-time actor carve-out (spec §8.3) — TODO §14 step 7
# verl's rolling checkpoint (max_actor_ckpt_to_keep=1) preserves resume
# integrity but loses per-epoch trajectory. The hook below mirrors only
# sharded model_path + huggingface/ into actor_ckpts/step_*/ AFTER each
# save step. Implementation requires a callback inside verl's save path
# (the SLURM body cannot observe verl's internal saves); the design and
# code are deferred to §14 step 7 per spec.
# ============================================================
PIE_CARVE_HOOK="${REPO_ROOT}/experiment11_code_optim/scripts/actor_carveout_hook.sh"
if [[ -x "${PIE_CARVE_HOOK}" ]]; then
    "${PIE_CARVE_HOOK}" "${OUTPUT_DIR}" "${ACTOR_CKPT_DIR}" || echo "WARN: carve-out hook returned non-zero"
else
    echo "NOTE: actor carve-out hook not yet implemented (spec §14 step 7); per-epoch trajectory will NOT be persisted from this run."
fi

echo ""
echo "=== Training segment complete: PIE | MaxRL_unnorm (GOS) | G=16 seed=${SEED} (exit=${TRAIN_EXIT}) ==="
