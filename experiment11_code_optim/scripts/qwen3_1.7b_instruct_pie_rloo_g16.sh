#!/usr/bin/env bash
#SBATCH --job-name=q3_1.7b_inst_pie_rloo_g16
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
# Qwen3-1.7B-Instruct | PIE (log2-speedup @ -O0) | RLOO | G=16
# 3-arm comparison: this is the RLOO arm. MaxRL_unnorm and GRPO siblings differ ONLY
# in algorithm.adv_estimator + naming strings; nothing else.
#
# No SFT cold-start: trains directly from the Instruct model. The reward is
#   r = max(0, log2(src_instr_count / cand_instr_count))   on gate-pass
#   r = 0                                                   otherwise
# src_instr_count is per-record, embedded in ground_truth by build_pie_parquet.py.
#
# Self-chaining: resubmits with --dependency=afterany
# Usage: sbatch qwen3_1.7b_instruct_pie_rloo_g16.sh [SEED] [CHAIN_LEFT]
# ============================================================

SEED=${1:-0}
CHAIN_LEFT=${2:-4}

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/work/nvme/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# ============================================================
# PIE-specific paths
# ============================================================
PIE_DATA_ROOT="/work/nvme/bgmo/sramasubramanian/pie_data"
PIE_TEST_CASE_DIR="${PIE_TEST_CASE_DIR:-${PIE_DATA_ROOT}/tests}"
PIE_TRAIN_PARQUET="${PIE_TRAIN_PARQUET:-${PIE_DATA_ROOT}/parquet/pie_train.parquet}"
PIE_VAL_PARQUET="${PIE_VAL_PARQUET:-${PIE_DATA_ROOT}/parquet/pie_val_200.parquet}"
export PIE_TEST_CASE_DIR

# ============================================================
# Epoch arithmetic — PLACEHOLDERS. After running build_src_instr_table.py +
# build_pie_parquet.py the actual N_train will be known; recompute:
#   PIE_STEPS_PER_EPOCH = ceil(N_train / 256)
#   PIE_TOTAL_STEPS     = 3 * PIE_STEPS_PER_EPOCH
#   PIE_EVAL_STEPS      = every 100 steps
# These defaults are derived from the 2026-05-24 rebuild: N_train = 69,464
# after the -O0 drop (8,385 no_instr + 118 overlong); ceil(N/256)=272.
# ============================================================
PIE_STEPS_PER_EPOCH="${PIE_STEPS_PER_EPOCH:-272}"
PIE_TOTAL_STEPS="${PIE_TOTAL_STEPS:-816}"
PIE_EVAL_STEPS="${PIE_EVAL_STEPS:-[100,200,300,400,500,600,700,800]}"

# ============================================================
# Model init: Qwen3-1.7B-Instruct directly. No SFT cold-start.
# Native <|im_start|> chat template — do NOT apply the identity-template
# hack used for the Base/SFT setup; verl's apply_chat_template renders
# correctly.
# ============================================================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"

OUTPUT_DIR="${STORAGE_ROOT}/results/qwen3_1.7b_instruct_pie_rloo_g16_s${SEED}"
ACTOR_CKPT_DIR="${OUTPUT_DIR}/actor_ckpts"
mkdir -p "${OUTPUT_DIR}" "${ACTOR_CKPT_DIR}"
# PIE_ACTOR_CARVEOUT_DIR disabled — with save_freq=10 we want pure rolling-1
# (carve-out would create ~80 hard-linked ckpts ≈ 1.5TB/arm at -O0). Re-enable
# only if per-step archive is wanted; tweak guards/actor_carveout.py to filter.
# export PIE_ACTOR_CARVEOUT_DIR="${ACTOR_CKPT_DIR}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL, single-node)
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# ============================================================
# W&B
# ============================================================
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in submission env}"
export WANDB_ENTITY="${WANDB_ENTITY:-stablegradients}"
export WANDB_RUN_GROUP="qwen3-1.7b-instruct_pie_rloo_G16"
EXPERIMENT_NAME="${WANDB_RUN_GROUP}_s${SEED}"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export VERL_REWARD_WORKERS=64

# Route insitu_eval's hardcoded reward import to PIE's reward fn.
export PIE_MODE=1
# Skip the actor old_log_prob recompute via vendored verl patches.
# PIE_SKIP_OLD_LOG_PROB_RECOMPUTE disabled — let verl recompute old_log_prob
# using the actor model itself, so pg_clipfrac = exactly 0 (no bf16 drift
# between vLLM and FSDP). Also unlocks real entropy logging via the
# entropy = compute_log_prob(... calculate_entropy=True) side-effect path.
# Cost: ~30 s/step extra; over 816 steps adds ~7h per arm.
# export PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1

# Insitu eval metrics: pass@K AND best@K AND best_speedup@K at K∈{1,4,16,64,256}.
# 256 lets us hit best_speedup@256 (the spec's headline metric).
export INSITU_EVAL_K_VALUES='[1,4,16,64,256]'
export INSITU_EVAL_BEST_AT_K=1

# Carve-out sidecar metadata (consumed by guards/actor_carveout.py)
export PIE_RUN_ID="qwen3_1.7b_instruct_pie_rloo_g16_s${SEED}"
export PIE_ALGO="rloo"
export PIE_INIT_CKPT="${MODEL_PATH}"

# ============================================================
# Reward config
# -O0 compile flags inside core.py — the prior -O3 flatten was the reason for
# this rewrite (see docs/DATA_ANALYSIS_2026_05_22_23.md). PIE_BACKEND=perf is
# the only backend stable enough under -O0 (walltime had 28× rel-spread).
# ============================================================
export PIE_MEASUREMENT_MAX_TESTS=5
export PIE_PER_TEST_TIMEOUT=30
export PIE_COMPILE_SCRATCH="${TMPDIR:-/tmp}/${USER}_pie_scratch"
mkdir -p "${PIE_COMPILE_SCRATCH}"
export PIE_BACKEND=perf
PIE_PCH_INCLUDE_PATH=$(${PYTHON} -c "import sys; sys.path.insert(0, '${REPO_ROOT}'); from experiment11_code_optim.measurement.core import setup_pch; print(setup_pch())")
export PIE_PCH_INCLUDE_PATH
echo "PIE reward config: BACKEND=${PIE_BACKEND} MAX_TESTS=${PIE_MEASUREMENT_MAX_TESTS} PCH=${PIE_PCH_INCLUDE_PATH} SCRATCH=${PIE_COMPILE_SCRATCH}"

export VLLM_USE_V1=0
export RAY_OBJECT_STORE_MEMORY=32000000000
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
echo "=== Train: Qwen3-1.7B-Instruct | PIE -O0 | RLOO | G=16 | seed=${SEED} | chain_left=${CHAIN_LEFT} ==="
echo "GPUs: ${N_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "Model init: ${MODEL_PATH}"
echo "Epoch arithmetic: steps_per_epoch=${PIE_STEPS_PER_EPOCH} total=${PIE_TOTAL_STEPS} eval_steps=${PIE_EVAL_STEPS}"

# ============================================================
# Upfront chaining
# ============================================================
NEXT_CHAIN=$((CHAIN_LEFT - 1))
if [[ ${NEXT_CHAIN} -gt 0 ]]; then
    NEXT_JOB=$(sbatch \
        --account="${SLURM_JOB_ACCOUNT}" \
        --dependency=afterany:${SLURM_JOB_ID} \
        "${REPO_ROOT}/experiment11_code_optim/scripts/qwen3_1.7b_instruct_pie_rloo_g16.sh" \
        "${SEED}" "${NEXT_CHAIN}")
    echo "Upfront-chained next job: ${NEXT_JOB} (chains_left=${NEXT_CHAIN})"
else
    echo "Last chain link (chain_left=${CHAIN_LEFT})"
fi

set +e
$PYTHON "${REPO_ROOT}/experiment11_code_optim/train.py" \
  +data.seed=${SEED} \
  algorithm.adv_estimator=rloo \
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
  actor_rollout_ref.actor.clip_ratio=100 \
  actor_rollout_ref.actor.clip_ratio_high=100 \
  actor_rollout_ref.actor.grad_clip=2.0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=6656 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.max_num_seqs=1024 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment11_code_optim/reward/pie_reward_fn.py" \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.test_case_dir="${PIE_TEST_CASE_DIR}" \
  trainer.total_training_steps=${PIE_TOTAL_STEPS} \
  trainer.total_epochs=3 \
  trainer.project_name=cont-maxrl-pie \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  "trainer.logger=['console','wandb']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  "+actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train=False \
  insitu_eval.data_file="${PIE_VAL_PARQUET}" \
  insitu_eval.n_completions=256 \
  insitu_eval.eval_batch_size=32 \
  insitu_eval.steps_per_epoch=${PIE_STEPS_PER_EPOCH} \
  "insitu_eval.eval_steps=${PIE_EVAL_STEPS}" \
  insitu_eval.output_dir="${OUTPUT_DIR}/insitu_eval" \
  insitu_eval.label="pie_val_200" \
  ray_init.num_cpus=280

TRAIN_EXIT=$?
set -e

PIE_CARVE_HOOK="${REPO_ROOT}/experiment11_code_optim/scripts/actor_carveout_hook.sh"
if [[ -x "${PIE_CARVE_HOOK}" ]]; then
    "${PIE_CARVE_HOOK}" "${OUTPUT_DIR}" "${ACTOR_CKPT_DIR}" || echo "WARN: carve-out hook returned non-zero"
fi

echo ""
echo "=== Training segment complete: PIE -O0 | RLOO | G=16 seed=${SEED} (exit=${TRAIN_EXIT}) ==="
