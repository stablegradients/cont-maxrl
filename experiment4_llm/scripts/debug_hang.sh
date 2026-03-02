#!/usr/bin/env bash
#SBATCH --job-name=debug_hang
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --exclude=gpu20,gpu21,gpu22,gpu23
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/results/debug_hang/slurm_%j.out
#SBATCH --error=/zfsauton/scratch/shrinivr/cont-maxrl/results/debug_hang/slurm_%j.err
set -euo pipefail

# ============================================================
# Debug: 1.5B model, 4 GPUs, tiny batch, verbose logging
# Goal: find where the training hangs
# ============================================================

REPO_ROOT="/zfsauton/scratch/shrinivr/cont-maxrl"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

OUTPUT_DIR="${REPO_ROOT}/results/debug_hang"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA
export CUDA_HOME="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Caches
export HF_HOME="/zfsauton/scratch/shrinivr/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

# Project importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── Debug logging ──
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=DEBUG
export RAY_DEDUP_LOGS=0

# Force vLLM v0 engine
export VLLM_USE_V1=0

echo "=== Debug Hang: 1.5b, 4 GPUs, batch=4, G=4, 3 steps ==="
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "VLLM_USE_V1=${VLLM_USE_V1}"
echo "Output: ${OUTPUT_DIR}"
date

$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${REPO_ROOT}/data/taco_train.parquet" \
  data.val_files="${REPO_ROOT}/data/taco_val.parquet" \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-1.5B-Instruct \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=128 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  reward_model.enable=False \
  custom_reward_function.path="${REPO_ROOT}/experiment4_llm/src/reward_function.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=3 \
  trainer.total_epochs=30 \
  trainer.project_name=debug \
  trainer.experiment_name=debug_hang_1.5b \
  "trainer.logger=['console']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  ray_init.num_cpus=32

echo ""
echo "=== Debug complete ==="
date
