#!/usr/bin/env bash
# Minimal dry-run: 0.5B model, G=4, 3 steps, GPUs 2,3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Use conda env's python directly (no conda activate needed)
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# Use only GPUs 2,3
export CUDA_VISIBLE_DEVICES=1,2

# Conda's CUDA 12.4
export CUDA_HOME="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# All caches under repo root
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export RAY_TMPDIR="/tmp/ray_${USER}"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${RAY_TMPDIR}" "${VLLM_CACHE_ROOT}" "${TMPDIR}"

# HF token for model downloads (set in environment or ~/.huggingface/token)
# export HF_TOKEN="your_token_here"

# Ensure our project is importable
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR="${REPO_ROOT}/results/dry_run"
mkdir -p "${OUTPUT_DIR}"

echo "=== Dry Run: 0.5B model, maxrl_unnorm, G=4, 3 steps ==="
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Output: ${OUTPUT_DIR}"

# Only override keys that exist in verl's ppo_trainer.yaml config.
# Use + prefix for keys that don't exist in the default config.
$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  algorithm.adv_estimator=maxrl_unnorm \
  algorithm.use_kl_in_reward=False \
  data.train_files="${REPO_ROOT}/data/dry_run/train.parquet" \
  data.val_files="${REPO_ROOT}/data/dry_run/test.parquet" \
  data.train_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  data.shuffle=False \
  data.filter_overlong_prompts=True \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.load_format=dummy_dtensor \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  reward_model.enable=False \
  custom_reward_function.path="${REPO_ROOT}/experiment4_llm/src/reward_function.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=3 \
  trainer.total_epochs=30 \
  trainer.project_name=dry_run \
  trainer.experiment_name=test_maxrl_0.5b \
  "trainer.logger=['console']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  ray_init.num_cpus=8

echo ""
echo "=== Dry run complete ==="
