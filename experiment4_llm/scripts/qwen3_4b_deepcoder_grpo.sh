#!/usr/bin/env bash
#SBATCH --job-name=qwen3_4b_grpo
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
# Qwen3-4B (Thinking) | DeepCoder | GRPO | G=16
# Continuous reward (fraction of test cases passed)
# ============================================================

# Activate conda environment (before set -u, conda scripts use unset vars)
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

OUTPUT_DIR="${REPO_ROOT}/results/qwen3_4b_deepcoder_grpo"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system network plugin (incompatible with conda NCCL)
# Single-node uses NVLink/shared memory, no libfabric needed
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# W&B
export WANDB_API_KEY="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"
export WANDB_ENTITY="stablegradients"

# Caches
export HF_HOME="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
# Maximize parallel reward computation (subprocess-based sandbox)
export VERL_REWARD_WORKERS=256
# verl requires vLLM v0 engine (v1 doesn't implement free_cache_engine)
export VLLM_USE_V1=0
# Ray and tmp on local disk (avoids NFS socket/cleanup issues)
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

# Project must be importable (for custom advantage estimators)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# wandb
export WANDB_ENTITY="stablegradients"
export WANDB_RUN_GROUP="qwen3_4b_deepcoder_grpo"

# Use SLURM-assigned GPU count, fall back to nvidia-smi for local runs
N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
echo "=== Train: Qwen3-4B GRPO on DeepCoder ==="
echo "GPUs: ${N_GPUS}, Model: Qwen/Qwen3-4B"
echo "Method: grpo, G=16, max_completion_length=4096"
echo "Output: ${OUTPUT_DIR}"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  algorithm.adv_estimator=grpo \
  algorithm.filter_overlong_responses=True \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${REPO_ROOT}/data/deepcoder_train.parquet" \
  data.val_files="${REPO_ROOT}/data/deepcoder_val.parquet" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=True \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=50 \
  actor_rollout_ref.actor.optim.warmup_style=cosine \
  actor_rollout_ref.actor.optim.total_training_steps=2000 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  reward_model.enable=False \
  custom_reward_function.path="${REPO_ROOT}/experiment4_llm/src/reward_function.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=2000 \
  trainer.total_epochs=15 \
  trainer.project_name=cont-maxrl-code \
  trainer.experiment_name=qwen3_4b_deepcoder_grpo \
  "trainer.logger=['console','wandb']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=146 \
  trainer.test_freq=146 \
  trainer.val_before_train=False \
  insitu_eval.data_file="${REPO_ROOT}/data/livecodebench_v5_test.parquet" \
  insitu_eval.n_completions=200 \
  insitu_eval.eval_batch_size=16 \
  insitu_eval.steps_per_epoch=146 \
  "insitu_eval.eval_steps=[0,730,1460,2000]" \
  insitu_eval.output_dir="${OUTPUT_DIR}/eval_livecodebench" \
  ray_init.num_cpus=280

echo ""
echo "=== Training complete: qwen3_4b_deepcoder_grpo ==="
