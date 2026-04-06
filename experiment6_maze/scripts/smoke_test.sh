#!/usr/bin/env bash
#SBATCH --job-name=maze_smoke
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1
#SBATCH --array=0-2
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%A_%a.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%A_%a.err

# ============================================================
# Smoke test: 5 training steps, NO insitu eval (just verify pipeline)
#   0: GOS (maxrl_unnorm) + continuous
#   1: GRPO + continuous
#   2: BinMaxRL (binary_maxrl) + continuous
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

TASK_ID=${SLURM_ARRAY_TASK_ID}

case ${TASK_ID} in
  0) METHOD="gos";      ADV_ESTIMATOR="maxrl_unnorm"; REWARD_FN_NAME="compute_score" ;;
  1) METHOD="grpo";     ADV_ESTIMATOR="grpo";         REWARD_FN_NAME="compute_score" ;;
  2) METHOD="binmaxrl"; ADV_ESTIMATOR="binary_maxrl"; REWARD_FN_NAME="compute_score" ;;
  *) echo "Invalid TASK_ID=${TASK_ID}"; exit 1 ;;
esac

SEED=0

# Use the latest SFT checkpoint
SFT_DIR="${REPO_ROOT}/checkpoints/maze17x17_qwen2_3m_sft"
MODEL_PATH="${SFT_DIR}/ckpt-$(ls "${SFT_DIR}" | grep '^ckpt-' | sed 's/ckpt-//' | sort -n | tail -1)"
if [ -z "${MODEL_PATH}" ]; then
  echo "ERROR: No SFT checkpoint found in ${SFT_DIR}"
  exit 1
fi
echo "Using SFT checkpoint: ${MODEL_PATH}"

TRAIN_DATA="${REPO_ROOT}/data/maze_17x17/train.parquet"
VAL_DATA="${REPO_ROOT}/data/maze_17x17/val.parquet"
OUTPUT_DIR="${REPO_ROOT}/results/smoke_${METHOD}_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# No W&B for smoke test
export WANDB_MODE="disabled"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VERL_REWARD_WORKERS=16
export RAY_TMPDIR="/tmp/ray_${USER}_smoke_${TASK_ID}"
export TMPDIR="/tmp/${USER}_verl_smoke_${TASK_ID}"
mkdir -p "${TORCH_HOME}" "${RAY_TMPDIR}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

N_GPUS=1
echo "=== SMOKE TEST: ${METHOD} (${ADV_ESTIMATOR}) seed=${SEED} ==="
echo "GPUs: ${N_GPUS}, Steps: 5, Model: Qwen2 3.94M (HF rollout)"
echo "Start time: $(date)"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  algorithm.filter_overlong_responses=True \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=256 \
  data.max_prompt_length=320 \
  data.max_response_length=180 \
  data.shuffle=True \
  data.filter_overlong_prompts=False \
  data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  ++actor_rollout_ref.rollout.micro_batch_size=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment6_maze/src/reward_function.py" \
  custom_reward_function.name=${REWARD_FN_NAME} \
  trainer.total_training_steps=5 \
  trainer.project_name=cont-maxrl-maze \
  trainer.experiment_name=smoke_${METHOD}_s${SEED} \
  "trainer.logger=['console']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train=False \
  ray_init.num_cpus=64 \
  "$@"

echo "End time: $(date)"
echo "=== SMOKE TEST COMPLETE: ${METHOD} ==="
