#!/usr/bin/env bash
#SBATCH --job-name=maze_binmaxrl_cont
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# SmolLM2-135M | Maze 17x17 | Binary MaxRL | Continuous reward
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

# --- Method config (ONLY these 4 lines vary per script) ---
METHOD="binmaxrl"
REWARD="cont"
ADV_ESTIMATOR="binary_maxrl"
REWARD_FN_NAME="compute_score"
# -----------------------------------------------------------

SEED="${SEED:-0}"

MODEL_PATH="${REPO_ROOT}/checkpoints/smollm2_135m_maze17x17_sft"
TRAIN_DATA="${REPO_ROOT}/data/maze_17x17/train.parquet"
VAL_DATA="${REPO_ROOT}/data/maze_17x17/val.parquet"
OUTPUT_DIR="${REPO_ROOT}/results/maze17x17_${METHOD}_${REWARD}_g16_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# W&B
export WANDB_API_KEY="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"
export WANDB_ENTITY="stablegradients"
export WANDB_RUN_GROUP="maze17x17_${METHOD}_${REWARD}_g16"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export VERL_REWARD_WORKERS=64
export VLLM_USE_V1=0
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

N_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
echo "=== Maze Train: ${METHOD} ${REWARD} seed=${SEED} ==="
echo "GPUs: ${N_GPUS}, Model: SmolLM2-135M (SFT)"

$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  algorithm.filter_overlong_responses=True \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=384 \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=False \
  data.apply_chat_template=False \
  data.reward_fn_key=data_source \
  data.seed=${SEED} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-5 \
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
  actor_rollout_ref.rollout.seed=${SEED} \
  actor_rollout_ref.rollout.val_kwargs.n=64 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment6_maze/src/reward_function.py" \
  custom_reward_function.name=${REWARD_FN_NAME} \
  trainer.total_training_steps=500 \
  trainer.project_name=cont-maxrl-maze \
  trainer.experiment_name=maze17x17_${METHOD}_${REWARD}_g16_s${SEED} \
  "trainer.logger=['console','wandb']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=25 \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.val_before_train=True \
  insitu_eval.data_file="${REPO_ROOT}/data/maze_17x17/val.parquet" \
  insitu_eval.n_completions=64 \
  insitu_eval.eval_batch_size=32 \
  "insitu_eval.eval_steps=[0,25,50,75,100,125,150,175,200,250,300,350,400,450,500]" \
  insitu_eval.output_dir="${OUTPUT_DIR}/eval" \
  ray_init.num_cpus=280 \
  "$@"

echo "=== Training complete: maze17x17_${METHOD}_${REWARD}_g16_s${SEED} ==="
