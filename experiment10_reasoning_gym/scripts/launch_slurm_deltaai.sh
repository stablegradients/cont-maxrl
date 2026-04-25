#!/usr/bin/env bash
#SBATCH --job-name=rgym
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --time=47:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/rgym_%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/rgym_%x_%j.err
#
# SLURM wrapper for a single Reasoning Gym training run on DeltaAI (1 GH200).
#
# Positional args forwarded to launch.sh:
#   $1 task        word_sorting | sentence_reordering | game_of_life | knights_knaves
#   $2 difficulty  int (task-specific)
#   $3 estimator   gos | maxrl | rloo | grpo
#   $4 reward_mode continuous | binary
#   $5 seed        (default 42)
#   $6 chain_left  remaining chain links (default 1, i.e. this job chains 1 more after itself)
#
# Usage:
#   sbatch --account=bgmo-dtai-gh scripts/launch_slurm_deltaai.sh word_sorting 4 gos continuous 42
#
# The script upfront-submits its next chain link (with --dependency=afterany)
# BEFORE training starts, so even a SIGKILL leaves the chain intact. The child
# run resumes from the latest rolling checkpoint via trainer.resume_mode=auto.

set -eo pipefail    # -u added after conda activate (conda hooks use unbound vars)

TASK=$1
DIFFICULTY=$2
ESTIMATOR=$3
REWARD_MODE=$4
SEED="${5:-42}"
CHAIN_LEFT="${6:-1}"

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -u

export REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
export STORAGE_ROOT="/work/nvme/bgmo/sramasubramanian/cont-maxrl"

cd "${REPO_ROOT}"

# Source .env for WANDB_API_KEY, HF_TOKEN, WANDB_ENTITY
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

# Conda CUDA (not system CUDA)
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL: disable system plugin (incompatible with conda NCCL on this cluster)
unset NCCL_NET || true
unset NCCL_NET_PLUGIN || true
export NCCL_PLUGIN_P2P=0

# Verl / Ray / vLLM env
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VERL_REWARD_WORKERS=64
export RAY_OBJECT_STORE_MEMORY=32000000000
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${VLLM_CACHE_ROOT}"

# Per-run tmp dirs (Ray + /tmp). Must stay short: AF_UNIX socket path
# limit is 108 bytes, and Ray appends ~60 chars of session/socket suffix.
RUN_TAG="${TASK}_${ESTIMATOR}_${REWARD_MODE}_diff${DIFFICULTY}_s${SEED}"   # matches launch.sh EXPERIMENT_NAME
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID:-$$}"
export TMPDIR="/tmp/tmp_${SLURM_JOB_ID:-$$}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

# W&B grouping: multi-seed runs on the same (task,est,reward,diff) share a group
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${TASK}_${ESTIMATOR}_${REWARD_MODE}_diff${DIFFICULTY}"

# Cross-difficulty validation per task (from dispatch_wssr.sh defaults)
case "$TASK" in
    word_sorting)        VAL_DIFFS_DEFAULT="4,8,12" ;;
    sentence_reordering) VAL_DIFFS_DEFAULT="3,7,10" ;;
    game_of_life)        VAL_DIFFS_DEFAULT="4,7,10" ;;
    *)                   VAL_DIFFS_DEFAULT="${DIFFICULTY}" ;;
esac
export VAL_DIFFS="${VAL_DIFFS:-$VAL_DIFFS_DEFAULT}"

# Checkpoint / experiment naming — single W&B project for the whole sweep
PROJECT_NAME="Reasoning-GYM-GOS"
EXPERIMENT_NAME="${RUN_TAG}"
OUTPUT_DIR="${STORAGE_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"
export PROJECT_NAME EXPERIMENT_NAME
export N_GPUS=1
export TOTAL_EPOCHS=10
export SAVE_FREQ=78           # save every epoch (20000/256 = 78)
export MAX_CKPT_KEEP=3
export TEST_FREQ=78           # eval every epoch

# Upfront chain submission — survives SIGKILL of current job
if [[ ${CHAIN_LEFT} -gt 0 ]]; then
    NEXT_CHAIN=$((CHAIN_LEFT - 1))
    NEXT_JOB=$(sbatch --parsable \
        --account="${SLURM_JOB_ACCOUNT}" \
        --dependency=afterany:${SLURM_JOB_ID} \
        --job-name="rgym_${RUN_TAG}" \
        "${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch_slurm_deltaai.sh" \
        "${TASK}" "${DIFFICULTY}" "${ESTIMATOR}" "${REWARD_MODE}" "${SEED}" "${NEXT_CHAIN}")
    echo "Chained next job: ${NEXT_JOB} (chain_left=${NEXT_CHAIN}, account=${SLURM_JOB_ACCOUNT})"
fi

echo "=== Reasoning Gym Train [DeltaAI] ==="
echo "  run_tag=${RUN_TAG}"
echo "  account=${SLURM_JOB_ACCOUNT:-unknown} job_id=${SLURM_JOB_ID:-local}"
echo "  val_diffs=${VAL_DIFFS}"
# GoL has longer rollouts (1024 vs 512) + larger answer grids → KV cache OOMs
# when waking up after training frees GPU memory. Drop util for GoL only.
case "$TASK" in
    game_of_life) GPU_MEM_UTIL=0.7  ;;
    *)            GPU_MEM_UTIL=0.85 ;;
esac

echo "  output_dir=${OUTPUT_DIR}"
echo "  chain_left=${CHAIN_LEFT}"
echo "  gpu_memory_utilization=${GPU_MEM_UTIL}"
echo

set +e
bash "${REPO_ROOT}/experiment10_reasoning_gym/scripts/launch.sh" \
    "${TASK}" "${DIFFICULTY}" "${ESTIMATOR}" "${REWARD_MODE}" "${SEED}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.val_before_train=False \
    ray_init.num_cpus=64 \
    reward_model.launch_reward_fn_async=True \
    data.enable_thinking=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True
TRAIN_EXIT=$?
set -e

echo ""
echo "=== Training segment complete: ${RUN_TAG} (exit=${TRAIN_EXIT}) ==="
