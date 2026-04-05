#!/usr/bin/env bash
#SBATCH --job-name=q3_1.7b_maxrl_g64
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=47:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Qwen3-1.7B (Base) | DeepCoder | MaxRL (gap-over-survivors) | G=64 | bgmo | 4-node (16 GPUs)
# Continuous reward (fraction of test cases passed)
# Self-chaining: resubmits with --dependency=afterany
# Usage: sbatch qwen3_1.7b_base_deepcoder_maxrl_g64.sh [SEED] [CHAIN_LEFT]
# ============================================================

SEED=${1:-0}
CHAIN_LEFT=${2:-20}

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

# Trap SIGTERM (SLURM wall-time kill) so self-chaining still fires
trap 'echo "Caught SIGTERM (wall-time limit) — proceeding to self-chain"' TERM

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

OUTPUT_DIR="${STORAGE_ROOT}/results/qwen3_1.7b_deepcoder_maxrl_g64_s${SEED}"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Multi-node networking: system nccl-ofi-plugin for Slingshot CXI
export LD_LIBRARY_PATH="/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib:/opt/cray/libfabric/2.3.1/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_NET_PLUGIN="/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib/libnccl-net.so"
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=1
export NCCL_PLUGIN_P2P=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DMABUF_ENABLE=0
export FI_PROVIDER=cxi
export NCCL_NET_OFI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_DEBUG=WARN

# W&B
export WANDB_API_KEY="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"
export WANDB_ENTITY="stablegradients"
export WANDB_RUN_GROUP="qwen3-1.7b-base_deepcoder_maxrl_G64"

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export VERL_REWARD_WORKERS=64
export VLLM_USE_V1=0
export RAY_OBJECT_STORE_MEMORY=32000000000
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ============================================================
# Multi-node Ray cluster setup (4 nodes)
# ============================================================
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE="${NODES[0]}"
NUM_NODES=${#NODES[@]}
N_GPUS_PER_NODE=4

echo "=== Train: Qwen3-1.7B Base | MaxRL | G=64 | seed=${SEED} | chain_left=${CHAIN_LEFT} ==="
echo "Nodes: ${NUM_NODES}, GPUs/node: ${N_GPUS_PER_NODE}, Total GPUs: $((NUM_NODES * N_GPUS_PER_NODE))"
echo "Head node: ${HEAD_NODE}, All nodes: ${NODES[*]}"
echo "Output: ${OUTPUT_DIR}"

HEAD_IP=$(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1)
RAY_PORT=6379
echo "Head IP: ${HEAD_IP}:${RAY_PORT}"
if [[ -z "${HEAD_IP}" ]]; then
    echo "FATAL: Could not resolve IPv4 address for head node" >&2
    exit 1
fi

# Capture Slingshot VNI credentials from srun context
echo "Capturing Slingshot VNI credentials..."
SLINGSHOT_ENV=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    bash -c 'echo "SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-}"; echo "SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-}"; echo "SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-}"')
eval "${SLINGSHOT_ENV}"
export SLINGSHOT_VNIS SLINGSHOT_SVC_IDS SLINGSHOT_DEVICES
echo "SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-unset}"

if [[ -z "${SLINGSHOT_VNIS:-}" ]]; then
    echo "WARNING: SLINGSHOT_VNIS is empty — cross-node NCCL may fail"
fi

# Start Ray head locally
ray stop --force 2>/dev/null || true
echo "Starting Ray head locally on $(hostname)..."
ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port=${RAY_PORT} \
    --num-cpus=$((SLURM_CPUS_PER_TASK - N_GPUS_PER_NODE)) \
    --num-gpus=${N_GPUS_PER_NODE}
sleep 10

ray status
echo "Head node ready"

# Start Ray workers on remaining nodes
WORKER_NUM=$((NUM_NODES - 1))
for ((i=1; i<=WORKER_NUM; i++)); do
    WORKER_NODE="${NODES[$i]}"
    echo "Starting Ray worker on ${WORKER_NODE}..."
    srun --nodes=1 --ntasks=1 -w "${WORKER_NODE}" --export=ALL \
        bash -c "
            export SLINGSHOT_VNIS='${SLINGSHOT_VNIS}'
            export SLINGSHOT_SVC_IDS='${SLINGSHOT_SVC_IDS}'
            export SLINGSHOT_DEVICES='${SLINGSHOT_DEVICES}'
            export LD_LIBRARY_PATH='/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib:/opt/cray/libfabric/2.3.1/lib64:${LD_LIBRARY_PATH:-}'
            export NCCL_NET_PLUGIN='/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib/libnccl-net.so'
            export NCCL_SOCKET_IFNAME=hsn
            export GLOO_SOCKET_IFNAME=hsn0
            export NCCL_CROSS_NIC=1
            export NCCL_IB_DISABLE=1
            export NCCL_PLUGIN_P2P=0
            export NCCL_NET_GDR_LEVEL=PHB
            export NCCL_DMABUF_ENABLE=0
            export FI_PROVIDER=cxi
            export NCCL_NET_OFI_PROVIDER=cxi
            export FI_CXI_DEFAULT_CQ_SIZE=131072
            export FI_CXI_DEFAULT_TX_SIZE=2048
            export FI_CXI_DISABLE_HOST_REGISTER=1
            export FI_CXI_RX_MATCH_MODE=hybrid
            export FI_CXI_RDZV_PROTO=alt_read
            export FI_CXI_RDZV_EAGER_SIZE=0
            export FI_MR_CACHE_MONITOR=userfaultfd
            export NCCL_DEBUG=WARN
            ray start --address='${HEAD_IP}:${RAY_PORT}' \
                --num-cpus=$((SLURM_CPUS_PER_TASK - N_GPUS_PER_NODE)) \
                --num-gpus=${N_GPUS_PER_NODE}
            sleep infinity
        " &
    sleep 10
done

sleep 40

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "=== Ray cluster status ==="
ray status
echo "=========================="

# ============================================================
# Training
# ============================================================
# Disable errexit so SIGTERM/non-zero exit doesnt skip chaining
set +e
$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  +data.seed=${SEED} \
  algorithm.adv_estimator=maxrl_unnorm \
  algorithm.filter_overlong_responses=True \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${REPO_ROOT}/data/deepcoder_train.parquet" \
  data.val_files="${REPO_ROOT}/data/deepcoder_train.parquet" \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
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
  actor_rollout_ref.rollout.n=64 \
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
  custom_reward_function.path="${REPO_ROOT}/experiment4_llm/src/reward_function.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=1000 \
  trainer.total_epochs=15 \
  trainer.project_name=cont-maxrl-code \
  trainer.experiment_name="qwen3_1.7b_deepcoder_maxrl_g64_s${SEED}" \
  "trainer.logger=['console','wandb']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=${NUM_NODES} \
  trainer.save_freq=83 \
  trainer.test_freq=-1 \
  trainer.resume_mode=auto \
  trainer.val_before_train=False \
  insitu_eval.data_file="${REPO_ROOT}/data/livecodebench_v5_test.parquet" \
  insitu_eval.n_completions=200 \
  insitu_eval.eval_batch_size=32 \
  insitu_eval.steps_per_epoch=83 \
  "insitu_eval.eval_steps=[83,166,249,332,415,498,581,664,747,830,913,996]" \
  insitu_eval.output_dir="${OUTPUT_DIR}/eval_livecodebench" \
  +insitu_eval.extra_data_files.humaneval_plus="${REPO_ROOT}/data/evalplus_humaneval_test.parquet" \
  +insitu_eval.extra_data_files.mbpp_plus="${REPO_ROOT}/data/evalplus_mbpp_test.parquet"

TRAIN_EXIT=$?
set -e

echo ""
echo "=== Training segment complete: MaxRL G=64 seed=${SEED} (exit=${TRAIN_EXIT}) ==="

# Cleanup Ray
ray stop --force 2>/dev/null || true

# ============================================================
# Self-chaining: submit next segment if chains remain
# ============================================================
NEXT_CHAIN=$((CHAIN_LEFT - 1))
if [[ ${NEXT_CHAIN} -gt 0 ]]; then
    NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} \
        "${REPO_ROOT}/experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_maxrl_g64.sh" \
        "${SEED}" "${NEXT_CHAIN}")
    echo "Chained next job: ${NEXT_JOB} (chains_left=${NEXT_CHAIN})"
else
    echo "No more chains (next_chain=${NEXT_CHAIN}, train_exit=${TRAIN_EXIT})"
fi
