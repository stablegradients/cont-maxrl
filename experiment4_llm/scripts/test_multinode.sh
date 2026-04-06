#!/usr/bin/env bash
#SBATCH --job-name=test_multinode
#SBATCH --account=bgtv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=72
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Multi-node smoke test: 2 nodes (8 GPUs), 2 training steps
# Validates: Ray cluster, FSDP init, vLLM rollout, reward, ckpt
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -euo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1

OUTPUT_DIR="${REPO_ROOT}/results/test_multinode_2node"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Conda CUDA
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Multi-node networking: use system nccl-ofi-plugin for Slingshot CXI
# Settings from: module show nccl-ofi-plugin/1.18.0-cuda129
export LD_LIBRARY_PATH="/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib:/opt/cray/libfabric/2.3.1/lib64:${LD_LIBRARY_PATH:-}"
export NCCL_NET_PLUGIN="/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib/libnccl-net.so"
export NCCL_SOCKET_IFNAME=hsn
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=1
export NCCL_PLUGIN_P2P=0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DMABUF_ENABLE=0
# CXI fabric tuning (from system module)
export FI_PROVIDER=cxi
export NCCL_NET_OFI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_MR_CACHE_MONITOR=userfaultfd
# Debug (verbose for test)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
# Note: SLINGSHOT_VNIS captured later from srun context (not available in sbatch env)

# W&B — disabled for test
export WANDB_MODE=disabled

# Caches
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export VERL_REWARD_WORKERS=16
export VLLM_USE_V1=0
export RAY_TMPDIR="/tmp/ray_${USER}"
export TMPDIR="/tmp/${USER}_verl_tmp"
mkdir -p "${TORCH_HOME}" "${VLLM_CACHE_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ============================================================
# Multi-node Ray cluster setup
# Following Fahim's pattern: head locally, workers via srun
# ============================================================
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE="${NODES[0]}"
NUM_NODES=${#NODES[@]}
N_GPUS_PER_NODE=4

echo "=== Multi-node test: ${NUM_NODES} nodes, ${N_GPUS_PER_NODE} GPUs/node ==="
echo "Head node: ${HEAD_NODE}"
echo "All nodes: ${NODES[*]}"

# Get head node IPv4 address
HEAD_IP=$(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1)
RAY_PORT=6379
echo "Head IP: ${HEAD_IP}:${RAY_PORT}"
if [[ -z "${HEAD_IP}" ]]; then
    echo "FATAL: Could not resolve IPv4 address for head node" >&2
    exit 1
fi

# ============================================================
# Capture Slingshot VNI credentials from srun context
# SLURM's CXI plugin only injects SLINGSHOT_VNIS/SVC_IDS into
# srun-launched tasks, NOT the sbatch script itself. We must
# capture them and export so Ray workers inherit them.
# ============================================================
echo "Capturing Slingshot VNI credentials from srun context..."
SLINGSHOT_ENV=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
    bash -c 'echo "SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-}"; echo "SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-}"; echo "SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-}"')
eval "${SLINGSHOT_ENV}"
export SLINGSHOT_VNIS SLINGSHOT_SVC_IDS SLINGSHOT_DEVICES
echo "Captured: SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-unset}"
echo "Captured: SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-unset}"
echo "Captured: SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-unset}"

if [[ -z "${SLINGSHOT_VNIS:-}" ]]; then
    echo "WARNING: SLINGSHOT_VNIS is empty — cross-node NCCL may fail"
fi

# Stop any lingering Ray processes
ray stop --force 2>/dev/null || true

# Start Ray head locally (sbatch script runs on head node — no srun needed)
echo "Starting Ray head locally on $(hostname)..."
ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port=${RAY_PORT} \
    --num-cpus=$((SLURM_CPUS_PER_TASK - N_GPUS_PER_NODE)) \
    --num-gpus=${N_GPUS_PER_NODE}
sleep 10

ray status
echo "Head node ready"

# Start Ray workers on remaining nodes via srun
# Pass all Slingshot/NCCL env vars explicitly inside bash -c
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
            export NCCL_DEBUG=INFO
            export NCCL_DEBUG_SUBSYS=INIT,NET
            ray start --address='${HEAD_IP}:${RAY_PORT}' \
                --num-cpus=$((SLURM_CPUS_PER_TASK - N_GPUS_PER_NODE)) \
                --num-gpus=${N_GPUS_PER_NODE}
            sleep infinity
        " &
    sleep 10
done

# Wait for workers to register (Ray worker connection takes ~20s)
sleep 40

# Set RAY_ADDRESS so ray status and train.py connect to existing cluster
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

# Verify Ray cluster (run directly — srun slots are consumed by Ray head/workers)
echo "=== Ray cluster status ==="
ray status
echo "=========================="

# ============================================================
# Run minimal training: 2 steps, small batch, G=4
# ============================================================
echo "=== Starting training (2 steps) ==="

$PYTHON "${REPO_ROOT}/experiment4_llm/src/train.py" \
  +data.seed=42 \
  algorithm.adv_estimator=grpo \
  algorithm.filter_overlong_responses=True \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${REPO_ROOT}/data/deepcoder_train.parquet" \
  data.val_files="${REPO_ROOT}/data/deepcoder_train.parquet" \
  data.train_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.grad_clip=0.3 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  reward_model.enable=False \
  reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment4_llm/src/reward_function.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=2 \
  trainer.total_epochs=1 \
  trainer.project_name=test-multinode \
  trainer.experiment_name=test_2node_8gpu \
  "trainer.logger=['console']" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=${NUM_NODES} \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.val_before_train=False

TRAIN_EXIT=$?

echo ""
echo "=== Training exit code: ${TRAIN_EXIT} ==="

# Check checkpoint
if [ -d "${OUTPUT_DIR}/global_step_1" ]; then
    echo "=== CHECKPOINT SAVED ==="
    ls -la "${OUTPUT_DIR}/global_step_1/actor/"
    echo ""
    echo "World size in checkpoint files:"
    ls "${OUTPUT_DIR}/global_step_1/actor/" | grep model_world_size || echo "NO MODEL FILES"
else
    echo "=== NO CHECKPOINT FOUND ==="
    ls -la "${OUTPUT_DIR}/" 2>/dev/null || echo "Output dir empty"
fi

echo ""
echo "=== Cleaning up test artifacts ==="
rm -rf "${OUTPUT_DIR}"
echo "Deleted ${OUTPUT_DIR}"

echo ""
echo "=== Test complete ==="

# Cleanup Ray
ray stop --force 2>/dev/null || true
