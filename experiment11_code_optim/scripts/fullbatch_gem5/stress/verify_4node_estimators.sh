#!/usr/bin/env bash
#SBATCH --job-name=pie_v4n
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=03:30:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# ============================================================================
# MULTI-NODE FEASIBILITY + THROUGHPUT TEST for the PIE-4B gem5 RL run.
# 2 nodes x 4 GH200 = 8 GPUs. Reuses the PROVEN ray-on-slurm bootstrap from the
# 8B/32B DeepCoder runs (Slingshot/cxi NCCL). Runs the REAL PIE-4B config
# (B=64, G=16, P3, real gem5 reward, rmpad+dynamic_bsz) for 3 steps (2 warmup +
# 1 steady) and prints the per-step timing -> compare vs single-node C5 to get
# the multi-node speedup. Proves: (a) Ray cluster forms across nodes, (b) verl
# trains across 8 GPUs, (c) the gem5 reward works in a multi-node Ray cluster.
#   sbatch --account=<acct>-dtai-gh stress_4b_multinode.sh
# ============================================================================
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -uo pipefail
export PYTHONNOUSERSITE=1

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/work/nvme/bgtv/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
P3_ROOT="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet_p3"
[ -f "${REPO_ROOT}/.env" ] && { set -a; source "${REPO_ROOT}/.env"; set +a; }   # secrets from .env (never hardcode)
: "${WANDB_API_KEY:?WANDB_API_KEY must be set via .env}"

# ---- Conda CUDA + Slingshot/cxi NCCL (verbatim from the proven 8B 4-node run) ----
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib:/opt/cray/libfabric/2.3.1/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_NET_PLUGIN="/sw/user/nccl/aws-ofi-nccl-1.18.0-lf2.3.1-cu12.9/lib/libnccl-net.so"
export NCCL_SOCKET_IFNAME=hsn GLOO_SOCKET_IFNAME=hsn0 NCCL_CROSS_NIC=1 NCCL_IB_DISABLE=1
export NCCL_PLUGIN_P2P=0 NCCL_NET_GDR_LEVEL=PHB NCCL_DMABUF_ENABLE=0
export FI_PROVIDER=cxi NCCL_NET_OFI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0
export FI_MR_CACHE_MONITOR=userfaultfd NCCL_DEBUG=WARN

export HF_HOME="${REPO_ROOT}/.cache/huggingface"; export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"; export VLLM_USE_V1=0
export RAY_OBJECT_STORE_MEMORY=32000000000
export RAY_TMPDIR="/tmp/ray_${USER}_mn"; export TMPDIR="/tmp/${USER}_mn_${SLURM_JOB_ID}"
export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"; mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${PIE_COMPILE_SCRATCH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-stablegradients}"

# ---- PIE-4B + gem5 reward env ----
export PIE_MODE=1 PIE_REWARD_IMPL=gem5 PIE_GEM5_REWARD_KIND=ratio PIE_GEM5_REWARD_K=1 \
       PIE_GEM5_CASE_SELECT=biggest PIE_GEM5_BUDGET_MULT=3.0 PIE_GEM5_REWARD_WORKERS=128 \
       PIE_NAT_TIMEOUT=5 PIE_GEM5_REWARD_NATIVE_BUDGET=30 PIE_GEM5_REWARD_DEADLINE=1200 \
       PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases

OUT="${STORAGE_ROOT}/stress_4b/mn_${SLURM_JOB_ID}"; mkdir -p "${OUT}"

# ============================================================
# Multi-node Ray cluster bootstrap (proven pattern, 8B/32B runs)
# ============================================================
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE="${NODES[0]}"; NUM_NODES=${#NODES[@]}; N_GPUS_PER_NODE=4
HEAD_IP=$(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -1); RAY_PORT=6379
echo "=== PIE-4B MULTINODE TEST | nodes=${NUM_NODES} gpus/node=${N_GPUS_PER_NODE} total=$((NUM_NODES*N_GPUS_PER_NODE)) GPUs ==="
echo "Head: ${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) | nodes: ${NODES[*]} | out=${OUT}"
[ -z "${HEAD_IP}" ] && { echo "FATAL: no head IPv4"; exit 1; }

# Slingshot VNI credentials (needed for cross-node NCCL on cxi)
SLINGSHOT_ENV=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  bash -c 'echo "SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-}"; echo "SLINGSHOT_SVC_IDS=${SLINGSHOT_SVC_IDS:-}"; echo "SLINGSHOT_DEVICES=${SLINGSHOT_DEVICES:-}"')
eval "${SLINGSHOT_ENV}"; export SLINGSHOT_VNIS SLINGSHOT_SVC_IDS SLINGSHOT_DEVICES
echo "SLINGSHOT_VNIS=${SLINGSHOT_VNIS:-unset}"
[ -z "${SLINGSHOT_VNIS:-}" ] && echo "WARNING: SLINGSHOT_VNIS empty — cross-node NCCL may fail"

ray stop --force 2>/dev/null || true
ray start --head --node-ip-address="${HEAD_IP}" --port=${RAY_PORT} \
  --num-cpus=$((SLURM_CPUS_PER_TASK - N_GPUS_PER_NODE)) --num-gpus=${N_GPUS_PER_NODE}
sleep 10; ray status; echo "head ready"

for ((i=1; i<NUM_NODES; i++)); do
  WORKER_NODE="${NODES[$i]}"; echo "starting Ray worker on ${WORKER_NODE}..."
  srun --nodes=1 --ntasks=1 -w "${WORKER_NODE}" --export=ALL bash -c "
      export LD_LIBRARY_PATH='${LD_LIBRARY_PATH}' NCCL_NET_PLUGIN='${NCCL_NET_PLUGIN}'
      export NCCL_SOCKET_IFNAME=hsn GLOO_SOCKET_IFNAME=hsn0 NCCL_CROSS_NIC=1 NCCL_IB_DISABLE=1
      export NCCL_PLUGIN_P2P=0 NCCL_NET_GDR_LEVEL=PHB NCCL_DMABUF_ENABLE=0 FI_PROVIDER=cxi NCCL_NET_OFI_PROVIDER=cxi
      export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_DISABLE_HOST_REGISTER=1
      export FI_CXI_RX_MATCH_MODE=hybrid FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0
      export FI_MR_CACHE_MONITOR=userfaultfd NCCL_DEBUG=WARN
      export SLINGSHOT_VNIS='${SLINGSHOT_VNIS}' SLINGSHOT_SVC_IDS='${SLINGSHOT_SVC_IDS}' SLINGSHOT_DEVICES='${SLINGSHOT_DEVICES}'
      ray start --address='${HEAD_IP}:${RAY_PORT}' --num-cpus=$((SLURM_CPUS_PER_TASK - N_GPUS_PER_NODE)) --num-gpus=${N_GPUS_PER_NODE}
      sleep infinity
  " &
  sleep 10
done
sleep 40
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
echo "=== Ray cluster status (expect $((NUM_NODES*N_GPUS_PER_NODE)) GPUs) ==="; ray status

# ============================================================
# PIE-4B training: 3 steps (2 warmup + 1 steady), B=64, P3, real gem5 reward,
# rmpad + dynamic_bsz (the fast batching path). nnodes drives the resource pool.
# ============================================================
set +e
NGPU=$((NUM_NODES*N_GPUS_PER_NODE))
echo "name|status|grad_norm_per_step|step_steady_s|gen_s|update_s|resp_max|mean_spd|max_spd" > "${OUT}/estimators.psv"
# Loop the 3 estimators against the SAME Ray cluster, SAME data (seed 0) so the
# only difference is the estimator -> grad_norms are directly comparable.
for ADV in gos rloo grpo; do
  echo ""; echo "################ ESTIMATOR=${ADV} | ${NGPU} GPUs (${NUM_NODES} nodes) ################"
  LOG="${OUT}/train_${ADV}.log"; t0=$(date +%s)
  $PYTHON "${REPO_ROOT}/experiment11_code_optim/train.py" \
    +data.seed=0 \
    algorithm.adv_estimator=${ADV} algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${P3_ROOT}/pie_gem5_train.parquet" data.val_files="${P3_ROOT}/pie_gem5_val_200.parquet" \
    data.train_batch_size=64 data.max_prompt_length=2560 data.max_response_length=16384 \
    data.shuffle=True data.filter_overlong_prompts=True data.enable_thinking=False data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18944 \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio=100 actor_rollout_ref.actor.clip_ratio_high=100 \
    actor_rollout_ref.actor.grad_clip=2.0 actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18944 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.max_model_len=18944 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18944 \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.enable=False reward_model.launch_reward_fn_async=True \
    custom_reward_function.path="${REPO_ROOT}/experiment11_code_optim/reward/gem5_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.total_training_steps=3 trainer.total_epochs=1 \
    trainer.project_name=cont-maxrl-pie-v4n trainer.experiment_name="pie_v4n_${ADV}_${SLURM_JOB_ID}" \
    "trainer.logger=['console']" trainer.default_local_dir="${OUT}/verl_${ADV}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=10000 trainer.test_freq=-1 trainer.resume_mode=disable trainer.val_before_train=False \
    trainer.max_actor_ckpt_to_keep=1 "insitu_eval.eval_steps=[]" \
    insitu_eval.data_file="${P3_ROOT}/pie_gem5_val_200.parquet" insitu_eval.output_dir="${OUT}/insitu_${ADV}" insitu_eval.label=disabled \
    ray_init.num_cpus=null \
    2>&1 | tee "${LOG}"
  rc=$?; t1=$(date +%s)
  gnorms=$(grep -aoE "actor/grad_norm:[0-9.]+" "${LOG}" | cut -d: -f2 | paste -sd',' -)
  stp=$(grep -aoE "timing_s/step:[0-9.]+" "${LOG}" | tail -1 | cut -d: -f2)
  gen=$(grep -aoE "timing_s/gen:[0-9.]+" "${LOG}" | tail -1 | cut -d: -f2)
  upd=$(grep -aoE "timing_s/update_actor:[0-9.]+" "${LOG}" | tail -1 | cut -d: -f2)
  rmax=$(grep -aoE "response_length/max:[0-9.]+" "${LOG}" | tail -1 | cut -d: -f2)
  msp=$(grep -aoE "critic/score/mean_speedup:[0-9.]+" "${LOG}" | tail -1 | cut -d: -f2)
  xsp=$(grep -aoE "critic/score/max_speedup:[0-9.]+" "${LOG}" | tail -1 | cut -d: -f2)
  oom="ok"; grep -qaiE "out of memory|OutOfMemory" "${LOG}" && oom="OOM"; [ "${rc}" -ne 0 ] && [ "${oom}" = "ok" ] && oom="FAIL(${rc})"
  printf "%s|%s|%s|%s|%s|%s|%s|%s|%s\n" "${ADV}" "${oom}" "${gnorms:-?}" "${stp:-?}" "${gen:-?}" "${upd:-?}" "${rmax:-?}" "${msp:-?}" "${xsp:-?}" >> "${OUT}/estimators.psv"
  echo "[${ADV}] rc=${rc} oom=${oom} grad_norm=[${gnorms}] step_steady=${stp}s gen=${gen}s update=${upd}s mean_spd=${msp} max_spd=${xsp} wall=$((t1-t0))s"
done
ray stop --force 2>/dev/null || true
echo ""; echo "============= 4-NODE ESTIMATOR + SCALING VERIFICATION (${NGPU} GPUs) ============="
column -t -s'|' "${OUT}/estimators.psv" 2>/dev/null || cat "${OUT}/estimators.psv"
echo "SCALING: 2-node(8 GPU) steady step was ~933s; 4-node(16 GPU) ideal ~466s. Actual above."
echo "GRAD-NORM: post-xG-removal, gos/rloo/grpo should be in the same ballpark (clip=2.0 uniform)."
echo "results: ${OUT}"
