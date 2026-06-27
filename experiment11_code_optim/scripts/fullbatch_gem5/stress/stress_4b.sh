#!/usr/bin/env bash
#SBATCH --job-name=pie_stress_4b
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:40:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# ============================================================================
# 4B-Instruct-2507 MEMORY/THROUGHPUT STRESS TEST for the 16384-token RL config,
# with a 1 Hz embedded resource sampler (per-GPU mem+util, CPU%, RAM).
#
# Real 4B rollouts are short (~2.5k), so a normal step never hits the 16k peak.
# C0-C4 FORCE the worst case with rollout.ignore_eos=True -> EVERY rollout
# generates exactly 16384 tokens. The actor peak is per-micro-batch
# ([micro_bs,18944]) hence B-INDEPENDENT, so small B exercises the true peak.
# A trivial reward isolates the GPU on those.
# C5_real runs the ACTUAL workload (B=64, real gem5 reward, real generations,
# 2 steps) so we also capture realistic CPU/RAM under the 256-worker reward and
# the real per-step throughput (-> walltime estimate).
#
# Sweeps the batching modes the launcher leaves at verl defaults:
#   use_remove_padding (False=pad to 18944 ; True=flash-varlen pack)
#   use_dynamic_bsz    (False=static micro_bs ; True=token-budget packing)
#   micro_bs / ppo_max_token_len_per_gpu ; gpu_memory_utilization
#
#   sbatch --account=<acct>-dtai-gh stress_4b.sh
# ============================================================================
# NOTE: 'set -u' is enabled AFTER conda activate below. conda's gcc activate.d
# (activate-gcc_linux-aarch64.sh) references SYS_SYSROOT unbound and dies under
# set -u — which killed an earlier spray at 7s. The real launcher does the same.
set -o pipefail
REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
STORAGE_ROOT="/work/nvme/bgtv/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
HERE="${REPO_ROOT}/experiment11_code_optim/scripts/fullbatch_gem5/stress"
P3_ROOT="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet_p3"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
RESULTS="${STORAGE_ROOT}/stress_4b/run_${SLURM_JOB_ID:-manual}"
CSV="${RESULTS}/resources.csv"
mkdir -p "${RESULTS}" "${REPO_ROOT}/slurm_logs"

module load python/miniforge3_pytorch/2.10.0 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -uo pipefail   # safe now that conda's non--u-clean gcc activate.d has already run
export PYTHONNOUSERSITE=1
[ -f "${REPO_ROOT}/.env" ] && { set -a; source "${REPO_ROOT}/.env"; set +a; }
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET || true; unset NCCL_NET_PLUGIN || true; export NCCL_PLUGIN_P2P=0
export HF_HOME="${REPO_ROOT}/.cache/huggingface"; export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"; export VLLM_USE_V1=0
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PIE_MODE=1
export RAY_OBJECT_STORE_MEMORY=32000000000
export TMPDIR="/tmp/${USER}_stress_${SLURM_JOB_ID:-$$}"; mkdir -p "${TMPDIR}"
export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"; mkdir -p "${PIE_COMPILE_SCRATCH}"

# ---------------------------------------------------------------------------
# 1 Hz resource sampler: per-GPU mem/util + node CPU% (from /proc/stat delta) +
# RAM. Tagged with the CURRENT_CONFIG marker the main loop sets. Card-level
# nvidia-smi mem.used is the true OOM number (actor torch + vLLM KV combined).
# ---------------------------------------------------------------------------
echo "ts,phase,cpu_pct,ram_used_mib,ram_total_mib,gpu,mem_used_mib,mem_total_mib,gpu_util,mem_io_util" > "${CSV}"
echo "-" > "${RESULTS}/CURRENT_CONFIG"
sampler() {
  local a b c d e f g h pidle ptot idle tot didle dtot cpu ru rt ts phase
  read -r _ a b c d e f g h _ < /proc/stat; pidle=$((d+e)); ptot=$((a+b+c+d+e+f+g+h))
  while [ ! -f "${RESULTS}/STOP_SAMPLER" ]; do
    sleep 1
    read -r _ a b c d e f g h _ < /proc/stat; idle=$((d+e)); tot=$((a+b+c+d+e+f+g+h))
    didle=$((idle-pidle)); dtot=$((tot-ptot)); pidle=$idle; ptot=$tot
    cpu=$(awk -v dt=$dtot -v di=$didle 'BEGIN{ if(dt>0) printf "%.1f",100*(dt-di)/dt; else printf "0.0" }')  # this awk rejects the ?: ternary
    read -r ru rt < <(free -m | awk '/^Mem:/{print $3, $2}')
    ts=$(date +%s); phase=$(cat "${RESULTS}/CURRENT_CONFIG" 2>/dev/null || echo "-")
    while IFS=',' read -r idx mu mt gu mio; do
      echo "${ts},${phase},${cpu},${ru},${rt},${idx},${mu},${mt},${gu},${mio}" >> "${CSV}"
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits | tr -d ' ')
  done
}
sampler & SAMPLER_PID=$!
trap 'touch "${RESULTS}/STOP_SAMPLER"; kill ${SAMPLER_PID} 2>/dev/null' EXIT

# config rows: name|rmpad|dyn|micro|tok|util|B|ignore_eos|reward
CONFIGS=(
  "C0_baseline_pad_micro1|False|False|1|0|0.5|4|True|trivial"
  "C1_rmpad_micro1|True|False|1|0|0.5|4|True|trivial"
  "C2_rmpad_micro2|True|False|2|0|0.5|4|True|trivial"
  "C3_rmpad_dyn_18944|True|True|0|18944|0.5|4|True|trivial"
  "C4_rmpad_dyn_36864_u060|True|True|0|36864|0.6|4|True|trivial"
  "C5_real_b64_gem5|True|True|0|18944|0.5|64|False|gem5"
)

run_one() {
  local name="$1" rmpad="$2" dyn="$3" micro="$4" tok="$5" util="$6" B="$7" ieos="$8" rew="$9"
  local odir="${RESULTS}/${name}"; mkdir -p "${odir}"; local log="${odir}/run.log"
  echo "${name}" > "${RESULTS}/CURRENT_CONFIG"
  echo ""; echo "######## ${name} (rmpad=${rmpad} dyn=${dyn} micro=${micro} tok=${tok} util=${util} B=${B} ignore_eos=${ieos} reward=${rew}) ########"
  local BATCH_ARGS
  if [ "${dyn}" = "True" ]; then
    BATCH_ARGS="actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${tok} actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${tok} actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${tok}"
  else
    BATCH_ARGS="actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro} actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro} actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro}"
  fi
  # reward: trivial (isolate GPU) or the real gem5 engine (realistic CPU/RAM)
  local REWARD_PATH REWARD_ENV=""
  if [ "${rew}" = "gem5" ]; then
    REWARD_PATH="${REPO_ROOT}/experiment11_code_optim/reward/gem5_reward.py"
    export PIE_REWARD_IMPL=gem5 PIE_GEM5_REWARD_KIND=ratio PIE_GEM5_REWARD_K=1 \
           PIE_GEM5_CASE_SELECT=biggest PIE_GEM5_BUDGET_MULT=3.0 \
           PIE_GEM5_REWARD_WORKERS=256 PIE_NAT_TIMEOUT=5 \
           PIE_GEM5_REWARD_NATIVE_BUDGET=30 PIE_GEM5_REWARD_DEADLINE=1200 \
           PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
    local STEPS=3   # 2 warmup + 1 steady (kernel autotune + allocator settle before measuring)
  else
    REWARD_PATH="${HERE}/stress_reward.py"; unset PIE_REWARD_IMPL || true
    local STEPS=3   # 2 warmup + 1 steady
  fi
  local t0=$(date +%s)
  timeout 2700 $PYTHON "${REPO_ROOT}/experiment11_code_optim/train.py" \
    algorithm.adv_estimator=gos \
    algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${P3_ROOT}/pie_gem5_train.parquet" \
    data.val_files="${P3_ROOT}/pie_gem5_val_200.parquet" \
    data.train_batch_size=${B} data.max_prompt_length=2560 data.max_response_length=16384 \
    data.shuffle=True data.filter_overlong_prompts=True data.enable_thinking=False \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=${rmpad} \
    actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${B} actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio=100 actor_rollout_ref.actor.clip_ratio_high=100 \
    actor_rollout_ref.actor.grad_clip=2.0 actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.use_torch_compile=False \
    ${BATCH_ARGS} \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${util} actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=18944 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.ignore_eos=${ieos} \
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.enable=False reward_model.launch_reward_fn_async=True \
    custom_reward_function.path="${REWARD_PATH}" custom_reward_function.name=compute_score \
    trainer.total_training_steps=${STEPS} trainer.total_epochs=1 \
    trainer.project_name=cont-maxrl-pie-stress trainer.experiment_name="${name}" \
    "trainer.logger=['console']" trainer.default_local_dir="${odir}/verl" \
    trainer.n_gpus_per_node=4 trainer.nnodes=1 \
    trainer.save_freq=10000 trainer.test_freq=-1 trainer.resume_mode=disable \
    trainer.val_before_train=False trainer.max_actor_ckpt_to_keep=1 \
    "insitu_eval.eval_steps=[]" insitu_eval.data_file="${P3_ROOT}/pie_gem5_val_200.parquet" \
    insitu_eval.output_dir="${odir}/insitu" insitu_eval.label=disabled \
    ray_init.num_cpus=256 \
    > "${log}" 2>&1
  local rc=$?; local t1=$(date +%s)
  echo "-" > "${RESULTS}/CURRENT_CONFIG"
  ray stop --force >/dev/null 2>&1 || true
  local memres memall gen olp upd stp stp1 respmax oom="-"
  # memory = MAX across all steps (robust whether verl resets peak per-step or not) -> OOM safety
  memres=$(grep -aoE "perf/max_memory_reserved_gb:[0-9.]+" "${log}" | cut -d: -f2 | sort -g | tail -1)
  memall=$(grep -aoE "perf/max_memory_allocated_gb:[0-9.]+" "${log}" | cut -d: -f2 | sort -g | tail -1)
  # timing = LAST (steady-state) step after warmup; stp1 = first step (warmup) for reference
  gen=$(grep -aoE "timing_s/gen:[0-9.]+" "${log}" | tail -1 | cut -d: -f2)
  olp=$(grep -aoE "timing_s/old_log_prob:[0-9.]+" "${log}" | tail -1 | cut -d: -f2)
  upd=$(grep -aoE "timing_s/update_actor:[0-9.]+" "${log}" | tail -1 | cut -d: -f2)
  stp=$(grep -aoE "timing_s/step:[0-9.]+" "${log}" | tail -1 | cut -d: -f2)
  stp1=$(grep -aoE "timing_s/step:[0-9.]+" "${log}" | head -1 | cut -d: -f2)
  respmax=$(grep -aoE "response_length/max:[0-9.]+" "${log}" | tail -1 | cut -d: -f2)
  grep -qaiE "out of memory|CUDA out of memory|OutOfMemory" "${log}" && oom="OOM"
  [ "${rc}" -ne 0 ] && [ "${oom}" = "-" ] && oom="FAIL(rc=${rc})"
  printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" \
    "${name}" "${oom:-ok}" "${memres:-?}" "${memall:-?}" "${respmax:-?}" \
    "${gen:-?}" "${olp:-?}" "${upd:-?}" "${stp1:-?}" "${stp:-?}" "$((t1-t0))" \
    "rmpad=${rmpad},dyn=${dyn},micro=${micro},tok=${tok},util=${util},B=${B},ieos=${ieos},rew=${rew}" \
    >> "${RESULTS}/summary.psv"
  echo "[${name}] rc=${rc} oom=${oom:-ok} torch_reserved=${memres:-?}GB resp_max=${respmax:-?} gen=${gen:-?}s update=${upd:-?}s step_warmup=${stp1:-?}s step_steady=${stp:-?}s wall=$((t1-t0))s"
  sleep 20
}

echo "=== 4B STRESS+MONITOR | results=${RESULTS} | sampler pid=${SAMPLER_PID} (1 Hz) ==="
echo "name|status|torch_reserved_gb|torch_alloc_gb|resp_max|gen_s|oldlogp_s|update_s|step_warmup_s|step_steady_s|wall_s|config" > "${RESULTS}/summary.psv"
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r name rmpad dyn micro tok util B ieos rew <<< "${row}"
  run_one "${name}" "${rmpad}" "${dyn}" "${micro}" "${tok}" "${util}" "${B}" "${ieos}" "${rew}"
done

touch "${RESULTS}/STOP_SAMPLER"; sleep 2; kill ${SAMPLER_PID} 2>/dev/null || true
echo ""; echo "============= STRESS SUMMARY (verl torch metrics) ============="
column -t -s'|' "${RESULTS}/summary.psv" 2>/dev/null || cat "${RESULTS}/summary.psv"
echo ""; echo "============= CARD-LEVEL PEAKS (nvidia-smi, per config) ============="
# peak card mem + max gpu util + peak RAM + max CPU% per phase, across all GPUs
$PYTHON - "${CSV}" <<'PY' 2>/dev/null || echo "(python summary skipped)"
import csv,sys,collections
rows=list(csv.DictReader(open(sys.argv[1])))
agg=collections.defaultdict(lambda:{"mem":0,"gu":0,"cpu":0.0,"ram":0,"ramtot":0,"memtot":0,"n":0})
for r in rows:
    p=r["phase"]
    if p=="-": continue
    a=agg[p]
    a["mem"]=max(a["mem"],int(r["mem_used_mib"])); a["memtot"]=int(r["mem_total_mib"])
    a["gu"]=max(a["gu"],int(r["gpu_util"]))
    a["cpu"]=max(a["cpu"],float(r["cpu_pct"]))
    a["ram"]=max(a["ram"],int(r["ram_used_mib"])); a["ramtot"]=int(r["ram_total_mib"]); a["n"]+=1
print(f"{'phase':28s} {'peak_gpu_mem':>14s} {'max_gpu_util':>12s} {'max_cpu%':>8s} {'peak_ram':>16s}")
for p,a in agg.items():
    print(f"{p:28s} {a['mem']/1024:6.1f}/{a['memtot']/1024:.0f}GB  {a['gu']:>10d}%  {a['cpu']:>7.1f}%  {a['ram']/1024:6.1f}/{a['ramtot']/1024:.0f}GB")
PY
echo "==============================================================="
echo "results dir: ${RESULTS}  (resources.csv = full 1 Hz trace)"
