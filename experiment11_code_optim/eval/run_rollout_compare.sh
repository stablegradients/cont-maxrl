#!/usr/bin/env bash
#SBATCH --job-name=rollout_cmp
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# ============================================================================
# Generate + gem5-score N rollouts for a few hand-picked highest-headroom test
# problems, for grpo@500 / gos@300 / rloo@500 in ONE 1-GPU job (each arm loads,
# generates, scores, frees the GPU, next arm). Same gem5 reward config as the
# n=1024 best_reward@k eval so speedups are directly comparable.
#   sbatch --account=<acct>-dtai-gh run_rollout_compare.sh
#   override: PROBLEMS=... NROLL=32 TEMP=1.0
# ============================================================================
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -uo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
PY=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="${REPO}:${PYTHONPATH:-}"
[ -f "${REPO}/.env" ] && { set -a; source "${REPO}/.env"; set +a; }

STORAGE=/work/nvme/bgtv/sramasubramanian/cont-maxrl
OUT_DIR="${STORAGE}/rollout_compare"
VAL=/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet/pie_gem5_test.parquet
mkdir -p "${OUT_DIR}"

# the 3 highest src/tgt-ratio distinct test problems (40.0x / 26.7x / 14.8x)
PROBLEMS="${PROBLEMS:-p03275:s242733854,p03161:s106426206,p03730:s784784078}"
NROLL="${NROLL:-32}"; TEMP="${TEMP:-1.0}"; TOPP="${TOPP:-1.0}"; MAXTOK="${MAXTOK:-4096}"

export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET || true; unset NCCL_NET_PLUGIN || true; export NCCL_PLUGIN_P2P=0
export VLLM_USE_V1=0

# ---- gem5 reward: speedup ratio on the single BIGGEST usable case (K=1), the
#      SAME config the n=1024 best_reward@k eval used; correctness on ALL usable ----
export PIE_MODE=1 PIE_REWARD_IMPL=gem5 PIE_GEM5_REWARD_KIND=ratio
export PIE_GEM5_REWARD_K=1 PIE_GEM5_CASE_SELECT=biggest
export PIE_GEM5_REWARD_SAMPLE=group PIE_GEM5_BUDGET_MULT=3.0
export PIE_GEM5_REWARD_WORKERS="${PIE_GEM5_REWARD_WORKERS:-64}"
export PIE_NAT_TIMEOUT="${PIE_NAT_TIMEOUT:-10}" PIE_GEM5_REWARD_DEADLINE=6000
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
export TMPDIR="/tmp/${USER}_rollcmp_${SLURM_JOB_ID:-$$}"; export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"
mkdir -p "${TMPDIR}" "${PIE_COMPILE_SCRATCH}"
export HF_HOME="${REPO}/.cache/huggingface"; export VLLM_CACHE_ROOT="${REPO}/.cache/vllm"

run_arm () {
  local arm="$1" step="$2"
  # ONLY_ARM=<arm> -> run just that ckpt (for sharding one job per ckpt across nodes)
  [ -n "${ONLY_ARM:-}" ] && [ "${ONLY_ARM}" != "${arm}" ] && { echo "skip ${arm} (ONLY_ARM=${ONLY_ARM})"; return; }
  local hf="${STORAGE}/results/qwen3_1.7b_pie_gem5_fb_${arm}_g16_bs64_s0/actor_ckpts/step_${step}/huggingface"
  [ -f "${hf}/config.json" ] || { echo "SKIP ${arm}: no ${hf}"; return; }
  echo "================ ${arm} step ${step} ================"
  $PY -m experiment11_code_optim.eval.compare_rollouts \
    --arm "${arm}" --step "${step}" --hf_dir "${hf}" --val_parquet "${VAL}" \
    --problems "${PROBLEMS}" --n "${NROLL}" --temperature "${TEMP}" --top_p "${TOPP}" \
    --max_tokens "${MAXTOK}" --tensor_parallel_size 1 --out_dir "${OUT_DIR}"
}

echo "=== rollout compare | problems=${PROBLEMS} | n=${NROLL} temp=${TEMP} ==="
run_arm grpo 500
run_arm gos  300
run_arm rloo 500
echo "=== all arms done -> ${OUT_DIR} ==="
echo "aggregate: $PY -m experiment11_code_optim.eval.compare_rollouts_aggregate --out_dir ${OUT_DIR}"
