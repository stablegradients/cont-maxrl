#!/usr/bin/env bash
#SBATCH --job-name=besteval
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=160G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# ============================================================================
# One 1-GPU cell of the n=1024 best_reward@k test eval: generates + gem5-scores
# one contiguous prompt-shard for one arm's pinned snapshot. TP=1 so cells
# backfill partial nodes without displacing the 4-GPU training jobs. Idempotent.
#   Usage: sbatch --account=<acct>-dtai-gh run_besteval_cell.sh <arm> <step> <shard> <n_shards>
# ============================================================================
ARM="${1:?arm}"; STEP="${2:?step}"; SHARD="${3:?shard}"; N_SHARDS="${4:?n_shards}"
NCOMP="${NCOMP:-1024}"; TEMP="${TEMP:-1.0}"; TOPP="${TOPP:-1.0}"; MAXTOK="${MAXTOK:-4096}"

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -euo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
PY=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="${REPO}:${PYTHONPATH:-}"
[ -f "${REPO}/.env" ] && { set -a; source "${REPO}/.env"; set +a; }

STORAGE=/work/nvme/bgtv/sramasubramanian/cont-maxrl
EVAL_ROOT="${STORAGE}/besteval"
# Model-aware (default = 1.7B, back-compat). The 4B-Instruct-2507 P3 eval sets
# PIE_MODEL_TAG / PIE_BASE_MODEL_HF / BESTEVAL_TEST_PARQUET so the eval model +
# PROMPT match training (the prompt is read FROM the parquet, so the test parquet
# MUST be the same variant — P3 — as training).
MODEL_TAG="${PIE_MODEL_TAG:-qwen3_1.7b}"
BASE_MODEL_HF="${PIE_BASE_MODEL_HF:-/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
if [ "${ARM}" = "base" ]; then
  HF_DIR="${BASE_MODEL_HF}"                       # untrained init (the RL base) — local HF cache
  OUT_DIR="${EVAL_ROOT}/${MODEL_TAG}_base_step0"
else
  HF_DIR="${STORAGE}/results/${MODEL_TAG}_pie_gem5_fb_${ARM}_g16_bs64_s0/actor_ckpts/step_${STEP}/huggingface"
  OUT_DIR="${EVAL_ROOT}/${MODEL_TAG}_${ARM}_step${STEP}"
fi
[ -f "${HF_DIR}/config.json" ] || { echo "FATAL: no model ${HF_DIR}"; exit 2; }
VAL="${BESTEVAL_TEST_PARQUET:-/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet/pie_gem5_test.parquet}"

export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET || true; unset NCCL_NET_PLUGIN || true; export NCCL_PLUGIN_P2P=0
export VLLM_USE_V1=0

# ---- eval gem5 reward: speedup ratio on the single BIGGEST (most discriminating)
#      usable case (K=1, case_select=biggest), correctness gate on ALL usable ----
export PIE_MODE=1 PIE_REWARD_IMPL=gem5 PIE_GEM5_REWARD_KIND=ratio
export PIE_GEM5_REWARD_K="${PIE_GEM5_REWARD_K:-1}"
export PIE_GEM5_CASE_SELECT="${PIE_GEM5_CASE_SELECT:-biggest}"
export PIE_GEM5_REWARD_SAMPLE=group PIE_GEM5_BUDGET_MULT=3.0
export PIE_GEM5_REWARD_WORKERS="${PIE_GEM5_REWARD_WORKERS:-64}"
export PIE_NAT_TIMEOUT="${PIE_NAT_TIMEOUT:-10}" PIE_GEM5_REWARD_DEADLINE=6000
export PIE_GEM5_REWARD_NATIVE_BUDGET="${PIE_GEM5_REWARD_NATIVE_BUDGET:-120}"   # gate cutoff; 4B eval sets 30 to match training's admission set
export BESTEVAL_DUMP_GENS="${BESTEVAL_DUMP_GENS:-5}"   # save top-5 rollouts by speedup + prompt per problem
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
export TMPDIR="/tmp/${USER}_besteval_${SLURM_JOB_ID:-$$}"; export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"
mkdir -p "${TMPDIR}" "${PIE_COMPILE_SCRATCH}" "${OUT_DIR}"
export HF_HOME="${REPO}/.cache/huggingface"; export VLLM_CACHE_ROOT="${REPO}/.cache/vllm"

echo "=== besteval cell | ${ARM} step${STEP} | shard ${SHARD}/${N_SHARDS} | n=${NCOMP} temp=${TEMP} K=${PIE_GEM5_REWARD_K} case=${PIE_GEM5_CASE_SELECT} ==="
$PY -m experiment11_code_optim.eval.besteval_shard \
  --hf_dir "${HF_DIR}" --val_parquet "${VAL}" \
  --shard "${SHARD}" --n_shards "${N_SHARDS}" \
  --n_completions "${NCOMP}" --temperature "${TEMP}" --top_p "${TOPP}" \
  --max_tokens "${MAXTOK}" --tensor_parallel_size 1 --out_dir "${OUT_DIR}"
echo "=== cell done: ${ARM} step${STEP} shard ${SHARD}/${N_SHARDS} ==="
