#!/usr/bin/env bash
#SBATCH --job-name=prerl
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=5:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# ============================================================================
# Pre-RL analysis: generate + gem5-score 32 rollouts for the frozen 32-problem
# train set, per (model, prompt, think) config. Shard ONE config per node with
# ONLY_CONFIG=<name>; aggregate with prerl_analyze.py.
#   sbatch --account=<acct>-dtai-gh -J prerl_inst2507_P1 \
#          --export=ALL,ONLY_CONFIG=inst2507_P1 run_prerl.sh
#   smoke (run directly on a GPU node): SMOKE=1 ONLY_CONFIG=inst2507_P1 bash run_prerl.sh
# ============================================================================
module load python/miniforge3_pytorch/2.10.0 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl 2>/dev/null || true
set -uo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
PY=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="${REPO}:${PYTHONPATH:-}"
[ -f "${REPO}/.env" ] && { set -a; source "${REPO}/.env"; set +a; }

OUT_DIR="${OUT_DIR:-${REPO}/experiment11_code_optim/results/prerl_analysis}"
PROBLEMS_JSON="${OUT_DIR}/problems32.json"
mkdir -p "${OUT_DIR}"

export HF_HOME=/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface
export VLLM_CACHE_ROOT="${REPO}/.cache/vllm"
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET || true; unset NCCL_NET_PLUGIN || true; export NCCL_PLUGIN_P2P=0
export VLLM_USE_V1=0

# gem5 reward: speedup ratio on the single biggest usable case (K=1), correctness
# on ALL usable cases, 3x per-case tick budget — identical to the n=1024 eval.
export PIE_MODE=1 PIE_REWARD_IMPL=gem5 PIE_GEM5_REWARD_KIND=ratio
export PIE_GEM5_REWARD_K=1 PIE_GEM5_CASE_SELECT=biggest
export PIE_GEM5_REWARD_SAMPLE=group PIE_GEM5_BUDGET_MULT=3.0
export PIE_GEM5_REWARD_WORKERS="${PIE_GEM5_REWARD_WORKERS:-64}"
export PIE_NAT_TIMEOUT="${PIE_NAT_TIMEOUT:-10}" PIE_GEM5_REWARD_DEADLINE="${PIE_GEM5_REWARD_DEADLINE:-14400}"
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
export TMPDIR="/tmp/${USER}_prerl_${SLURM_JOB_ID:-$$}"; export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"
mkdir -p "${TMPDIR}" "${PIE_COMPILE_SCRATCH}"

N="${N:-32}"; TEMP="${TEMP:-1.0}"; TOPP="${TOPP:-1.0}"; LIMIT="${LIMIT:-0}"; SUF=""
if [ -n "${SMOKE:-}" ]; then N=2; LIMIT=2; SUF="_smoke"; fi

run_cfg () {
  local name="$1" model="$2" prompt="$3" think="$4" maxtok="$5" maxseqs="${6:-0}"
  [ -n "${ONLY_CONFIG:-}" ] && [ "${ONLY_CONFIG}" != "${name}" ] && { echo "skip ${name}"; return; }
  [ -n "${SMOKE:-}" ] && maxtok=2048
  local cfg="${name}${SUF}"
  [ -f "${OUT_DIR}/gen_${cfg}.json" ] && [ -z "${FORCE:-}" ] && { echo "exists ${cfg}, skip (FORCE=1 to redo)"; return; }
  echo "================ ${cfg} (n=${N} limit=${LIMIT} maxtok=${maxtok} maxseqs=${maxseqs}) ================"
  local extra=""; [ "${LIMIT}" != "0" ] && extra="--limit ${LIMIT}"
  $PY -m experiment11_code_optim.eval.prerl_gen_score \
    --config_name "${cfg}" --model "${model}" --prompt "${prompt}" --think_mode "${think}" \
    --problems_json "${PROBLEMS_JSON}" --n "${N}" --temperature "${TEMP}" --top_p "${TOPP}" \
    --max_tokens "${maxtok}" --max_num_seqs "${maxseqs}" --tensor_parallel_size 1 \
    --out_dir "${OUT_DIR}" ${extra}
}

INST=Qwen/Qwen3-4B-Instruct-2507
THINK=Qwen/Qwen3-4B-Thinking-2507
echo "=== prerl | OUT=${OUT_DIR} | N=${N} temp=${TEMP} | ONLY_CONFIG=${ONLY_CONFIG:-<all>} SMOKE=${SMOKE:-0} ==="
run_cfg inst2507_P1  "${INST}"  P1 off    16384
run_cfg inst2507_P2  "${INST}"  P2 off    16384
run_cfg inst2507_P3  "${INST}"  P3 off    16384
run_cfg think2507_P1 "${THINK}" P1 native 32768 48
run_cfg think2507_P2 "${THINK}" P2 native 32768 48
# matched 1.7B anchor: the instruct model used for the gem5 RL training (enable_thinking=False)
INST17B=Qwen/Qwen3-1.7B
run_cfg inst17b_P1   "${INST17B}" P1 off 16384
run_cfg inst17b_P2   "${INST17B}" P2 off 16384
run_cfg inst17b_P3   "${INST17B}" P3 off 16384
echo "=== done -> ${OUT_DIR} ==="
