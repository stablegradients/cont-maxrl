#!/usr/bin/env bash
#SBATCH --job-name=qwen3_base_eval
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=288
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# PIE zero-shot baseline eval for ONE (model, template, decoding) combo.
# Driven by the parent script which calls sbatch with the right env vars.
#
# Required env (passed via --export):
#   MODEL_ID           HF model id (e.g., "Qwen/Qwen3-0.6B-Base")
#   TEMPLATE_MODE      "chat" | "raw" | "raw_oneshot"
#   N_COMPLETIONS      e.g., 16 (sampled) or 1 (greedy)
#   TEMPERATURE        1.0 or 0.0
#   STOP_TOKEN_IDS_STR space-separated EOS list, e.g. "151643" or "151645 151643"
#   OUTPUT_TAG         a string discriminating this run, e.g. "base_0.6B_chat_n16_t1"
#   ONESHOT_FILE       (optional) preamble file path for raw_oneshot
#
# Optional env:
#   EVAL_PARQUET       default = pie_val_200.parquet
#   MAX_TOKENS         default 4096
#   TOP_P              default 1.0
#   TOP_K              default -1

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -eo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
cd "$REPO"
set -a; source .env 2>/dev/null || true; set +a
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# PIE test cases + walltime backend (pass-rate doesn't need perf or refs)
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
export PIE_TEST_CASE_PICKLE=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases.pkl
export PIE_MEASUREMENT_MAX_TESTS=5
export PIE_COMPILE_SCRATCH="${TMPDIR:-/tmp}/${USER}_pie_scratch"
mkdir -p "${PIE_COMPILE_SCRATCH}"
unset PIE_BACKEND  # walltime is fine for pass-rate

# vLLM V0 — V1's torch.compile path hits a triton_key ImportError with the
# triton version in our conda env (same workaround as post_eval_v2_job.sh
# and the training scripts).
export VLLM_USE_V1=0

PYTHON=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python

EVAL_PARQUET="${EVAL_PARQUET:-/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_val_200.parquet}"
N_COMPLETIONS="${N_COMPLETIONS:-16}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

OUTPUT_DIR="${REPO}/experiment11_code_optim/docs/baseline_eval/${OUTPUT_TAG}"
mkdir -p "${OUTPUT_DIR}"

# Derive EOS list from model class (avoids sbatch --export comma-in-value
# parsing issue). Qwen3-Base: <|endoftext|>=151643. Qwen3-Instruct: prefer
# <|im_end|>=151645, fall back to <|endoftext|>=151643.
STOP_ARGS=""
if [[ "${MODEL_ID}" == *-Base ]]; then
  STOP_ARGS="--stop_token_ids 151643"
else
  STOP_ARGS="--stop_token_ids 151645 151643"
fi

ONESHOT_ARGS=""
if [ "${TEMPLATE_MODE}" = "raw_oneshot" ] && [ -n "${ONESHOT_FILE:-}" ]; then
  ONESHOT_ARGS="--oneshot_preamble ${ONESHOT_FILE}"
fi

echo "============================================================"
echo "node: $(hostname)  gpus: $(nvidia-smi -L | wc -l)  $(date)"
echo "MODEL_ID=${MODEL_ID}  TEMPLATE_MODE=${TEMPLATE_MODE}"
echo "N_COMPLETIONS=${N_COMPLETIONS}  T=${TEMPERATURE}  STOP=${STOP_TOKEN_IDS_STR:-default}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "============================================================"

$PYTHON "${REPO}/experiment4_llm/scripts/post_evaluation/sharded_eval_v2_worker.py" \
  --results_dir "${OUTPUT_DIR}/results_dir_stub" \
  --run_name "baseline_${OUTPUT_TAG}" \
  --step 0 \
  --hf_dir "${MODEL_ID}" \
  --benchmark pie_val \
  --data_file "${EVAL_PARQUET}" \
  --output_dir "${OUTPUT_DIR}" \
  --n_gpus 4 \
  --n_completions "${N_COMPLETIONS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --max_tokens "${MAX_TOKENS}" \
  --gpu_memory_utilization 0.85 \
  --timeout_per_test 6 \
  --n_workers 64 \
  --k_values 1 5 16 \
  --pie_mode \
  --template_mode "${TEMPLATE_MODE}" \
  ${ONESHOT_ARGS} \
  ${STOP_ARGS}

echo "=== Done $(date) ==="
