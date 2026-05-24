#!/usr/bin/env bash
#SBATCH --job-name=pie_probe_q3_1.7b
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# ============================================================
# Spec §9 probe: 256 rollouts × the val_200 problems on Qwen3-1.7B (Instruct
# variant; bare name is the chat model per project_qwen3_naming.md). Reports
# gate-pass rate, fraction with positive reward, mean positive reward.
#
# REQUIRES: src_instr_table_O0.pkl + rebuilt parquets (with src_instr_count
# embedded in ground_truth). If parquets are not yet rebuilt, the score
# column will be zero everywhere (gate-pass still meaningful, but the
# log2-speedup signal won't be).
# ============================================================

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -eo pipefail

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
cd "${REPO_ROOT}"

set -a; source .env 2>/dev/null || true; set +a
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PIE_DATA_ROOT="/work/nvme/bgmo/sramasubramanian/pie_data"
export PIE_TEST_CASE_DIR="${PIE_DATA_ROOT}/tests"
export PIE_TEST_CASE_PICKLE="${PIE_DATA_ROOT}/tests.pkl"
export PIE_BACKEND=perf
export PIE_MEASUREMENT_MAX_TESTS=5
export PIE_PER_TEST_TIMEOUT=30
export PIE_COMPILE_SCRATCH="${TMPDIR:-/tmp}/${USER}_pie_probe"
mkdir -p "${PIE_COMPILE_SCRATCH}"
export VLLM_USE_V1=0

VAL_PARQUET="${PIE_DATA_ROOT}/parquet/pie_val_200.parquet"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"

echo "=== Spec §9 probe: ${MODEL_ID} on ${VAL_PARQUET} (256 rollouts/problem) ==="
$PYTHON -m experiment11_code_optim.eval.go_no_go \
    --sft_ckpt "${MODEL_ID}" \
    --val_parquet "${VAL_PARQUET}" \
    --n_completions 256 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_tokens 4096 \
    --max_model_len 6656 \
    --tensor_parallel_size 1 \
    --output_json "${REPO_ROOT}/experiment11_code_optim/docs/probe_pie_instruct_O0_log2.json"
echo "=== done ==="
