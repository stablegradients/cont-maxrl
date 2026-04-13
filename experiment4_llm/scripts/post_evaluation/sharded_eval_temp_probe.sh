#!/usr/bin/env bash
# Temperature probe eval: run HE+ only at a specific temperature.
# Usage: sbatch [slurm opts] sharded_eval_temp_probe.sh <results_dir> <run_name> <step> <temperature>

set -o pipefail

RESULTS_DIR="$1"
RUN_NAME="$2"
STEP="$3"
TEMPERATURE="$4"

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

export PYTHONNOUSERSITE=1
export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET; unset NCCL_NET_PLUGIN; export NCCL_PLUGIN_P2P=0
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_USE_V1=0
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export TORCH_COMPILE_DISABLE=1; export TORCHDYNAMO_DISABLE=1
export TMPDIR="/tmp/${USER}_temp_probe_tmp"; mkdir -p "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

HF_DIR="${RESULTS_DIR}/global_step_${STEP}/actor_hf"
if [ ! -f "${HF_DIR}/config.json" ]; then
    echo "ERROR: No merged model at ${HF_DIR}"
    exit 1
fi

# Use temperature 0.01 instead of 0.0 (vLLM requires temperature > 0 with top_p)
ACTUAL_TEMP="${TEMPERATURE}"
if [ "$(echo "$TEMPERATURE == 0" | bc -l)" = "1" ]; then
    ACTUAL_TEMP="0.01"
fi

PROBE_DIR="${RESULTS_DIR}/temp_probe_T${TEMPERATURE}"
mkdir -p "${PROBE_DIR}"

echo "=== Temp probe: ${RUN_NAME} step ${STEP} T=${TEMPERATURE} ==="
echo "Output: ${PROBE_DIR}"

$PYTHON "${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/sharded_eval_save_completions_worker.py" \
    --results_dir "${PROBE_DIR}" \
    --run_name "${RUN_NAME}_T${TEMPERATURE}" \
    --step "${STEP}" \
    --hf_dir "${HF_DIR}" \
    --n_gpus 4 \
    --n_completions 16 \
    --temperature ${ACTUAL_TEMP} \
    --top_p 0.95 \
    --top_k -1 \
    --max_tokens 4096 \
    --gpu_memory_utilization 0.95 \
    --timeout_per_test 6 \
    --n_workers 64 \
    --k_values 1 4 16

echo "=== Temp probe done: ${RUN_NAME} T=${TEMPERATURE} ==="
