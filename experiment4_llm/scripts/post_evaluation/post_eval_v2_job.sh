#!/usr/bin/env bash
# Post-eval v2 single-job wrapper.
# Usage: sbatch [slurm opts] post_eval_v2_job.sh <run_dir> <step> <benchmark> <data_file>
# benchmark: humaneval_plus | mbpp_plus | livecodebench_v5

set -eo pipefail

RUN_DIR="$1"
STEP="$2"
BENCHMARK="$3"
DATA_FILE="$4"

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
export TMPDIR="/tmp/${USER}_posteval_v2"; mkdir -p "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR="${RUN_DIR}/post_eval_v2/${BENCHMARK}/step_${STEP}"
HF_DIR="${RUN_DIR}/global_step_${STEP}/actor_hf"

# FSDP merge if needed
if [ ! -f "${HF_DIR}/config.json" ]; then
    ACTOR_DIR="${RUN_DIR}/global_step_${STEP}/actor"
    echo "Merging FSDP checkpoint..."
    $PYTHON -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from experiment4_llm.src.eval_checkpoints import merge_fsdp_checkpoint
merge_fsdp_checkpoint('${ACTOR_DIR}', '${HF_DIR}', 'Qwen/Qwen3-1.7B-Base', '${REPO_ROOT}/vendor/verl/scripts/model_merger.py')
"
    if [ ! -f "${HF_DIR}/config.json" ]; then
        echo "ERROR: FSDP merge failed"
        exit 1
    fi
fi

# Skip if already done
if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "Already done: ${OUTPUT_DIR}/metrics.json exists. Skipping."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== Post-eval v2: $(basename ${RUN_DIR}) step ${STEP} ${BENCHMARK} ==="

# Write sampling config
cat > "${OUTPUT_DIR}/sampling_config.json" << 'SAMPLING_EOF'
{
    "n_completions": 256,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "max_tokens": 2048,
    "enable_thinking": false,
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.90,
    "timeout_per_test": 6,
    "n_workers": 128,
    "k_values": [1, 4, 16, 64, 256]
}
SAMPLING_EOF

# Use sharded approach: 4×TP=1 parallel vLLM instances, ~2.4x faster than TP=4
$PYTHON "${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/sharded_eval_v2_worker.py" \
    --results_dir "${RUN_DIR}" \
    --run_name "$(basename ${RUN_DIR})" \
    --step "${STEP}" \
    --hf_dir "${HF_DIR}" \
    --benchmark "${BENCHMARK}" \
    --data_file "${REPO_ROOT}/data/${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_gpus 4 \
    --n_completions 256 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.95 \
    --timeout_per_test 6 \
    --n_workers 64 \
    --k_values 1 4 16 64 256

echo "=== Done: $(basename ${RUN_DIR}) step ${STEP} ${BENCHMARK} ==="
