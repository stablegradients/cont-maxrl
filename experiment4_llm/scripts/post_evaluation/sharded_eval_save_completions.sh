#!/usr/bin/env bash
# Sharded eval WITH completion saving for reward hacking analysis.
# Generates 16 completions per problem (enough to inspect patterns).
# Saves raw completions + extracted code + per-completion scores.
# Usage: sbatch [slurm opts] sharded_eval_save_completions.sh <results_dir> <run_name> <step>

set -o pipefail

RESULTS_DIR="$1"
RUN_NAME="$2"
STEP="$3"

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
export TMPDIR="/tmp/${USER}_compl_eval_tmp"; mkdir -p "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== Completion-saving eval: ${RUN_NAME} step ${STEP} ==="

# Merge FSDP checkpoint if needed
ACTOR_DIR="${RESULTS_DIR}/global_step_${STEP}/actor"
HF_DIR="${RESULTS_DIR}/global_step_${STEP}/actor_hf"
if [ ! -f "${HF_DIR}/config.json" ]; then
    echo "Merging FSDP checkpoint..."
    $PYTHON -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from experiment4_llm.src.eval_checkpoints import merge_fsdp_checkpoint
merge_fsdp_checkpoint('${ACTOR_DIR}', '${HF_DIR}', 'Qwen/Qwen3-1.7B-Base', '${REPO_ROOT}/vendor/verl/scripts/model_merger.py')
"
    if [ ! -f "${HF_DIR}/config.json" ]; then
        echo "ERROR: FSDP merge failed for step ${STEP}"
        exit 1
    fi
fi

$PYTHON "${REPO_ROOT}/experiment4_llm/scripts/post_evaluation/sharded_eval_save_completions_worker.py" \
    --results_dir "${RESULTS_DIR}" \
    --run_name "${RUN_NAME}" \
    --step "${STEP}" \
    --hf_dir "${HF_DIR}" \
    --n_gpus 4 \
    --n_completions 16 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k -1 \
    --max_tokens 4096 \
    --gpu_memory_utilization 0.95 \
    --timeout_per_test 6 \
    --n_workers 64 \
    --k_values 1 4 16

echo "=== Completion-saving eval done: ${RUN_NAME} step ${STEP} ==="
