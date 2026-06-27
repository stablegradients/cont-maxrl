#!/usr/bin/env bash
#SBATCH --job-name=pie_src_instr_O0
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
# Precompute src_instr_count per (src_code, problem_id) at -O0 / perf backend.
# Produces /work/nvme/bgmo/sramasubramanian/pie_data/src_instr_table_O0.pkl
# Wall time estimate: ~13 minutes for ~74k unique src programs at ~93 grades/s
# (reference-table build rate); the time budget is 2 h to leave slack for
# the long-tail at -O0.
#
# Run BEFORE build_pie_parquet.py and BEFORE any RL training, since the
# parquet rebuild joins this pickle and the reward fn reads src_instr_count
# out of every ground_truth at training time.
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

export PIE_BACKEND=perf
export PIE_MEASUREMENT_MAX_TESTS=5
export PIE_COMPILE_SCRATCH="${TMPDIR:-/tmp}/${USER}_pie_src_instr"
mkdir -p "${PIE_COMPILE_SCRATCH}"
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
export PIE_TEST_CASE_PICKLE=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases.pkl

echo "=== build_src_instr_table.py (O0 / perf backend) ==="
$PYTHON "${REPO_ROOT}/experiment11_code_optim/precompute/build_src_instr_table.py"
echo "=== done ==="
