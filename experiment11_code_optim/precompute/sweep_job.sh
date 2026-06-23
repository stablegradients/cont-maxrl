#!/usr/bin/env bash
#SBATCH --job-name=g5sweep
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# One 1-GPU slice = 72 Grace cores (a 4xGH200 node has 288; 1 GPU -> 72),
# which backfills easily. Times shard <SHARD> of <NSHARDS> for <SPLIT>.
#   sbatch --account=<acct>-dtai-gh sweep_job.sh <split> <shard> <nshards>
SPLIT="${1:?split}"; SHARD="${2:?shard}"; NSHARDS="${3:?nshards}"
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
set +u                          # conda gcc activate script is not nounset-clean
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -eo pipefail
PY=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python
export PYTHONNOUSERSITE=1
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
export PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl:${PYTHONPATH:-}
echo "=== g5sweep ${SPLIT} shard ${SHARD}/${NSHARDS} on $(hostname) $(date) ==="
$PY /projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/precompute/dataset_gem5_sweep.py \
    --split "${SPLIT}" --shard "${SHARD}" --nshards "${NSHARDS}"
echo "=== g5sweep ${SPLIT} shard ${SHARD}/${NSHARDS} done $(date) ==="
