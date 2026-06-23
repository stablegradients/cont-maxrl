#!/usr/bin/env bash
#SBATCH --job-name=g5retime
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
# Re-time the partial programs WITHOUT the per-program wall (generous, full
# coverage). 1 GPU / 72 cores per job.
#   sbatch --account=<acct>-dtai-gh retime_job.sh <split> <shard> <nshards>
SPLIT="${1:?split}"; SHARD="${2:?shard}"; NSHARDS="${3:?nshards}"
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
set +u
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -eo pipefail
PY=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python
export PYTHONNOUSERSITE=1
export PIE_TEST_CASE_DIR=/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases
# Fixed pipeline: native pre-screen drops hanging/crashing cases so the forkloop
# no longer stalls. Keep amortization (it's fine); the ONLY thing that needs the
# wall is genuinely-slow gem5 sim. Each case is budget-capped at 120s, so <=104
# cases <= ~3.5h -> 3.5h wall lets every program complete instead of going partial.
export PIE_SWEEP_FORKLOOP_WALL=12600
export PIE_NATIVE_SCREEN=1               # explicit: screen out hanging cases (PIE 120s-timeout analog)
export PIE_RETIME_TAG=zzz_retime_fix     # fresh tag, sorts AFTER zz_retime_ so it wins in consolidate
export PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl:${PYTHONPATH:-}
echo "=== g5retime ${SPLIT} shard ${SHARD}/${NSHARDS} on $(hostname) $(date) wall=${PIE_SWEEP_FORKLOOP_WALL}s tag=${PIE_RETIME_TAG} ==="
$PY /projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/precompute/retime_partials.py \
    --split "${SPLIT}" --shard "${SHARD}" --nshards "${NSHARDS}"
echo "=== g5retime ${SPLIT} shard ${SHARD}/${NSHARDS} done $(date) ==="
