#!/bin/bash
#SBATCH --job-name=g5sweep
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
module load gcc-native/14
CONDA=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
export LD_LIBRARY_PATH="$CONDA/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1 PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl
cd /projects/bgmo/sramasubramanian/cont-maxrl
echo "=== g5sweep shard ${SHARD} on $(hostname) ==="
$CONDA/bin/python experiment11_code_optim/scripts/reward_analysis/gem5_sweep.py \
  --shard "${SHARD}" --out_dir "${OUT:-/work/nvme/bgtv/sramasubramanian/gem5_sweep}" \
  --k_tests "${KT:-10}" --workers 44 --per_test_timeout 90
echo "=== shard ${SHARD} done ==="
