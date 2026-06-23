#!/usr/bin/env bash
#SBATCH --job-name=g5isect
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err
SPLIT="${1:?}"; SHARD="${2:?}"; NSHARDS="${3:?}"
module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"; set +u
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl
set -eo pipefail
ulimit -c 0
export PYTHONNOUSERSITE=1 PIE_NAT_TIMEOUT=4
export PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl:${PYTHONPATH:-}
/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \
  /projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/precompute/pie_intersect_shard.py \
  --split "$SPLIT" --shard "$SHARD" --nshards "$NSHARDS"
