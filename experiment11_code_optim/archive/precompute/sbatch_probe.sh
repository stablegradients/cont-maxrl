#!/usr/bin/env bash
#SBATCH --job-name=pie_probe
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=72
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/%x_%j.err

# Re-run of §13 probe items 3-6 inside a proper multi-CPU SLURM allocation.
# The previous interactive baseline ran with SLURM_CPUS_ON_NODE=1 + a single-
# core CPU mask (0x...0001), which explains the flat 0.99x "parallel" speedup
# — there was no parallelism available. This run requests --cpus-per-task=72
# (one NUMA-0 socket) so the per-core lock and ThreadPoolExecutor can actually
# do work. The compile-scratch hypothesis was secondary; /tmp and /local are
# the same NVMe device on this node so that was a red herring.

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Spec §4.2: per-record compile scratch on /local. Explicit override so the
# measurement core does not depend on whatever $TMPDIR happens to be.
export PIE_COMPILE_SCRATCH="/local/slurmjobs/${SLURM_JOB_ID}/pie_scratch"
mkdir -p "${PIE_COMPILE_SCRATCH}"

# Pin compute to NUMA-0 (cores 0-71 on this 4-socket node) per spec §1.
# numactl is available; falling back to taskset if not.
NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"

echo "=== sbatch probe context ==="
echo "  SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "  SLURM_JOB_PARTITION  = ${SLURM_JOB_PARTITION}"
echo "  SLURM_JOB_ACCOUNT    = ${SLURM_JOB_ACCOUNT}"
echo "  SLURM_CPUS_ON_NODE   = ${SLURM_CPUS_ON_NODE}"
echo "  SLURM_CPUS_PER_TASK  = ${SLURM_CPUS_PER_TASK}"
echo "  hostname             = $(hostname)"
echo "  TMPDIR               = ${TMPDIR:-<unset>}"
echo "  PIE_COMPILE_SCRATCH  = ${PIE_COMPILE_SCRATCH}"
echo "  cpu bind mask        = ${SLURM_CPU_BIND:-<unset>}"
echo "  scratch FS:"
df -hT "${PIE_COMPILE_SCRATCH}" 2>&1 | head -3 || true
echo ""

time ${NUMA_PREFIX} $PYTHON "${REPO_ROOT}/experiment11_code_optim/precompute/probe.py"

echo ""
echo "=== probe done ==="
