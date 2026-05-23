#!/bin/bash
#SBATCH --account=bgmo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=pie_ref_table_perf
#SBATCH --output=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/pie_ref_table_perf_%j.out
#SBATCH --error=/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/pie_ref_table_perf_%j.err

# Rebuild the reference table under the perf retired-instruction backend.
# Output lands in a NEW directory (`reference_table_perf/`) and pickle so the
# original wall-clock table is left untouched for A/B comparison + rollback.

module load python/miniforge3_pytorch/2.10.0
eval "$(conda shell.bash hook)"
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl

set -eo pipefail
REPO=/projects/bgmo/sramasubramanian/cont-maxrl
cd "$REPO"
set -a; source .env 2>/dev/null || true; set +a
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# Inputs
export PIE_TEST_CASE_DIR=/work/nvme/bgmo/sramasubramanian/pie_data/tests
export PIE_TEST_CASE_PICKLE=/work/nvme/bgmo/sramasubramanian/pie_data/tests.pkl
export PIE_MEASUREMENT_MAX_TESTS=5
export PIE_COMPILE_SCRATCH=/tmp/${USER}_pie_scratch
mkdir -p "${PIE_COMPILE_SCRATCH}"

# Critical: build under perf backend so MANIFEST.json records backend="perf".
export PIE_BACKEND=perf

# Reward worker concurrency — the perf path has no kernel-level fd
# serialisation cap (that was the perf-stat-subprocess wrapper, not
# perf_event_open direct), so we can stay at production worker count.
export VERL_REWARD_WORKERS=64

# Output paths — NEW dirs/files, walltime table not overwritten
NEW_REF_DIR=/work/nvme/bgmo/sramasubramanian/pie_data/reference_table_perf

PY=/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python

echo "============================================================"
echo "node: $(hostname)  cpus: $(nproc)  gpus: $(nvidia-smi -L | wc -l)  $(date)"
echo "PIE_BACKEND=${PIE_BACKEND}  output=${NEW_REF_DIR}"
echo "============================================================"

$PY "${REPO}/experiment11_code_optim/precompute/build_reference_table.py" \
    --output-root "${NEW_REF_DIR}"

echo
echo "=== verifying MANIFEST.json carries backend=perf ==="
$PY -c "
import json
with open('${NEW_REF_DIR}/MANIFEST.json') as f:
    m = json.load(f)
print(json.dumps(m, indent=2))
assert m.get('backend') == 'perf', 'MANIFEST.json missing backend=perf'
print('OK: backend recorded')
"

echo
echo "=== verifying pickle loads via preload_reference_table ==="
PIE_REFERENCE_TABLE_DIR="${NEW_REF_DIR}" \
PIE_REFERENCE_TABLE_PICKLE="${NEW_REF_DIR}.pkl" \
$PY -c "
import os
os.environ['PIE_BACKEND'] = 'perf'
from experiment11_code_optim.reward.pie_reward_fn import preload_reference_table
n = preload_reference_table('${NEW_REF_DIR}')
print(f'OK: loaded {n} (split, pid) entries from perf-units table')
"

echo
echo "=== Done $(date) ==="
