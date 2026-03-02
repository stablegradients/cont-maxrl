#!/bin/bash
# Master script: runs all experiment steps in order
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
SCRIPT_DIR="$(dirname "$0")"

cd "$REPO_ROOT"

echo "=== Step 0: Generate run scripts ==="
$PYTHON -m experiment5_ordinal.sweep --output_dir experiment5_ordinal/scripts/runs

echo "=== Step 1: Download/verify data ==="
$PYTHON -m experiment5_ordinal.data --output_dir ./data/utkface --verify

echo "=== Step 2: Supervised methods (L=100) ==="
bash "$SCRIPT_DIR/run_supervised.sh"

echo "=== Step 3: RL sweeps (L=100) ==="
for N in 4 16 64 256 1024; do
    echo "--- N=$N ---"
    bash "$SCRIPT_DIR/run_rl_N${N}.sh"
done

echo "=== Step 4: Binary recovery (L=2) ==="
bash "$SCRIPT_DIR/run_binary_recovery.sh"

echo "=== Step 5: Gradient analysis ==="
bash "$SCRIPT_DIR/run_gradient_analysis.sh"

echo "=== Step 6: Generate figures ==="
$PYTHON -m experiment5_ordinal.plot \
    --results_dir experiment5_ordinal/results \
    --output_dir experiment5_ordinal/results/figures

echo "=== Done ==="
