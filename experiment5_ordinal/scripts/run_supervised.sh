#!/bin/bash
# Run all supervised methods (3 methods × 5 seeds = 15 runs)
# Methods: ordinal_ce, plain_ce, mse
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")/runs"

for METHOD in ordinal_ce plain_ce mse; do
    for SEED in 42 43 44 45 46; do
        echo "Starting: ${METHOD} seed=${SEED}"
        bash "$SCRIPT_DIR/train_${METHOD}_L100_seed${SEED}.sh" &
    done
done

wait
echo "All supervised runs complete."
