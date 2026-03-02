#!/bin/bash
# Run all RL methods at N=4 (4 methods × 5 seeds = 20 runs)
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")/runs"

for METHOD in cont_maxrl binary_maxrl grpo reinforce; do
    for SEED in 42 43 44 45 46; do
        echo "Starting: ${METHOD} N=4 seed=${SEED}"
        bash "$SCRIPT_DIR/train_${METHOD}_N4_L100_seed${SEED}.sh" &
    done
done

wait
echo "All N=4 runs complete."
