#!/bin/bash
# Binary recovery sanity check (L=2)
# Supervised: ordinal_ce, plain_ce (10 runs)
# RL at N=64: cont_maxrl, binary_maxrl, grpo, reinforce (20 runs)
# RL at N=256,1024: cont_maxrl, binary_maxrl (20 runs)
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")/runs"

# Supervised at L=2
for METHOD in ordinal_ce plain_ce; do
    for SEED in 42 43 44 45 46; do
        echo "Starting: ${METHOD} L=2 seed=${SEED}"
        bash "$SCRIPT_DIR/train_${METHOD}_L2_seed${SEED}.sh" &
    done
done

# RL at L=2, N=64
for METHOD in cont_maxrl binary_maxrl grpo reinforce; do
    for SEED in 42 43 44 45 46; do
        echo "Starting: ${METHOD} N=64 L=2 seed=${SEED}"
        bash "$SCRIPT_DIR/train_${METHOD}_N64_L2_seed${SEED}.sh" &
    done
done

# Higher N at L=2 for convergence check
for METHOD in cont_maxrl binary_maxrl; do
    for N in 256 1024; do
        for SEED in 42 43 44 45 46; do
            echo "Starting: ${METHOD} N=${N} L=2 seed=${SEED}"
            bash "$SCRIPT_DIR/train_${METHOD}_N${N}_L2_seed${SEED}.sh" &
        done
    done
done

wait
echo "All binary recovery runs complete."
