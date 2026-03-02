#!/bin/bash
# Run gradient analysis on saved checkpoints
# Analyzes epochs 1, 10, 25, 50 for key methods
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
RESULTS_DIR="$REPO_ROOT/experiment5_ordinal/results"

cd "$REPO_ROOT"

METHODS="ordinal_ce plain_ce cont_maxrl grpo reinforce"
EPOCHS="1 10 25 50"

for METHOD in $METHODS; do
    for EPOCH in $EPOCHS; do
        # Determine run name based on method type
        if [ "$METHOD" = "ordinal_ce" ] || [ "$METHOD" = "plain_ce" ]; then
            RUN_NAME="${METHOD}_L100_N0_seed42"
        else
            RUN_NAME="${METHOD}_L100_N64_seed42"
        fi

        CKPT="$RESULTS_DIR/$RUN_NAME/checkpoints/epoch_${EPOCH}.pt"

        if [ ! -f "$CKPT" ]; then
            echo "Skipping: $CKPT not found"
            continue
        fi

        echo "Gradient analysis: $METHOD epoch=$EPOCH"
        $PYTHON -m experiment5_ordinal.gradient_analysis \
            --checkpoint "$CKPT" \
            --data_dir "$REPO_ROOT/data/utkface" \
            --method "$METHOD" \
            --L 100 \
            --seed 42 \
            --output_dir "$RESULTS_DIR/$RUN_NAME/gradient_analysis" \
            --n_images 500 &
    done
done

wait
echo "All gradient analyses complete."
