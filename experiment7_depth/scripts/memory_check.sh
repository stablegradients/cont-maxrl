#!/bin/bash
# Verify memory at the worst cells with 4-GPU DDP, local batch 8.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Memory check: RL cont_maxrl K=1000 G=256, 4-GPU DDP global B=32 ==="
torchrun --standalone --nproc_per_node=4 \
    -m experiment7_depth.run \
    --method cont_maxrl --K 1000 --G 256 --seed 42 \
    --global_batch_size 32 --epochs 1 --epochs_hard_cap 1 --max_steps 3 \
    --lr 2e-4 --momentum 0.9 --weight_decay 5e-4 --amp \
    --data_dir "$REPO_ROOT/data/nyu" --output_dir /tmp/memcheck_rl_$$ \
    --no_wandb --print_memory --num_workers 2

echo ""
echo "=== Memory check: supervised ordinal_ce K=1000, 4-GPU DDP global B=32 ==="
torchrun --standalone --nproc_per_node=4 \
    -m experiment7_depth.run \
    --method ordinal_ce --K 1000 --seed 42 \
    --global_batch_size 32 --epochs 1 --epochs_hard_cap 1 --max_steps 3 \
    --lr 2e-4 --momentum 0.9 --weight_decay 5e-4 --amp \
    --data_dir "$REPO_ROOT/data/nyu" --output_dir /tmp/memcheck_sup_$$ \
    --no_wandb --print_memory --num_workers 2

echo ""
echo "=== Memory checks complete ==="
echo "Expected per-GPU peak: ~12 GB for RL, ~22 GB for supervised ordinal."
echo "Both should be well under 48 GB."

# Cleanup
rm -rf /tmp/memcheck_rl_$$ /tmp/memcheck_sup_$$
