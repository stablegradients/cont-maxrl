#!/bin/bash
#SBATCH --job-name=gos_K50_N65536_seed44_fastSingle
#SBATCH --partition=general
# A5000 or A6000 — exclude smaller cards.
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/gos_K50_N65536_seed44_fastSingle_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N65536_fastSingle"

# Speedup stack:
#  - bf16 autocast (~1.4×, gradient cos ≈ 1.0 vs fp32 verified previously).
#  - torch.compile mode='default' (~1.2×) — CUDA-graph mode disabled,
#    so the prior CPU-RAM-growth issue does not apply.
#  - vectorized batched_max_iou (~2-4× at N=16k/65k).
#  - online metrics accumulator (frees ~70 GB / ~270 GB at N=16k/65k).
export BF16=1

cd "/zfsauton/scratch/shrinivr/cont-maxrl"
source ~/.bashrc
conda activate cont-maxrl
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}

python -m experiment9_imagenet_localization.run \
    --method gos \
    --K 50 \
    --N 65536 \
    --seed 44 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/gos_K50_N65536_seed44_fastSingle" \
    --num_workers 8 \
    --wandb
