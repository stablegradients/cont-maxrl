#!/bin/bash
# Run all bandit experiments (1, 1b, 1c, 1d, 1e, 1f) and generate plots.
# 4 GPUs, 8 workers total.
#
# Usage:
#   nohup bash run_all_bandits.sh > logs/bandits_all.log 2>&1 &
set -euo pipefail

PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

SEEDS="42 43 44"
GPU_IDS="0 1 2 3"
WORKERS=32

overall_start=$(date +%s)


# ─── Experiment 1c: Scaling N=4..256 (GPU) ───
echo ""
echo "=========================================="
echo "  Experiment 1c: Scaling (N=4..256)"
echo "=========================================="
t0=$(date +%s)
$PYTHON experiment1c_scaling.py \
    --output-dir results-bandits/results_1c \
    --n-values 4 8 16 32 64 128 256 \
    --seeds $SEEDS \
    --steps 5000 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $WORKERS
t1=$(date +%s)
echo "Exp 1c done in $(( (t1 - t0) / 60 )) min"

# ─── Experiment 1d: Hard scaling N=4..1024 (GPU) ───
echo ""
echo "=========================================="
echo "  Experiment 1d: Hard scaling (N=4..1024)"
echo "=========================================="
t0=$(date +%s)
$PYTHON experiment1d_hard_bandit.py \
    --output-dir results-bandits/results_1d \
    --n-values 4 8 16 32 64 128 256 512 1024 \
    --seeds $SEEDS \
    --steps 3000 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $WORKERS
t1=$(date +%s)
echo "Exp 1d done in $(( (t1 - t0) / 60 )) min"

# ─── Experiment 1e: Inference MaxRL (GPU, 3 phases) ───
echo ""
echo "=========================================="
echo "  Experiment 1e: Inference MaxRL"
echo "  Phase 1: alpha sweep at N=32"
echo "=========================================="
t0=$(date +%s)
$PYTHON experiment1e_inference_maxrl.py \
    --output-dir results-bandits/results_1e/phase1 \
    --phase 1 \
    --n-fixed 32 \
    --seeds $SEEDS \
    --steps 3000 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $WORKERS
t1=$(date +%s)
echo "Exp 1e phase 1 done in $(( (t1 - t0) / 60 )) min"

echo ""
echo "=========================================="
echo "  Experiment 1e Phase 2: full N sweep"
echo "=========================================="
t0=$(date +%s)
$PYTHON experiment1e_inference_maxrl.py \
    --output-dir results-bandits/results_1e/phase2 \
    --phase 2 \
    --n-values 4 8 16 32 64 128 256 512 1024 \
    --seeds $SEEDS \
    --steps 3000 \
    --inference-alphas 0.05 0.1 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $WORKERS
t1=$(date +%s)
echo "Exp 1e phase 2 done in $(( (t1 - t0) / 60 )) min"

# ─── Experiment 1f: Q-weighted MaxRL (GPU, 2 phases) ───
echo ""
echo "=========================================="
echo "  Experiment 1f: Q-weighted MaxRL"
echo "  Phase 1: N=32 focused"
echo "=========================================="
t0=$(date +%s)
$PYTHON experiment1f_qweighted.py \
    --output-dir results-bandits/results_1f/phase1 \
    --n-values 32 \
    --seeds $SEEDS \
    --steps 3000 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $WORKERS
t1=$(date +%s)
echo "Exp 1f phase 1 done in $(( (t1 - t0) / 60 )) min"

echo ""
echo "=========================================="
echo "  Experiment 1f Phase 2: N scaling"
echo "=========================================="
t0=$(date +%s)
$PYTHON experiment1f_qweighted.py \
    --output-dir results-bandits/results_1f/phase2 \
    --n-values 4 8 16 32 64 128 \
    --seeds $SEEDS \
    --steps 3000 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $WORKERS
t1=$(date +%s)
echo "Exp 1f phase 2 done in $(( (t1 - t0) / 60 )) min"

# ─── Summary ───
overall_end=$(date +%s)
echo ""
echo "=========================================="
echo "  All bandit experiments done!"
echo "  Total time: $(( (overall_end - overall_start) / 60 )) min"
echo "  Results: results-bandits/"
echo "=========================================="
