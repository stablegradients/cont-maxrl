#!/usr/bin/env bash
set -euo pipefail

PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
SCRIPT="experiment1e_inference_maxrl.py"
OUT_BASE="results-bandits/results_1e"
NUM_WORKERS=32  # 4 GPUs × 8 per GPU

# ──────────────────────────────────────────────────────────────────────
# Phase 0: Sanity checks
# ──────────────────────────────────────────────────────────────────────
echo "===== Phase 0: Sanity checks ====="
$PYTHON -u $SCRIPT --sanity-only
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 1: Alpha sweep at N=32, softmax only
#   5 baselines + 5 softmax alphas = 10 methods × 1 N × 5 seeds = 50 runs
# ──────────────────────────────────────────────────────────────────────
PHASE1_DIR="${OUT_BASE}/phase1_alpha_sweep"

if [ -f "$PHASE1_DIR/metrics.json" ]; then
  echo "===== Phase 1: SKIPPED (already done: $PHASE1_DIR/metrics.json) ====="
else
  echo "===== Phase 1: Alpha sweep at N=32, softmax only ====="
  echo "  Output: $PHASE1_DIR"
  echo "  Methods: 10 (5 baseline/survival + 5 softmax)"
  echo "  Runs: 50"
  echo ""

  $PYTHON -u $SCRIPT \
    --output-dir "$PHASE1_DIR" \
    --n-fixed 32 \
    --seeds 42 43 44 45 46 \
    --steps 3000 \
    --log-every 25 \
    --eval-trials 300 \
    --inference-alphas 0.05 0.1 0.2 0.5 1.0 \
    --phase 1 \
    --device auto \
    --num-workers $NUM_WORKERS

  echo ""
  echo "Phase 1 complete."
fi

# ──────────────────────────────────────────────────────────────────────
# Extract best softmax alpha from phase 1 metrics
# ──────────────────────────────────────────────────────────────────────
echo "Selecting best alpha..."

BEST_ALPHA=$($PYTHON -u -c "
import json, numpy as np

with open('${PHASE1_DIR}/metrics_summary.json') as f:
    summary = json.load(f)

agg = summary['aggregated_final_by_method_and_n']
best_key, best_val = None, -float('inf')
for method_key, by_n in agg.items():
    if not method_key.startswith('inference_softmax_'):
        continue
    for n_key, data in by_n.items():
        val = data['metrics']['best_of_n']['mean']
        if val > best_val:
            best_val = val
            best_key = method_key

# Parse alpha from key like inference_softmax_a020
a_str = best_key.split('_a')[1]
alpha = float(a_str[:1] + '.' + a_str[1:])
print(f'{alpha}')
")

echo "  Best softmax alpha (by best-of-N): $BEST_ALPHA"

# Build alpha list for phase 2: best alpha + 0.2 (deduplicated)
if [ "$BEST_ALPHA" = "0.2" ]; then
  PHASE2_ALPHAS="0.2"
else
  PHASE2_ALPHAS="$BEST_ALPHA 0.2"
fi
echo "  Phase 2 alphas: $PHASE2_ALPHAS"
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 2: Full N sweep with best alpha(s), softmax + bernoulli
#   5 baselines + 2 softmax + 2 bernoulli = 9 methods × 9 N × 5 seeds
#   = 405 runs (or 7×9×5=315 if best=0.2)
# ──────────────────────────────────────────────────────────────────────
PHASE2_DIR="${OUT_BASE}/phase2_full_sweep"

if [ -f "$PHASE2_DIR/metrics.json" ]; then
  echo "===== Phase 2: SKIPPED (already done: $PHASE2_DIR/metrics.json) ====="
else
  echo "===== Phase 2: Full N sweep, softmax + bernoulli ====="
  echo "  Output: $PHASE2_DIR"
  echo "  Alphas: $PHASE2_ALPHAS"
  echo ""

  $PYTHON -u $SCRIPT \
    --output-dir "$PHASE2_DIR" \
    --n-values 4 8 16 32 64 128 256 512 1024 \
    --seeds 42 43 44 45 46 \
    --steps 3000 \
    --log-every 25 \
    --eval-trials 300 \
    --inference-alphas $PHASE2_ALPHAS \
    --phase 0 \
    --device auto \
    --num-workers $NUM_WORKERS

  echo ""
  echo "Phase 2 complete."
fi
echo ""

# ──────────────────────────────────────────────────────────────────────
# Phase 3: Full alpha grid (all 5 alphas × both estimators × all N)
#   5 baselines + 5 softmax + 5 bernoulli = 15 methods × 9 N × 5 seeds
#   = 675 runs
# ──────────────────────────────────────────────────────────────────────
PHASE3_DIR="${OUT_BASE}/phase3_full_grid"

if [ -f "$PHASE3_DIR/metrics.json" ]; then
  echo "===== Phase 3: SKIPPED (already done: $PHASE3_DIR/metrics.json) ====="
else
  echo "===== Phase 3: Full alpha grid ====="
  echo "  Output: $PHASE3_DIR"
  echo "  Methods: 15 (5 baseline/survival + 5 softmax + 5 bernoulli)"
  echo "  Runs: 675"
  echo ""

  $PYTHON -u $SCRIPT \
    --output-dir "$PHASE3_DIR" \
    --n-values 4 8 16 32 64 128 256 512 1024 \
    --seeds 42 43 44 45 46 \
    --steps 3000 \
    --log-every 25 \
    --eval-trials 300 \
    --inference-alphas 0.05 0.1 0.2 0.5 1.0 \
    --phase 0 \
    --device auto \
    --num-workers $NUM_WORKERS

  echo ""
  echo "Phase 3 complete."
fi
echo ""

# ──────────────────────────────────────────────────────────────────────
# Regenerate plots (always runs — cheap)
# ──────────────────────────────────────────────────────────────────────
echo "===== Regenerating plots ====="
$PYTHON -u $SCRIPT --plots-only --output-dir "$PHASE1_DIR" \
  --inference-alphas 0.05 0.1 0.2 0.5 1.0
$PYTHON -u $SCRIPT --plots-only --output-dir "$PHASE2_DIR" \
  --inference-alphas $PHASE2_ALPHAS
$PYTHON -u $SCRIPT --plots-only --output-dir "$PHASE3_DIR" \
  --inference-alphas 0.05 0.1 0.2 0.5 1.0

echo ""
echo "===== All phases complete ====="
echo "Results:"
echo "  Phase 1 (alpha sweep):  $PHASE1_DIR"
echo "  Phase 2 (full sweep):   $PHASE2_DIR"
echo "  Phase 3 (full grid):    $PHASE3_DIR"
echo ""
ls -lh ${PHASE3_DIR}/*.png 2>/dev/null || true
