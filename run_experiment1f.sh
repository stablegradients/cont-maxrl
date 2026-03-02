#!/usr/bin/env bash
set -euo pipefail

PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
SCRIPT="/zfsauton/scratch/shrinivr/cont-maxrl/experiment1f_qweighted.py"
OUT_BASE="results-bandits/results_1f"
NUM_WORKERS=24  # 3 GPUs × 8 per GPU
GPU_IDS="1 2 3"

# ── Sanity checks ─────────────────────────────────────────────────────
echo "===== Sanity checks ====="
$PYTHON -u $SCRIPT --sanity-only
echo ""

# ── Phase 1: N=32 focused comparison (the key experiment) ────────────
#   14 methods × 1 N × 5 seeds = 70 runs → ~3 waves → ~10 min
PHASE1_DIR="${OUT_BASE}/phase1_n32"

if [ -f "$PHASE1_DIR/metrics.json" ]; then
  echo "===== Phase 1 (N=32): SKIPPED (done: $PHASE1_DIR/metrics.json) ====="
else
  echo "===== Phase 1: N=32 focused comparison ====="
  echo "  14 methods × 5 seeds = 70 runs"

  $PYTHON -u $SCRIPT \
    --output-dir "$PHASE1_DIR" \
    --n-values 32 \
    --seeds 42 43 44 45 46 \
    --steps 3000 \
    --eval-trials 300 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $NUM_WORKERS

  echo "Phase 1 complete."
fi
echo ""

# ── Phase 2: N scaling sweep ─────────────────────────────────────────
#   14 methods × 6 N × 5 seeds = 420 runs → ~14 waves → ~45 min
PHASE2_DIR="${OUT_BASE}/phase2_scaling"

if [ -f "$PHASE2_DIR/metrics.json" ]; then
  echo "===== Phase 2 (scaling): SKIPPED (done: $PHASE2_DIR/metrics.json) ====="
else
  echo "===== Phase 2: N scaling sweep ====="
  echo "  14 methods × 6 N × 5 seeds = 420 runs"

  $PYTHON -u $SCRIPT \
    --output-dir "$PHASE2_DIR" \
    --n-values 4 8 16 32 64 128 \
    --seeds 42 43 44 45 46 \
    --steps 3000 \
    --eval-trials 300 \
    --device auto \
    --gpu-ids $GPU_IDS \
    --num-workers $NUM_WORKERS

  echo "Phase 2 complete."
fi
echo ""

# ── Regenerate plots ──────────────────────────────────────────────────
echo "===== Regenerating plots ====="
$PYTHON -u $SCRIPT --plots-only --output-dir "$PHASE1_DIR"
$PYTHON -u $SCRIPT --plots-only --output-dir "$PHASE2_DIR"

echo ""
echo "===== Done ====="
echo "  Phase 1 (N=32):    $PHASE1_DIR"
echo "  Phase 2 (scaling): $PHASE2_DIR"
echo ""
cat "$PHASE1_DIR/comparison_table.txt" 2>/dev/null || true
