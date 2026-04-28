#!/bin/bash
# Generate every figure for experiment9 after training + gradient analysis finish.
#
# Usage:
#   bash scripts/make_figures.sh                   # default: K=50, seeds 42+43
#   FIG_K=50 FIG_SEEDS="42 43" bash scripts/make_figures.sh
#
# Emits PDFs + PNGs under results/figures/:
#   figure1_training_dynamics.*
#   figure2_gradient_geometry.*
#   figure3_convergence.*
#   figure5_per_threshold.*
#   figure6_cosine_vs_G.*
#   figure7_grid_{ordinal_ce,plain_ce,mse}.*
#   val_iou_comparison_v{1_neurips,2_presentation,3_minimalist,4_grouped_zoom}.*
#   cosine_grid_{ordinal_ce,plain_ce,mse}.*
#   gradient_geometry_iou.*
#
# Prerequisites:
#   - metrics.json populated under results/<run>/ for each seed (written by
#     run.py when training completes at least one epoch).
#   - gradient_analysis/*.json populated by scripts/run_gradient_analysis.sh
#     (default SEED=43; checkpoint-based figures 2, 5, 6, 7, cosine_grid_*,
#     gradient_geometry_iou). Override with SEEDS="43 44 45" for multi-seed
#     gradient analysis.
# If gradient-analysis data is missing, the figures that depend on it are skipped.

set -euo pipefail

REPO_DIR="/zfsauton/scratch/shrinivr/cont-maxrl"

FIG_K="${FIG_K:-50}"
FIG_SEEDS="${FIG_SEEDS:-43 44 45}"
SEED_ANALYSIS="${SEED_ANALYSIS:-43}"

cd "${REPO_DIR}"

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source ~/.bashrc
    conda activate cont-maxrl
fi
PYBIN="${CONDA_PREFIX:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl}/bin/python"

echo "=== plot.py — figures 1, 2, 3, 5, 6, 7 ==="
"${PYBIN}" -m experiment9_imagenet_localization.plot \
    --K "${FIG_K}" \
    --seeds ${FIG_SEEDS} \
    --seed_analysis "${SEED_ANALYSIS}"

echo
echo "=== plot_val_iou_comparison.py — 4 stylistic variants ==="
"${PYBIN}" -m experiment9_imagenet_localization.plot_val_iou_comparison

echo
echo "=== plot_cosine_grid.py — heatmap grids ==="
"${PYBIN}" -m experiment9_imagenet_localization.plot_cosine_grid

echo
echo "=== plot_gradient_geometry.py — IoU-error scatter ==="
"${PYBIN}" -m experiment9_imagenet_localization.plot_gradient_geometry

echo
echo "Figures written under: ${REPO_DIR}/experiment9_imagenet_localization/results/figures/"
