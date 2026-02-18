#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-results_experiment4/smoke}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

echo "[smoke] using python: $PYTHON_BIN"
$PYTHON_BIN -V

echo "[smoke] environment check"
$PYTHON_BIN experiment4_llm/scripts/check_env.py --allow-missing

echo "[smoke] running unit tests"
PYTHONDONTWRITEBYTECODE=1 $PYTHON_BIN -m unittest discover -s experiment4_llm/tests -p 'test_*.py'

echo "[smoke] running custom advantage self-test"
$PYTHON_BIN -m experiment4_llm.src.custom_adv_estimator

echo "[smoke] running end-to-end smoke pipeline"
$PYTHON_BIN experiment4_llm/scripts/smoke_run.py --output-dir "$OUTPUT_DIR"

echo "[smoke] done"
