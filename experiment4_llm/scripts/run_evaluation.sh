#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_jsonl> <output_dir> [--raw-completions]"
  echo "  input_jsonl:"
  echo "    - problem-level rows (binary_passes + continuous_rewards), OR"
  echo "    - raw completion-level rows (generated_code + test_cases) with --raw-completions"
  exit 1
fi

INPUT_JSONL="$1"
OUTPUT_DIR="$2"
RAW_COMPLETIONS=0
if [ "${3:-}" = "--raw-completions" ]; then
  RAW_COMPLETIONS=1
fi

SUMMARY_JSON="$OUTPUT_DIR/metrics_summary.json"
PROBLEM_ROWS_JSONL="$INPUT_JSONL"

mkdir -p "$OUTPUT_DIR"

if [ "$RAW_COMPLETIONS" = "1" ]; then
  PROBLEM_ROWS_JSONL="$OUTPUT_DIR/problem_level_results.jsonl"
  python -m experiment4_llm.src.evaluate_completions \
    --input-jsonl "$INPUT_JSONL" \
    --output-jsonl "$PROBLEM_ROWS_JSONL" \
    --summary-json "$SUMMARY_JSON"
else
  python -m experiment4_llm.src.evaluate \
    --input-jsonl "$PROBLEM_ROWS_JSONL" \
    --output-json "$SUMMARY_JSON"
fi


python experiment4_llm/scripts/generate_figures.py \
  --metrics-summary "$SUMMARY_JSON" \
  --problem-results-jsonl "$PROBLEM_ROWS_JSONL" \
  --output-dir "$OUTPUT_DIR"

printf "Evaluation complete. Summary: %s\n" "$SUMMARY_JSON"
