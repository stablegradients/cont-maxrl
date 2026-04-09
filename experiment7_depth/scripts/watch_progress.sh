#!/bin/bash
# Live dashboard showing running jobs and completion progress.
# Usage: bash experiment7_depth/scripts/watch_progress.sh [runs_dir]
set -euo pipefail

SCRIPT_DIR="${1:-$(dirname "$(realpath "$0")")/runs}"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_DIR="$(cd "$(dirname "$0")/../.." && pwd)/experiment7_depth/results"

while true; do
    clear
    echo "=== Experiment 7 Depth — Progress Dashboard ==="
    echo "$(date)"
    echo ""

    # Count completed jobs (those with metrics.json)
    if [[ -d "$RESULTS_DIR" ]]; then
        completed=$(find "$RESULTS_DIR" -name "metrics.json" 2>/dev/null | wc -l)
    else
        completed=0
    fi
    total=$(ls "$SCRIPT_DIR"/zx7k_*.sh 2>/dev/null | wc -l)
    echo "Completed: $completed / $total"
    echo ""

    # Show running jobs
    echo "--- Running jobs ---"
    if [[ -d "$LOG_DIR" ]]; then
        for log in "$LOG_DIR"/*.log; do
            [[ -f "$log" ]] || continue
            jobname=$(basename "$log" .log)
            # Check if still running by looking for recent writes
            if [[ $(find "$log" -mmin -2 2>/dev/null | wc -l) -gt 0 ]]; then
                last_line=$(tail -1 "$log" 2>/dev/null || echo "")
                echo "  RUNNING: $jobname | $last_line"
            fi
        done
    fi

    # Show recent completions
    echo ""
    echo "--- Recent completions ---"
    if [[ -d "$RESULTS_DIR" ]]; then
        find "$RESULTS_DIR" -name "metrics.json" -mmin -30 2>/dev/null | \
            sort -t/ -k5 | tail -5 | while read f; do
            dir=$(dirname "$f")
            jobname=$(basename "$dir")
            echo "  DONE: $jobname"
        done
    fi

    sleep 30
done
