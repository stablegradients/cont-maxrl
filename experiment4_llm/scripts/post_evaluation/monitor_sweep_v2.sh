#!/bin/bash
# Quick sweep monitor
echo "=== Post-eval v2 sweep status $(date) ==="
TOTAL=$(cat /projects/bgmo/sramasubramanian/cont-maxrl/post_eval_v2_manifest.json | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
RUNNING=$(squeue -u sramasubramanian -p ghx4-interactive --states=RUNNING --noheader 2>/dev/null | grep "v2_" | wc -l)
PENDING=$(squeue -u sramasubramanian -p ghx4-interactive --states=PENDING --noheader 2>/dev/null | grep "v2_" | wc -l)
# Count completed metrics.json files
DONE=$(find /work/nvme/*/sramasubramanian/cont-maxrl/results/*/post_eval_v2/*/step_*/metrics.json 2>/dev/null | wc -l)
echo "  Total submitted: $TOTAL"
echo "  Running: $RUNNING"
echo "  Pending: $PENDING"  
echo "  Completed: $DONE / $((TOTAL + 1))"
echo "  Remaining: $((TOTAL + 1 - DONE))"
