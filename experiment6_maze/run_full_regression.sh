#!/bin/bash
set -e

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTEST="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python -m pytest"

echo "=== [1/5] Tier 0: backward-compat ==="
$PYTEST experiment6_maze/tests/test_backward_compat.py -v

echo "=== [2/5] Tier 3: advantage estimator equivalence (10K fuzz) ==="
$PYTEST experiment6_maze/tests/test_advantages_binary_equivalence.py -v

echo "=== [3/5] Tier 4: verl integration ==="
$PYTEST experiment6_maze/tests/test_verl_integration.py -v

echo "=== [4/5] DeepCoder custom reward still works ==="
$PYTEST experiment4_llm/tests/test_reward.py -v
$PYTEST experiment4_llm/tests/test_verl_integration.py -v

echo "=== [5/5] End-to-end smoke ==="
$PYTEST experiment6_maze/tests/test_end_to_end_smoke.py -v

echo "=== ALL REGRESSION TESTS PASSED ==="
