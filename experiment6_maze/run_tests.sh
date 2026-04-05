#!/bin/bash
set -e

REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTEST="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python -m pytest"

echo "=== [Tier 0] Backward compatibility ==="
$PYTEST experiment6_maze/tests/test_backward_compat.py -v

echo "=== [Tier 1] Maze env ==="
$PYTEST experiment6_maze/tests/test_maze_env.py -v

echo "=== [Tier 1] Reward function ==="
$PYTEST experiment6_maze/tests/test_reward_function.py -v

echo "=== [Tier 2] Data prep ==="
$PYTEST experiment6_maze/tests/test_data_prep.py -v

echo "=== [Tier 3] Advantage equivalence ==="
$PYTEST experiment6_maze/tests/test_advantages_binary_equivalence.py -v

echo "=== [Tier 4] verl integration ==="
$PYTEST experiment6_maze/tests/test_verl_integration.py -v

echo "=== ALL TIER 0-4 TESTS PASSED ==="
