#!/usr/bin/env bash
# ============================================================================
# Qwen3-4B-Instruct-2507 best_reward@k TEST eval — sets the 4B + P3 env so the
# eval prompt/model/tokens MATCH training, then defers to spray_besteval.sh.
# (besteval reads the prompt FROM the parquet, so pointing at the P3 test parquet
# makes the eval prompt byte-identical to the P3 training prompt.)
#   Usage: spray_besteval_4b.sh <N_SHARDS> <arm...>   e.g. ... 64 base grpo gos rloo
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIE_MODEL_TAG="qwen3_4b_instruct_2507"
# resolve the local 4B-Instruct-2507 snapshot (no internet on nodes)
SNAP_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
export PIE_BASE_MODEL_HF="$(find "${SNAP_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
[ -f "${PIE_BASE_MODEL_HF}/config.json" ] || { echo "FATAL: no 4B snapshot under ${SNAP_ROOT}"; exit 2; }
export BESTEVAL_TEST_PARQUET="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet_p3/pie_gem5_test.parquet"
[ -f "${BESTEVAL_TEST_PARQUET}" ] || { echo "FATAL: missing P3 test parquet ${BESTEVAL_TEST_PARQUET}"; exit 2; }
export MAXTOK="16384"                              # match training response budget (P3 reasoning)
export PIE_NAT_TIMEOUT="5" PIE_GEM5_REWARD_NATIVE_BUDGET="30"   # gate parity with training
echo "=== 4B besteval: model=${PIE_BASE_MODEL_HF##*/snapshots/} parquet=P3 maxtok=16384 ==="
exec "${HERE}/spray_besteval.sh" "$@"
