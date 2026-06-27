#!/usr/bin/env bash
# ============================================================================
# Qwen3-4B-Instruct-2507 full-batch gem5-speedup RL — thin wrapper that pins the
# 4B-specific config via env, then defers to spray_fullbatch.sh (which sprays the
# shared launcher across all accounts). sbatch propagates this exported env to
# each job (SBATCH_EXPORT defaults to ALL), and run_fullbatch_gem5.sh reads the
# PIE_* overrides — so a 1.7B run (no PIE_* set) is completely unaffected.
#
# Differences vs the 1.7B run (all confirmed in the pre-flight):
#   model     Qwen3-4B-Instruct-2507     (tag qwen3_4b_instruct_2507 -> own dirs/lock)
#   prompt    P3 step-by-step parquet    (pie_gem5_parquet_p3/, rebuilt)
#   reward    case_select=biggest        (heaviest case; matches eval; real signal)
#   response  max_resp=16384, model_len=18944
#   memory    micro_bs=1, gpu_util=0.5   (headroom for 16k-token actor fwd/bwd)
#   ckpt      save_freq=10 (<=10 steps lost), keep=2 (torn-save safe),
#             carveout every 250 + final step (eval the final ckpt)
#   timeouts  nat per-case=5s, per-rollout native budget=30s (measured src ~3.5s)
#
#   Usage: spray_fullbatch_4b.sh <grpo|gos|rloo|all> [SEED] [SEGMENTS]
#          spray_fullbatch_4b.sh all          # 3 estimators x 7 accounts = 21 jobs
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- model -----------------------------------------------------------------
export PIE_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
export PIE_MODEL_TAG="qwen3_4b_instruct_2507"

# --- data (P3) -------------------------------------------------------------
P3_ROOT="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet_p3"
export PIE_TRAIN_PARQUET="${P3_ROOT}/pie_gem5_train.parquet"
export PIE_VAL_PARQUET="${P3_ROOT}/pie_gem5_val_200.parquet"
for f in "${PIE_TRAIN_PARQUET}" "${PIE_VAL_PARQUET}"; do
  [ -f "${f}" ] || { echo "FATAL: missing P3 parquet ${f} (run build_pie_gem5_parquet.py --prompt p3 --output_root ${P3_ROOT})"; exit 2; }
done

# --- reward ----------------------------------------------------------------
export PIE_GEM5_CASE_SELECT="biggest"
export PIE_GEM5_REWARD_WORKERS="128"  # RAM-bound, not core-bound: ~1.5GB/gem5-sim, 256 OOMed (461/478GB). 128 => ~190GB reward + ~120GB train = ~310GB, safe under Ray's 454GB kill line. (reward is async-overlapped + gen-bound, so this isn't the throughput limit anyway)
export PIE_NAT_TIMEOUT="5"
export PIE_GEM5_REWARD_NATIVE_BUDGET="30"

# --- rollout / actor shapes ------------------------------------------------
export PIE_MAX_RESP="16384"
export PIE_MAX_MODEL_LEN="18944"      # = max_prompt(2560) + max_resp(16384)
export PIE_MICRO_BS="1"               # 1 seq/micro-batch: 16k-token actor fwd/bwd headroom
export PIE_REMOVE_PADDING="True"      # process actual tokens not the 18944 pad: ~2.5x faster actor + far less GPU mem (stress: rmpad=False spills over HBM)
export PIE_GPU_MEM_UTIL="0.5"         # conservative; bump only after the smoke test shows headroom

# --- checkpointing ---------------------------------------------------------
export PIE_SAVE_FREQ="10"             # rolling full-state every 10 steps (<=10 steps lost on a kill)
export PIE_KEEP_CKPT="2"              # keep 2 so a prior ckpt survives a torn save
export PIE_CARVEOUT_EVERY="250"       # permanent hf snapshot every 250 steps (+ final step, in actor_carveout)

echo "=== 4B spray config ==="
echo "  model=${PIE_MODEL_PATH} tag=${PIE_MODEL_TAG}"
echo "  data=${PIE_TRAIN_PARQUET}"
echo "  reward: case_select=${PIE_GEM5_CASE_SELECT} nat_to=${PIE_NAT_TIMEOUT}s nat_budget=${PIE_GEM5_REWARD_NATIVE_BUDGET}s"
echo "  rollout: max_resp=${PIE_MAX_RESP} max_model_len=${PIE_MAX_MODEL_LEN} micro_bs=${PIE_MICRO_BS} gpu_util=${PIE_GPU_MEM_UTIL}"
echo "  ckpt: save_freq=${PIE_SAVE_FREQ} keep=${PIE_KEEP_CKPT} carveout_every=${PIE_CARVEOUT_EVERY}(+final)"
echo "======================="

exec "${HERE}/spray_fullbatch.sh" "$@"
