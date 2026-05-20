# Step 7 — Closure Report

## Status: code-complete; SFT run + RL launch pending operator action

All code dependencies for the full RL launch (§9 GO/NO-GO → §11 multi-day
training → §8.5 post-hoc best@K eval) are in place. The remaining work is
operator-side: submit the SFT job, run the §9 GO/NO-GO check, then submit
the three RL chains.

## What landed in step 7

### Closed deferred fixes from the §6 review

| Tag | Title | Where |
|---|---|---|
| F5 | `global_timeout` enforcement | `reward/pie_reward_fn.py::grade_dedup_batch` + test |
| F6 | `PIE_MODE` sys.modules shim | `train.py` + 3 SLURM scripts + subprocess smoke test |

### New modules

| Path | Role |
|---|---|
| `sft/cold_start_sft.py` | HF Trainer + FSDP, masks prompt loss, saves full-state HF ckpt loadable by vLLM |
| `scripts/qwen3_1.7b_base_pie_sft.sh` | SLURM wrapper for the SFT job |
| `guards/isolation.py` | D1–D10 invariant assertions at training entry |
| `guards/actor_carveout.py` | Save-time hard-link mirror of verl checkpoints to `PIE_ACTOR_CARVEOUT_DIR/step_<N>/` |
| `precompute/build_pie_parquet.py` | JSONL → verl parquet, filters problems excluded by reference-table MANIFEST |
| `eval/post_hoc_eval.py` | Per-checkpoint best@K + pass@K with `eval_done.json` idempotency sentinel |

### Files modified

- `train.py` — installs (a) PIE_MODE shim and (b) actor carve-out hook before verl imports fire
- 3 PIE SLURM scripts — export `PIE_MODE=1` and `PIE_ACTOR_CARVEOUT_DIR=${ACTOR_CKPT_DIR}`
- `docs/DEFERRED_FIXES.md` — F5 and F6 → CLOSED

## How the pieces compose

```
data:  PIE jsonls
        │
        ├──> precompute/build_unified_test_cases.py  → tests/      (already done)
        ├──> precompute/build_reference_table.py     → reference_table/  (step 6, done)
        └──> precompute/build_pie_parquet.py         → parquet/    (NEW, this commit)
                                                            │
sft/cold_start_sft.py  ─── reads train_hq_only.jsonl  ──── ▼
                                                    {pie_train, pie_val_200}.parquet
                                                            │
                                                            ▼
                                                  train.py (PIE_MODE=1)
                                                  ├ shim:  experiment4_llm.src.reward_function = pie_reward_fn  ── F6 ✓
                                                  ├ hook:  actor_carveout.install_actor_carveout_hook()         ── §8.3 ✓
                                                  └ run:   experiment4_llm.src.train.main()  ─→ verl RL
                                                                                                  │
                                                                                                  ▼
                                                                   actor_ckpts/step_<N>/   (hard-linked, zero space)
                                                                                                  │
                                                                                                  ▼
                                                                          eval/post_hoc_eval.py
                                                                          best@K, pass@K at K ∈ {1,4,16,64,256}
                                                                          ── eval_done.json sentinel ─── idempotent
```

## How to run

```bash
# 1. Build the training parquets (one-time, ~30s)
PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \
    python experiment11_code_optim/precompute/build_pie_parquet.py
# Output: /work/nvme/bgmo/.../pie_data/parquet/{pie_train, pie_val_200, pie_test}.parquet
# Excluded 138 train + 4 val problems flagged by the reference table.

# 2. Submit cold-start SFT (~20-30 min wall on 4xGH200)
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_sft.sh
# Output: /work/nvme/bgmo/.../sft_ckpts/qwen3_1.7b_pie_sft_s0/final/

# 3. Once SFT is done, export and submit the three RL chains
export PIE_SFT_CKPT=/work/nvme/bgmo/.../sft_ckpts/qwen3_1.7b_pie_sft_s0/final
export PIE_STEPS_PER_EPOCH=290 PIE_TOTAL_STEPS=2900
export PIE_EVAL_STEPS='[290,580,870,1160,1450,1740,2030,2320,2610,2900]'
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_maxrl_g16.sh
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_grpo_g16.sh
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_rloo_g16.sh

# 4. After training, run the post-hoc best@K eval (idempotent; can re-run)
PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \
    python -m experiment11_code_optim.eval.post_hoc_eval \
    --carveout_dir /work/nvme/bgmo/.../qwen3_1.7b_pie_maxrl_unnorm_g16_s0/actor_ckpts \
    --val_parquet /work/nvme/bgmo/.../pie_val_200.parquet \
    --n_completions 256
```

## Parquet stats (this build)

| Split | Rows kept | Rows excluded | Output |
|---|---|---|---|
| train | 74,036 / 77,967 | 3,931 (138 problems) | `pie_train.parquet` |
| val (full) | 2,539 / 2,544 | 5 (4 problems) | `pie_val_full.parquet` |
| val (200-subsample, seed=0) | 200 | — | `pie_val_200.parquet` |
| test | 978 / 978 | 0 | `pie_test.parquet` |

Epoch arithmetic (per spec §3.1a, derived from `N_train=74036`):

- `PIE_STEPS_PER_EPOCH = ceil(74036 / 256) = 290`
- `PIE_TOTAL_STEPS = 10 * 290 = 2900`
- `PIE_EVAL_STEPS = [290, 580, ..., 2900]`

D2 of `guards/isolation.py` asserts this at training entry; DeepCoder's
inherited values (996/1000) would hard-stop the run.

## Smoke checks

- ast.parse OK on all new modules
- bash -n OK on all 4 SLURM scripts (3 RL + 1 SFT)
- best_at_k unit math sanity-checked (4 perfect + 4 zero gives best@1 = 0.5, best@8 = 1.0)
- 26/26 tests green (24 pre-existing + F5 cap + F6 shim subprocess)

## Open follow-ups (not blockers)

- §10 RL smoke step: actually launch one step under PIE_MODE=1 and assert
  `compute_score` returns non-zero on a known-good rollout (proves the
  shim is live in the verl-launched process, not just in unit tests).
- Plot generator for the metrics_step_<N>.json series (best@K vs. step).
- §8.5 best@K table in a final results doc once we have data.
