# experiment11_code_optim — PIE Instruction-Count Percentile Reward

RL-tunes a small LM (Qwen3-1.7B-Base) to **rewrite C++ programs to run faster**.
The reward is a *midrank percentile* of measured wall-clock cost against a
reference distribution of human-Accepted solutions for the same problem.

Implements **SPEC v9 (FINAL): PIE Instruction-Count Percentile Reward — GOS vs
GRPO/RLOO on Code Optimization** on a single 4×GH200 node.

The authoritative source of behavior is the SPEC v9 prompt; this README is the
operator's guide.

---

## Task

Each training example is a pair `(src_code, tgt_code)` where both compile and
solve the same competitive-programming problem; `tgt_code` is a faster
human-written rewrite of `src_code`. The model sees the prompt

```
You are a C++ optimization expert. Below is a working C++ program.
Rewrite it to run faster while producing identical output for all valid
inputs. Keep the same input/output format and only change the algorithm/
implementation.

### Slow Version:
```cpp
{src_code}
```

### Optimized Version:
```

and must complete with ```cpp ... ```. The reward grader extracts the
fenced block, compiles it with `g++ -O3 -std=c++17`, runs it on a
K=5-subsampled test suite for that `problem_id`, and on **strict all-pass**
emits the elapsed-ns midrank percentile.

---

## Reward math (§8.2)

Per rollout:

```
1.  extract_code(completion)          → C++ source (or "")
2.  compile (g++ -O3 -std=c++17)      → exe or compile-fail
3.  strict gate: run all K test cases → pass/fail per test
4.  if not all_pass:                  → score = 0
5.  else cost = Σ elapsed_ns          → score = midrank percentile vs reference
```

Midrank percentile (`_percentile_from_reference_table`):

```
refs        = sorted elapsed_ns values from precompute (same problem_id, same split)
lo          = bisect_left(refs, cost)
hi          = bisect_right(refs, cost)
score       = 1 − (lo + hi) / 2 / len(refs)
acc         = 1.0  if all_pass else  0.0
```

Lower cost ⇒ higher score ⇒ `score=1.0` when the candidate beats every
reference; `score=0.0` when slower than every reference; `score≈0.5` when
matching the median reference. `acc` is the strict binary outcome.

---

## RL pipeline schematic

```
                        ┌──────────────────────────────────────────────────┐
                        │   precompute/build_unified_test_cases.py         │
                        │   precompute/build_reference_table.py            │
                        │   precompute/build_pie_parquet.py                │
                        └─────┬──────────────┬──────────────┬──────────────┘
                              ▼              ▼              ▼
                       tests.pkl    reference_table.pkl   pie_*.parquet
                        21,758       1,592 distributions     verl
                        cases         (sorted ns lists)      schema
                              │              │              │
                              └──────┬───────┴──────────────┤
                                     ▼                      ▼
                              ┌─────────────────────────────────────────┐
                              │  sft/cold_start_sft.py (HF + FSDP)      │
                              │   trains 1 epoch on full train.jsonl    │
                              │   → PIE_SFT_CKPT (FSDP-merged HF dir)   │
                              └────────────────┬────────────────────────┘
                                               ▼
            §9 GO/NO-GO gate (eval/go_no_go.py): generate 16/problem on
            pie_val_200; require ≥ 20 % problems with ≥ 8/16 strict-gate
            passes. If GO → proceed to RL.
                                               ▼
            ┌──────────────────────────────────────────────────────────┐
            │  scripts/qwen3_1.7b_base_pie_{maxrl,grpo,rloo}_g16.sh    │
            │   • PIE_MODE=1 → train.py installs sys.modules shim so   │
            │      experiment4_llm.src.reward_function = pie_reward_fn │
            │   • guards.actor_carveout installs save-hook hard-links  │
            │     every verl ckpt to PIE_ACTOR_CARVEOUT_DIR/step_<N>/  │
            │   • verl + FSDP + vLLM rollouts, three estimators differ │
            │     only by algorithm.adv_estimator (spec §3.3)          │
            │   • custom_reward_function.path = reward/pie_reward_fn.py│
            └────────────────┬─────────────────────────────────────────┘
                             ▼
            reward/pie_reward_fn.compute_scores_batch (per-step, 256 × 16)
              Phase 1: extract + dedup-by-(code, problem_id)
              Phase 2: two-level scheduler
                • 2a/b: parallel compile (one mkdtemp per unique code)
                • 2c/d: parallel test runs (scheduling unit = ONE test)
              Phase 3: scatter → midrank percentile
            (~5 s/step at 256 × 16 = 4096 rollouts, dedup ~16x, K=5 cap)
                             │
                             ▼
            ┌──────────────────────────────────────────────────────────┐
            │  eval/post_hoc_eval.py (after RL)                        │
            │   walks actor_ckpts/step_<N>/ snapshots, vLLM-generates  │
            │   N completions/problem on pie_val_200, computes         │
            │   best@K and pass@K at K ∈ {1, 4, 16, 64, 256}.          │
            │   Idempotent via eval_done.json sentinel.                │
            └──────────────────────────────────────────────────────────┘
```

---

## Dataset

Raw JSONLs at `/work/nvme/bgmo/sramasubramanian/pie_data/raw/`:

| File | Rows | Use |
|---|---:|---|
| `train.jsonl` | 77,967 | RL training source (filtered to 73,893 after token filter + excluded problems) |
| `train_hq_only.jsonl` | 4,085 | High-quality SFT subset |
| `val.jsonl` | 2,544 | §9 GO/NO-GO + §8.5 best@K (subsampled to 200) |
| `test.jsonl` | 978 | Final hold-out (used by §11 results table only) |

Schema (LearningOpt/pie):

```
{"problem_id": "p02587",
 "src_code":   "<slow C++ that compiles and passes the testsuite>",
 "tgt_code":   "<fast C++ that compiles and passes the testsuite>",
 "src_status": "Accepted",   "tgt_status": "Accepted",
 "src_agg_runtime": <ms>,    "tgt_agg_runtime": <ms>,
 "speedup": <float>,
 ... (other meta)}
```

Derived artefacts at the same data root:

```
tests/                — unified test-case archive built by precompute/build_unified_test_cases.py
                        1,592 problem dirs, each {input.<i>.txt, output.<i>.txt} pairs,
                        21,758 cases total (Generated > Public > CodeNet precedence)
tests.pkl             — pickle of {problem_id: [(input, output)]} (1.5 MB) for 5-ms RAM-load
reference_table/      — per-problem percentile distributions built by build_reference_table.py
                        1,592 JSON files, each:
                            {"problem_id", "split", "n_programs_attempted", "n_programs_passed",
                             "n_programs_compile_failed", "n_programs_gate_failed",
                             "elapsed_ns_values": [sorted ascending]}
reference_table.pkl   — pickle of {(split, problem_id): [sorted ns]} (130 KB) for 1-ms RAM-load
parquet/              — verl-compatible parquets (pie_train, pie_val_full, pie_val_200, pie_test)
                        {data_source: "pie", prompt: [{user, content}], reward_model: {ground_truth}}
```

---

## File map

```
experiment11_code_optim/
├── README.md                          ← this file
├── PARITY_CONFIG.md                   D1-D10 drift items vs DeepCoder
├── SECURITY_NOTE.md                   W&B key sourcing
├── train.py                           2-line wrapper + PIE_MODE shim + actor carveout hook
│
├── reward/
│   └── pie_reward_fn.py               compute_score(s)_batch, grade_dedup_batch,
│                                      preload_test_cases, preload_reference_table,
│                                      _percentile_from_reference_table (midrank).
│
├── measurement/
│   ├── core.py                        measure(), measure_from_disk(), setup_pch(),
│                                      MEASUREMENT_BACKEND="walltime", _run_and_compare.
│   └── gate_runner.sh                 batched-disk gate (precompute path).
│
├── precompute/
│   ├── build_unified_test_cases.py    Generated > Public > CodeNet → tests/ + tests.pkl
│   ├── build_reference_table.py       grade all distinct Accepted programs → reference_table/ + .pkl
│   ├── build_pie_parquet.py           JSONL → verl parquet (token filter, excluded-problem filter)
│   └── probe.py                       §13 GO/NO-GO probe (perf vs walltime; speed-fix calibration)
│
├── sft/
│   └── cold_start_sft.py              HF Trainer + FSDP. max_length=8192 default.
│
├── eval/
│   ├── go_no_go.py                    §9 gate eval (G=16, T=1.0, ≥20% with ≥8/16)
│   ├── aggregate_go_no_go_shards.py   merge 4-shard data-parallel JSONs
│   └── post_hoc_eval.py               §8.5 best@K + pass@K with eval_done sentinel
│
├── guards/
│   ├── isolation.py                   D1-D10 invariants; run_all_guards() entrypoint
│   └── actor_carveout.py              monkey-patches verl FSDPCheckpointManager.save_checkpoint
│                                      to hard-link every save into $PIE_ACTOR_CARVEOUT_DIR
│
├── scripts/
│   ├── qwen3_1.7b_base_pie_sft.sh     SLURM SFT (1 epoch on full train.jsonl by default)
│   ├── qwen3_1.7b_base_pie_maxrl_g16.sh   RL — GOS (algorithm.adv_estimator=maxrl_unnorm)
│   ├── qwen3_1.7b_base_pie_grpo_g16.sh    RL — GRPO (algorithm.adv_estimator=grpo, verl-native)
│   └── qwen3_1.7b_base_pie_rloo_g16.sh    RL — RLOO (CUSTOM_ESTIMATORS["rloo"])
│
├── tests/                             26 pytest tests covering measurement core, reward fn,
│                                      F5 timeout, F6 PIE_MODE shim, GATE C regression
│
└── docs/
    ├── DEFERRED_FIXES.md              F1-F8 tracker (all closed)
    ├── PROBE_REPORT.md                §13 walltime decision
    ├── REFERENCE_TABLE_REPORT.md      §14 step 6 closure / GATE C
    ├── STEP7_REPORT.md                §14 step 7 closure (parquet, carve-out, best@K)
    └── PROVENANCE.md                  data lineage
```

---

## How to run

Prerequisites are once-only and idempotent:

```bash
# 0. Sanity env exports (uses pickle preloads; bgmo* account for D1)
set -a; source .env; set +a

# 1. Build the unified test-case archive (~2 min, idempotent)
PYTHONPATH=. python experiment11_code_optim/precompute/build_unified_test_cases.py

# 2. Build the reference table (~4 min)
PYTHONPATH=. python experiment11_code_optim/precompute/build_reference_table.py

# 3. Build training parquets (~2 min)
PYTHONPATH=. python experiment11_code_optim/precompute/build_pie_parquet.py

# 4. SFT cold-start (~52 min on 4×GH200, 1 epoch full train.jsonl)
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_sft.sh

# 5. § 9 GO/NO-GO on the SFT checkpoint (4-shard, ~4 min)
#    Need ≥ 20% of 200 val problems with ≥ 8/16 strict passes.
PYTHONPATH=. python -m experiment11_code_optim.eval.go_no_go ...

# 6. RL (3 chained jobs × 3 epochs × 289 steps = 867 steps total, days of wallclock)
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_maxrl_g16.sh
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_grpo_g16.sh
sbatch experiment11_code_optim/scripts/qwen3_1.7b_base_pie_rloo_g16.sh

# 7. Post-hoc best@K eval on each checkpoint (idempotent)
PYTHONPATH=. python -m experiment11_code_optim.eval.post_hoc_eval ...
```

---

## Required environment variables

`.env` (loaded via `set -a; source .env; set +a` — see `SECURITY_NOTE.md`):

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
```

SLURM scripts default these but accept overrides:

| Variable | Default | Meaning |
|---|---|---|
| `PIE_SFT_CKPT` | `/work/nvme/.../qwen3_1.7b_pie_sft_1ep_fulltrain_s3/final` | actor model dir |
| `PIE_STEPS_PER_EPOCH` | `289` | `ceil(73,893 / 256)` |
| `PIE_TOTAL_STEPS` | `867` | `3 × 289` |
| `PIE_EVAL_STEPS` | `[289,578,867]` | in-situ eval at each epoch boundary |
| `PIE_MEASUREMENT_MAX_TESTS` | `5` | K-cap; reward fn subsamples deterministically |
| `PIE_PCH_INCLUDE_PATH` | `$(setup_pch)` | per-script auto-built precompiled header |
| `PIE_COMPILE_SCRATCH` | `${TMPDIR}` | local NVMe path for per-rollout tempdirs |
| `PIE_TEST_CASE_PICKLE` | `/work/nvme/.../tests.pkl` | RAM-load test cases |
| `PIE_REFERENCE_TABLE_PICKLE` | `/work/nvme/.../reference_table.pkl` | RAM-load reference distributions |
| `PIE_MODE` | `1` | install sys.modules shim so insitu_eval grades C++ via pie_reward_fn |
| `PIE_ACTOR_CARVEOUT_DIR` | `${OUTPUT_DIR}/actor_ckpts` | per-step hard-linked checkpoint snapshots |
| `VERL_REWARD_WORKERS` | `64` | ThreadPoolExecutor size for two-level scheduler |

---

## Status (latest)

| Step | Status |
|---|---|
| §13 probe (GO config) | GO at 3.32 min/step pre-speedfix; 0.25 min/step post-speedfix |
| §14 step 6 reference table | GATE C PASS — 91.1% problem coverage |
| §14 step 7 (SFT + guards + eval) | code complete |
| § 9 GO/NO-GO | **GO** at **28.0%** (≥8/16) on 1ep full-train SFT |
| §11 RL launch | ready (scripts have GO defaults baked in) |

See `docs/STEP7_REPORT.md` for the closure write-up and trajectory across the
SFT escalations.
