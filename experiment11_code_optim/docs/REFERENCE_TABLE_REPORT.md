# Reference Table — §14 Step 6 Report (GATE C)

## Result: **GATE C PASS**

The reference-table precompute completed cleanly, the
percentile-rank lookup round-trips through `compute_scores_batch` on a
known-good program (score 1.0 on `p02587`'s val tgt_code), and both the
existing 22 tests plus the two new regression tests added in this commit
pass.

## Numbers

- **23,339 distinct Accepted programs graded** across 1,592 problems
- **Wall-clock: 199.9 s** (3.3 min) at **116.7 grades/s**
  - Beats the projected 67 grades/s — pure-unique batches have no
    Phase-1 dedup overhead, so the scheduler runs even leaner than the
    RL hot path.
- **Output:** `/work/nvme/bgmo/sramasubramanian/pie_data/reference_table/`
  - `MANIFEST.json` (schema_version=1, coverage stats, excluded list)
  - `<split>/<problem_id>.json` (one per problem)

Step 6 finished in **~5 minutes wall-clock** on the compute node
(setup 24s + grade 200s + write 1s) — over an order of magnitude faster
than the original 45-90 min estimate, thanks to the two-level
scheduler from commit `938f872`.

## Coverage

| Split | Problems | ≥1 passing reference | Programs attempted | Programs passed |
|---|---|---|---|---|
| train | 1474 | 1336 (**90.6 %**) | 21,970 | 19,262 (87.7 %) |
| val | 77 | 73 (**94.8 %**) | 812 | 758 (93.3 %) |
| test | 41 | 41 (**100 %**) | 557 | 543 (97.5 %) |
| **Total** | **1592** | **1450 (91.1 %)** | **23,339** | **20,563 (88.1 %)** |

**142 problems are excluded** (zero Accepted programs passed our strict
gate). 138 in train, 4 in val, 0 in test. At runtime,
`_percentile_from_reference_table` falls back to the stub 0.5 score when
`elapsed_ns_values` is empty — so excluded problems contribute a
flat-zero gradient signal but do not crash training. These problems
should ideally be filtered from `data.train_files`; a follow-up will add
the filter (tracked below).

### Failure-mode breakdown

| Bucket | train | val | test |
|---|---|---|---|
| Compile-failed programs | 627 (2.9 %) | 20 (2.5 %) | 7 (1.3 %) |
| Gate-failed programs | 2081 (9.5 %) | 34 (4.2 %) | 7 (1.3 %) |

Compile failures are dominated by older C++ dialects vs. our strict
`-O3 -std=c++17` (e.g., implicit `void main()`, missing `<cstdlib>`
for C-style I/O). Gate failures are mostly judging-convention mismatches
(precision-tolerance comparators, multiple-valid-output problems, or
problems where the K=5 subsampling lands on a 6s-blowing test) — the
strict equality gate rightly rejects these.

### Reference depth distribution (passing problems only)

| # passing refs | # problems | Note |
|---|---|---|
| 1 | 328 | Percentile is 3-valued {0, 0.5, 1.0} — coarse but usable |
| 2–5 | 472 | Modest resolution |
| 6–10 | 205 | Good |
| 11–25 | 243 | Strong |
| 26–50 | 120 | Strong |
| 51–200 | 76 | Saturated |

About 23% of passing problems have only one reference; the binary signal
they provide is still load-bearing — and the user is welcome to filter
these out later if they cause training noise.

### elapsed_ns ranges (sanity)

- Per-problem min (fastest known-good program): median **30.5 ms** range 3–3311 ms
- Per-problem max (slowest known-good program): median **53.5 ms**

The typical PIE problem has its solutions cluster in the 30–60 ms band
under K=5 cap + walltime backend on the GH200 node. The 3311 ms outlier
is a compute-heavy problem; this is fine — the gate's 6 s per-test
timeout is still 1.8× the worst observed.

## GATE C smoke test

```
[ref] p02587: 6/8 passed; elapsed_ns range [370916288, 577959906]
[reward] result = {'score': 1.0, 'acc': 1.0}
[GATE C] PASS: acc=1.0, score=1.000 (in (0, 1])
```

The known-Accepted val.jsonl `tgt_code` for `p02587` lands **at the top
of its reference distribution** (faster than every other Accepted
solution we graded — `score=1.0`). This makes sense: PIE's `tgt_code`
is the *optimized* version of `src_code`, so it should sit near the top.

## Regression tests added in this commit

1. `tests/test_pie_reward_fn.py::test_compute_scores_batch_with_reference_table_gives_nonzero_percentile`
   — permanent GATE C: known-good code through the full pipeline must
   return a percentile in (0, 1]. Skipped if the reference table is not
   built.
2. `tests/test_measurement.py::test_non_utf8_output_does_not_raise`
   — a program emitting `0xff` to stdout must not raise
   `UnicodeDecodeError`; the gate rejects it as output-mismatch. Real
   `train.jsonl` program tripped this on the first full precompute run;
   `subprocess.run(text=True, errors='replace')` is now load-bearing.

## Reference-table schema (v1)

```json
{
    "problem_id": "p02587",
    "split": "val",
    "n_programs_attempted": 8,
    "n_programs_passed": 6,
    "n_programs_compile_failed": 2,
    "n_programs_gate_failed": 0,
    "elapsed_ns_values": [370916288, 380453192, ..., 577959906]
}
```

`elapsed_ns_values` is sorted ascending; `bisect` gives midrank percentile
via `_percentile_from_reference_table` in `pie_reward_fn.py`.

## Things deferred from this step

- **F5** (step 7): `global_timeout` enforcement around the batch pool.
- **F6** (step 7): `PIE_MODE` `sys.modules` shim for `insitu_eval`.
- **Filter excluded problems from train_files** (small follow-up; today
  these still emit zero gradient and just waste rollouts).

See `docs/DEFERRED_FIXES.md` for the full tracker.

## Files touched this step

- `experiment11_code_optim/precompute/build_reference_table.py` (NEW, 233 lines)
- `experiment11_code_optim/reward/pie_reward_fn.py` (extract `grade_dedup_batch`; F1 rename)
- `experiment11_code_optim/measurement/core.py` (F1 rename; `errors='replace'` fix)
- `experiment11_code_optim/precompute/probe.py` (F1 rename)
- `experiment11_code_optim/tests/test_measurement.py` (F1 rename; non-UTF-8 regression test; bound widen)
- `experiment11_code_optim/tests/test_measurement_integration.py` (F1 rename)
- `experiment11_code_optim/tests/test_pie_reward_fn.py` (GATE C regression test)
- `experiment11_code_optim/docs/DEFERRED_FIXES.md` (F1 → CLOSED)
- `experiment11_code_optim/docs/REFERENCE_TABLE_REPORT.md` (NEW; this file)

## Output artifacts (NOT committed — too large)

Stored on `/work/nvme/bgmo/sramasubramanian/pie_data/reference_table/`:
- 1592 per-problem `<split>/<pid>.json` files
- 1 `MANIFEST.json`
- Total size: ~few MB, ~20 KB per problem on average
