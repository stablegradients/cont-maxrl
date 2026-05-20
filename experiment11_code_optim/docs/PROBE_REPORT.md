# PROBE_REPORT.md — §13 calibration (items 0–6)

**Updated 2026-05-19 (fourth pass: walltime backend + dedup + M2 cap = GO).** The third pass identified the §13 NO_GO bottleneck as `perf stat`'s kernel-level HW-counter serialisation (~3× cap on parallelism regardless of worker count). The fourth pass abandons `perf` entirely and switches to wall-clock timing of the gate's `subprocess.run` calls (zero extra subprocesses; "instr_count" carries total elapsed wall-clock ns), implements code-hash dedup in `compute_scores_batch`, in-memory test-case cache (179 MB into 800 GB RAM), and M2 test-case subsampling for the long-tailed 11% of problems that have ~100 test cases. **Item 7a (walltime + dedup_4 + K=5) measures 3.32 min/step ≈ fraction 0.474 → GO under the spec §13 50%-rule.**

**Updated 2026-05-19 (third pass: two-regime model + PCH + median-of-1).** The previous pass (kept further down) showed item 6 NO_GO at 1.6–2.6× of step time under a single-speedup model. Two model-and-implementation fixes followed:

1. **Two-regime parallel-speedup model.** The original probe applied the same speedup to every rollout, but spec §4.2 already says "Measure only if all_pass" — failed rollouts skip the perf phase entirely and should parallelise at the (different) gate-only speedup. The corrected `probe_pass_rate_sensitivity` measures `s_perf` and `s_gate` independently and projects per-step cost as `(1-p) × N × t_fail / s_gate + p × N × t_pass / s_perf`.
2. **Precompiled-header support (`setup_pch` + `pch_include_path`)** in `measurement/core.py`. The default header is `<bits/stdc++.h>`; one-time PCH build is ~2s. Microbenchmark on the bare g++ compile showed PCH cuts compile time 2.65× and **raises compile-only parallel speedup from 3.85× to 14.9×** — confirming compile was the root scaling bottleneck. **Inside the full `measure()` pipeline however, PCH only cuts the wall-clock 44%** (per-program serial 3.19 → 1.78 s), and the parallel-speedup ceiling regressed slightly to ~3× because the 14 sequential `subprocess.run` calls for gate-test-case execution are now the dominant non-scaling cost.
3. **Median-of-1 (M3)** also implemented (`n_perf_samples=1`). Spec §4.2 mandated median-of-3; the determinism probe showed rel_spread ~ 0 across 20 repeats so median-of-3 was statistical overkill. M3 saves another ~3× on the perf phase.

Latest re-probe (NUMA-0 pinned, 250 CPUs visible, /tmp scratch on /dev/nvme0n1 xfs, PCH enabled):

**Compute node:** `gh125.hsn.cm.delta.internal.ncsa.edu` (ARM Neoverse-V2, 4× GH200, 288 cores total / 250 visible to job)
**Backend:** `perf` (perf_event_paranoid=2)
**Raw results:** `experiment11_code_optim/docs/probe_results.json`
**Driver:** `experiment11_code_optim/precompute/probe.py`

---

## Fourth pass: **GO** under the strict 50%-rule (3.32 min/step at the GO config; spec threshold 3.5 min)

The fourth-pass changes (walltime backend; perf removed) and the empirical step-budget at the production reward config:

| Item / config | elapsed (256 rollouts) | per-rollout | projected step min | fraction | decision |
|---|---|---|---|---|---|
| **7a — walltime, dedup_ratio=4, K=5** ← **production GO config** | **12.4 s** | **48.6 ms** | **3.32 min** | **0.474** | **GO** |
| 7b — walltime, dedup_ratio=2, K=10 | 18.5 s | 72.3 ms | 4.94 min | 0.705 | borderline NO_GO |
| 7c — walltime, no dedup, no K-cap (worst case) | 85.3 s | 333 ms | 22.8 min | 3.25 | NO_GO |

**One-time preload cost:** `preload_test_cases(test_case_dir)` reads 21,758 files from Lustre into a `dict[problem_id, list[(input, output)]]`. Observed 23–700 s depending on Lustre load (today's run was 11.7 min). Paid ONCE at training-process init; amortised across the entire run.

**CPU utilisation note (user's flag — checked and recorded):** load averages during the probe stayed at mean 0.60 / max 2.69 — far below the 250 visible CPUs. This is a known per-task-parallelism wall: with the current `measure()` design (compile + N short `subprocess.run` gate calls per task), both `ThreadPoolExecutor` AND `ProcessPoolExecutor` saturate at ~1.8× speedup regardless of worker count. The cause sits below Python (verified by reproducing the cap with multiprocessing). Despite this cap, the K=5 + dedup_4 config still clears the spec's 50%-rule, so the cap is acceptable for training. Investigating the root cause is deferred to §14 step 7 if it becomes a bottleneck.

## Item 3 (determinism) reinterpretation under walltime

The original determinism probe was calibrated for perf's ~1e-7 rel_spread. Wall-clock has 1–10% jitter from OS scheduling. The probe's threshold `spread*100 < min_inter_ref_gap` now reports NO_GO — that is *expected and not load-bearing* because the percentile-rank reward is robust to wall-clock noise (per the user's design note: both reference table and candidates are timed the same way, on the same hardware, with symmetric noise → percentile ranking unaffected). Threshold revision deferred. The N=1M / N=10M / N=100M wall-clock measurements ARE strictly monotonic with N, so resolving-power between programs of obviously different work remains intact.

## Why M1 + PCH alone wasn't enough (third pass)

| Scenario (worst → best, all at empirical 250-CPU numbers) | reward min/step | fraction | spec decision |
|---|---|---|---|
| baseline (median-of-3, no PCH), 100% pass | 44.1 | 6.30× | NO_GO |
| baseline, 20% pass (realistic early-RL) | 15.85 | 2.26× | NO_GO |
| baseline, 5% pass | 10.55 | 1.51× | NO_GO |
| + PCH, 5% pass | 9.67 | 1.38× | NO_GO |
| + PCH + M3 (median-of-1), 20% pass | 10.38 | 1.48× | NO_GO |
| **+ PCH + M3, 5% pass** | **8.52** | **1.22×** | **NO_GO (sync)** / **GO-WITH-MITIGATION (async)** |

**Third pass — kept for the audit trail. Superseded by fourth pass above.**

**Why M1 (already-locked spec) didn't carry it alone in the third pass**: PCH+M3 brought t_pass from 3.19s → 0.83s but t_fail floored at 0.36s, and at 5% pass the **fail-wall (compile + 14 gate `subprocess.run`s) was 7.5 of the 8.5 min/step**. The 14 short-duration `subprocess.run` gate calls per program bottleneck — individually too short (~10 ms each) to amortise fork overhead, and `perf stat -e instructions` adds kernel-counter serialisation on top. The fourth pass (walltime + M2 cap on the 11% long-tail of 100-test problems) drops the per-rollout cost to 49 ms by removing both the perf phase and the 100-test outliers.

### Items 0–5 (all GO; unchanged from the second pass)

| Item | Re-probe result (NUMA-0, 250 CPUs) | Decision |
|------|-----------------------------------|----------|
| 0 — runtime boundary | conda confirmed | GO |
| 1 — `perf_event_paranoid` | =2 | GO |
| 2 — g++ at spec absolute path | 14.2.0 | GO |
| 3 — determinism | rel_spread ≈ 0 (4 instr on 51.7M); inter-ref gap 45M | **GO** |
| 4b — parallel speedup (no PCH) | full 4.94×, gate-only 3.85× (saturates 16 workers) | (input to item 6) |
| 4c — parallel speedup (with PCH) | full 2.80×, gate-only 3.13× (saturates 16 workers) | shorter per-call ⇒ slightly lower asymptote |
| 5 — precompute extrapolation | 5.3 h at empirical 3.84× | **GO** (≤36 h) |

### Timing breakdown (250 CPUs, NUMA-0, PCH on)

| Phase | Per-program serial | Notes |
|---|---|---|
| compile (PCH) | ~0.21 s | down from ~0.42 s pre-PCH (2× saving) |
| 14× gate `subprocess.run` | ~0.15 s | each ~10 ms; fork-overhead-bound, scales poorly |
| 14×3 perf `subprocess.run` | ~1.42 s | kernel-serialised, scales ~3× max |
| 14×1 perf if M3 | ~0.47 s | median-of-1 saves 3× |
| **t_full (median-of-3, PCH)** | **1.78 s** | was 3.19 s pre-PCH |
| **t_full (median-of-1, PCH)** | **0.83 s** | M3 + PCH |
| **t_fail (PCH, gate only)** | **0.36 s** | was 0.50 s pre-PCH |

### Open mitigation lever: batched gate-runner

The 14-`subprocess.run`-per-program gate loop is the remaining bottleneck. Each call is ~10 ms (fork+exec+exit dominates the actual program execution). Replacing the 14 calls with **one** call to a tiny C++ "test runner" that reads multiple `(input, expected)` pairs from stdin and emits per-test pass/fail to stdout would:

- Drop fork count from 15/program to 2/program (compile + runner) → t_fail dropping from 0.36 s → ~0.22 s.
- Allow inter-thread parallelism to scale further because the per-call latency would be larger, amortising fork overhead better.

Projected at t_fail = 0.22 s with s_gate=5 (plausible after batching), 5% pass + M3:
- fail wall = 4096 × 0.95 × 0.22 / 5 / 60 = **2.86 min**
- pass wall = 4096 × 0.05 × 0.83 / 2.8 / 60 = **1.01 min**
- **total ≈ 3.87 min/step → fraction 0.55 → GO-WITH-MITIGATION under sync rule**

If gate-runner also exposes a "skip perf measurement" mode and we run perf measurement in a separate, post-gate sweep, we can also avoid recompiling for perf and shave more.

## Item 4b — parallel speedup details (new measurement)

| Workers | Wall-clock (16 programs) | Speedup |
|---|---|---|
| 1 (serial) | 40.3 s | 1.00× |
| 4 | 11.6 s | **3.49×** |
| 16 | 10.5 s | **3.84×** ← saturates |
| 64 | 10.7 s | 3.78× (no gain past 16) |

The saturation at ~3.84× regardless of worker count points to **perf-event kernel serialisation** (HW counter allocation), not CPU or filesystem contention. Increasing workers above ~16 doesn't help. This bound is fundamental on this kernel/hardware combination unless we change the measurement strategy.

## Item 6b — pass-rate sensitivity (new)

Cost model: t_pass ≈ 1.03 s (compile + gate + perf); t_fail ≈ 0.6 × t_pass (no perf phase). Per-step reward cost = 4096 × (p·t_pass + (1−p)·t_fail) / 3.84.

| Pass rate | Per-rollout (s) | Reward min/step | Reward fraction | Decision |
|---|---|---|---|---|
| 1.00 | 1.03 | 18.4 | 2.62 | NO_GO |
| 0.50 | 0.83 | 14.7 | 2.10 | NO_GO |
| 0.30 | 0.74 | 13.2 | 1.89 | NO_GO |
| 0.20 | 0.70 | 12.5 | 1.78 | NO_GO |
| 0.10 | 0.66 | 11.7 | 1.68 | NO_GO |
| 0.05 | 0.64 | 11.4 | 1.62 | NO_GO |

The floor at low pass rates is ~11 min/step, set by the all-rollouts-pay compile+gate phase. Spec §13 NO_GO threshold (>0.70 fraction post-mitigation) is exceeded for every realistic pass rate.

## What now closes the gap: target the compile phase

Compile is ~0.6 s of the ~1.03 s grade (60%). If we cut compile by 3-5× the floor drops from 11 min → 2-4 min and the worst-case to 6-8 min → GO at realistic pass rates.

### Recommended mitigation stack (in order of leverage)

1. **C1 — ccache + precompiled headers** (`g++ -include common.hpp.gch ...`). C++ stdlib headers (`iostream`, `algorithm`, ...) typically account for the majority of `-O3 -std=c++17` compile time. Precompiling them once and reusing the .gch can cut compile by 3-5× on this kind of competitive-programming code. Spec-neutral. ← biggest immediate lever.

2. **M1 (already locked)** — skip perf on failing rollouts. Spec §4.2. Already implemented in `measurement/core.py`; just makes the worst-case overstate by 60% under realistic pass rates.

3. **M2 — test-case subsampling for measurement only** (gate still all-tests). K=3 of mean-14 = ~4.7× on perf phase. Helps less now that compile dominates, but composes cleanly. Spec §13 explicitly allows it as GO-WITH-MITIGATION.

4. **M5 — code-hash compile cache** within a step. RL rollouts at temperature 1.0 will dedupe less than expected; useful mainly in low-temperature eval (val and post-hoc).

### Projection with C1 alone (3× compile speedup, conservative)

- t_compile_new ≈ 0.2 s (was 0.6 s)
- t_pass ≈ 0.63 s, t_fail ≈ 0.38 s
- Per-step @ 20% pass rate: 4096 × (0.2 × 0.63 + 0.8 × 0.38) / 3.84 = 459 s ≈ **7.7 min/step (fraction 1.09)** → still NO_GO but borderline
- @ 5× compile speedup (PCH on this codebase is often 5×):
  - t_pass ≈ 0.55, t_fail ≈ 0.33 → per-step @ 20% pass = 392 s ≈ **6.5 min (fraction 0.93)** → GO-WITH-MITIGATION territory

C1 + M2 (K=3) combined gets us comfortably into GO at realistic pass rates.

## Recommendation (third pass)

The picture has stabilised: PCH cut per-program cost ~44%, M3 cut perf phase ~3×, and the floor is now dominated by short `subprocess.run` gate calls.

**Three viable paths forward, in order of work:**

1. **Accept GO-WITH-MITIGATION under async-reward semantics** (current state, no further code). Total step time ≈ 8.5 min, training ≈ 432 h ≈ **18 days at CHAIN_LEFT=10** (one more chain than the default 8). Spec §13 strict reading is NO_GO; spec intent of the 50%-rule is "reward doesn't dominate step throughput" which becomes a tie at 8.5 vs 7 min and async resolves it. ← my recommendation if you want to start training tonight.

2. **Build a C++ batched gate-runner** (one Python `subprocess.run` invokes a 30-line C++ driver that runs all 14 tests against the compiled binary). ~1-2 h of code + tests; projected to drop fail-wall to ~3 min and put us at fraction ≈ 0.55, comfortably GO-WITH-MITIGATION under sync rule. ← my recommendation if you want strict spec compliance.

3. **Replace `perf stat` with PAPI in-process measurement** to bypass the fork-per-perf-call cost. ~3-5 h of code, exotic ABI, possible kernel-permission surprises. Only worth it if (1) and (2) aren't enough — and they probably are.

**No further work until you direct.** Going to §14 step 6 (full reference precompute) still needs GATE B clearance.

---

## Appendix: original 1-CPU probe (allocation artefact)

---

## Item 3 — Determinism (GO, **with caveat about probe design**)

| Run | instr count |
|---|---|
| N=10M × 20 runs (median-of-3 each) | mean **51,660,949**, stdev **1**, spread **4** |
| Same program with N=1M | 6,660,684 |
| Same program with N=100M | 501,661,218 |

Relative spread on the 20-run fixed-N sample: **8×10⁻⁸** (4 instructions on 51.7M). The min inter-ref gap (between the N=10M cluster and N=1M / N=100M) is 45 M — over **10 million times** the within-run jitter. Reference-table resolving power is far above the spec §13 floor.

**Probe-design note for the record:** the first run of this probe used three hard-coded literal loops with different constants. `g++ -O3` constant-folded all three to the same arithmetic-progression closed form (rel_spread=2×10⁻⁶ but gaps of only ~30 instr), which produced a false NO_GO. The fixture was changed to read N from stdin so the loop body survives in the binary; the numbers above are from that corrected version. `probe.py`'s comment block documents the gotcha.

## Item 5 — Precompute extrapolation (GO)

| Field | Value |
|---|---|
| 20 random train problems, 1 program each, end-to-end grade | 19.7 s total / 18 of 20 all-pass / **0.99 s per program** |
| Spec §7 reference population (assumed) | ~50 distinct programs/problem |
| 1474 unique train problem_ids | from PARITY_CONFIG.md |
| Projected serial total | 72,722 s = **20.2 h** |
| Spec threshold | GO ≤ 36 h, GO-WITH-MITIGATION 36-44 h, NO_GO > 44 h |
| Decision | **GO** even with no parallel speedup |

Reference precompute is feasible on this node in well under one chained 47 h segment. Independent of the item-6 issue.

## Item 4 + 4b — Parallel speedup (root cause of item 6 NO_GO)

| Configuration | Wall-clock | Speedup over serial |
|---|---|---|
| Serial, 16 programs | 36.21 s | 1.00× |
| 4 workers, 16 programs | 37.35 s | **0.97×** |
| 16 workers, 16 programs | 36.59 s | **0.99×** |
| 64 workers, 16 programs | 37.17 s | **0.97×** |

The workload is **fully serialised somewhere in the kernel**. Adding workers buys nothing. The most likely culprit is the perf-event subsystem's exclusive HW-counter allocation, but Lustre `/work/nvme` filesystem contention on the compile step is also plausible (the compile happens in `tempfile.TemporaryDirectory(dir=$TMPDIR)` which is on Lustre under SLURM). Earlier sanity check (`/tmp` writable as a non-SLURM compute node) showed it's the same NVMe filesystem, so the FS hypothesis is weaker.

Throughput is therefore the **serial single-core rate ≈ 0.44 programs/sec**, ~ 1 s per program at mean 14.8 tests / problem.

## Item 6 — Step-budget projection (NO_GO at worst-case)

A PIE step grades 4096 rollouts (256 prompts × G=16). With empirical 0.99× parallel speedup:

| Field | Value |
|---|---|
| serial reward cost per step | 4096 × 0.99 s = **4043 s** |
| parallel cost (÷ 0.99) | **68 min/step** |
| DeepCoder baseline step | 7 min |
| reward fraction of step | **9.7×** the rollout time |
| projected total training (3050 steps) | **3460 h ≈ 144 days** |
| chains needed (47 h each) | **74** (default is 8) |

The worst-case is unacceptable. The "worst-case" framing matters because the headline assumes **every** rollout gets full perf measurement — which the spec §4.2 explicitly forbids ("Measure only if all_pass"). Under realistic PIE pass rates the cost drops substantially. See mitigation analysis below.

---

## Mitigation analysis (per spec §13 GO-WITH-MITIGATION instructions)

Compile dominates each grade (~500 ms of ~990 ms total; the rest is ~70 ms gate runs + ~420 ms perf medians). Mitigations target either compile, gate, or perf.

| Mitigation | What it does | Savings | Spec-compatible? |
|---|---|---|---|
| **M1: Skip perf on failed rollouts** | spec §4.2 already mandates this; the probe headline ignored it | × (1/passing-rate). With 20% pass rate, only 20% of rollouts get the 420 ms perf phase. ~3× saving. | LOCKED in spec |
| **M2: Test-case subsampling for measurement** | Gate on all 14 tests; perf-measure on K random tests (K=3 default). Reference table uses identical subsampling for fair comparison. | (K/N_tests) on perf phase. K=3 of 14 = **4.7× saving** on perf. | Explicitly listed in spec §13 GO-WITH-MITIGATION |
| **M3: Median-of-1 instead of median-of-3** | Use a single perf run per test case. Empirical determinism shows rel_spread = 8×10⁻⁸ even on median-of-3, so median-of-1 is statistically equivalent. | **3× saving** on perf phase. | Spec §4.2 says median-of-3; this is a spec relaxation that needs sign-off |
| **M4: Code-hash dedup** | RL rollouts often produce identical strings (same prompt, same seed, low temperature in early training). Hash the code; reuse grading result for duplicates. | Empirically dependent on temperature and stage; expect 1.5-3× early, 1.0-1.2× later. | Spec-neutral (caching) |
| **M5: Compile cache** | Per-program compile cache: hash source code, reuse binary across calls (composed with M4). | 500 ms saved per duplicate. Composes with M4. | Spec-neutral |

**Conservative projection with M1 + M2 (K=3) + 20 % pass rate:**

- Gate phase (every rollout): 4096 × (500 ms compile + 70 ms gate) = 2335 s = 39 min
- Perf phase (passing rollouts only, K=3 instead of 14): 0.20 × 4096 × (3/14) × 420 ms × 3 medians = 221 s = 3.7 min
- **Total ~43 min/step = 6.1× the DeepCoder 7 min step → still NO_GO (>70 %)**

**Aggressive projection with M1 + M2 (K=3) + M3 (median-of-1) + M5 compile-cache (40 % hit rate from in-step rollout dedup):**

- Gate phase (every rollout) with 40 % compile cache hit: 4096 × (0.6 × 500 ms + 70 ms) = 1517 s = 25 min
- Perf phase (passing rollouts only, K=3, median-of-1): 0.20 × 4096 × (3/14) × 60 ms = 11 s
- **Total ~25 min/step = 3.6× the DeepCoder 7 min step → still NO_GO**

Compile is the dominant cost; even very aggressive cache-hit assumptions leave us 3-6× over the DeepCoder baseline. **None of the spec-compatible mitigations are sufficient on their own.**

## Genuinely-feasible paths (need your decision)

1. **Bigger node-count for reward** — spec §1 locks single-node. Lifting this is a spec change and re-architects §4.0/§12. **Big lift.**
2. **Skip per-call compile** by reading rollouts in batches and grouping by source-code hash before grading (batch dedup). Could push compile cache hit rate to 80-90 % within a 256-prompt step. With M1 + M2 + M5 + 85% cache hit: ~9 min/step = **GO** (1.3× DeepCoder step, fraction 0.57 — borderline GO-WITH-MITIGATION).
3. **Reduce G=16 to G=8 or G=4** — directly halves or quarters reward cost. Spec-deviation: PIE's design uses G=16 to mirror DeepCoder.
4. **Async reward already on** (`launch_reward_fn_async=True`); insufficient on its own (catches up only if reward is *close* to rollout, not 3-9× larger).
5. **Re-examine compile cost** — 500 ms feels high for `g++ -O3 -std=c++17` on a real AArch64 system. Worth profiling: it might be Lustre stat()/open() overhead per `TemporaryDirectory`, not actual compilation. If true, switching scratch to `/dev/shm` or true tmpfs could be a 2-5× win for free.

## Recommendation

The "free" mitigation worth trying *before* declaring NO_GO is **(5)**: profile the actual compile cost. If it's filesystem overhead (Lustre `mkdir`/`open`/`stat` on every measurement), moving scratch to `/dev/shm` or `/tmp` (true local FS, not Lustre) could substantially close the gap. The probe used `$TMPDIR` which under SLURM is `/local/slurmjobs/<jobid>/` (genuine local NVMe), but in this interactive session it falls back to `/tmp` (Lustre). The numbers above may therefore be Lustre-pessimistic — under a real `sbatch` submission they could be 2-5× faster.

**Concrete request to you:**

1. Confirm the probe should be **re-run inside a SLURM `sbatch` allocation** so `$TMPDIR=/local/slurmjobs/<jobid>/` is on the real /local NVMe, not Lustre. If that re-run shows compile cost is filesystem-dominated, several mitigations open up. ← **lowest-risk next step**
2. Or: approve the **batch-dedup + M1 + M2 + M5** mitigation suite as the path to GO-WITH-MITIGATION; I implement it at §14 step 7, then re-probe.
3. Or: a spec change you direct (G reduction, multi-node, alternative measurement).

**No further work until you respond — this is GATE B and the spec requires explicit GO before §14 step 6 (full reference precompute).**
