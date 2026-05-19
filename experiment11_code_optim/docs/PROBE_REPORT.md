# PROBE_REPORT.md — §13 calibration (items 0–6)

**Updated 2026-05-19 (re-probe after CPU-allocation fix).** The original probe (kept as appendix below) ran in a 1-CPU shell — its 0.99× parallel speedup was an allocation artefact, not a kernel/Lustre issue. Re-probe was inline with 250 CPUs visible, NUMA-0 pinned via `numactl --cpunodebind=0 --membind=0`, scratch on `/tmp` (`/dev/nvme0n1` xfs, same on-node NVMe as `/local`).

**Compute node:** `gh125.hsn.cm.delta.internal.ncsa.edu` (ARM Neoverse-V2, 4× GH200, 288 cores total / 250 visible to job)
**Backend:** `perf` (perf_event_paranoid=2)
**Raw results:** `experiment11_code_optim/docs/probe_results.json`
**Driver:** `experiment11_code_optim/precompute/probe.py`

---

## Headline: **still NO_GO, but the gap is now small (1.6–2.6× of step), not 9.7×.** Plausible mitigation path to GO.

| Item | Re-probe result (NUMA-0, 250 CPUs) | Decision |
|------|-----------------------------------|----------|
| 0 — runtime boundary | conda confirmed | GO (§14 step 3) |
| 1 — `perf_event_paranoid` | =2 | GO |
| 2 — g++ at spec absolute path | 14.2.0 | GO |
| 3 — determinism | rel_spread ≈ 0 (4 instr on 51.7M); inter-ref gap 45M | **GO** |
| 4 — throughput @ 16 workers | 1.87 progs/sec, 27.7 tests/sec | (informational) |
| 4b — empirical parallel speedup | **3.84×** (saturates at 16 workers; 64 workers = no further gain) | new ceiling |
| 5 — precompute extrapolation | 5.3 h at empirical 3.84× | **GO** (≤36 h) |
| 6 — step-budget worst-case (100% pass) | reward = **18.4 min/step (2.6×)** of DeepCoder 7 min | NO_GO |
| 6b — pass-rate sensitivity | 5% pass → 11.4 min (1.6×); 20% → 12.5 min (1.8×) | still **NO_GO** at any realistic pass rate |

**Why M1 alone doesn't close the gap:** compile+gate (which every rollout pays regardless of pass/fail) is ~60% of the 1.03 s/program serial cost. Even at 5% pass rate the gate-only floor is ~11 min/step ≈ 1.6× of the 7 min DeepCoder step. We're not 9.7× away anymore, but we're not GO either.

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

## Recommendation

Implement C1 in `measurement/core.py` (precompiled-header path, optional PCH file) + a quick re-probe. Estimated work: ~1-2 hours of code + ~5 min probe re-run. If C1 alone is ≥ 3× compile speedup, we're at GO-WITH-MITIGATION; with M2 layered we're at GO. If C1 underperforms (<2×), fall back to spec-deviation discussion (G reduction, etc.).

**No further work until you confirm.** Going to §14 step 6 (full reference precompute) needs GATE B clearance, and the verdict is still NO_GO at the current measurement design.

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
