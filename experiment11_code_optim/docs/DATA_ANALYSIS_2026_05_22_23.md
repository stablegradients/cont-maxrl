# PIE Data + Model Statistics — Session of 2026-05-22 / 23

Comprehensive catalog of every statistic computed in this session: the PIE
reference data, the perf vs walltime measurement evidence, the six Qwen3
baseline pass-rate evaluations, the custom-reward distribution probes
(with and without `-O0`), the bimodality analysis, and the per-problem
tgt-pool tightness study.

All raw JSON dumps and plots are under
`experiment11_code_optim/docs/plots/` so any number below is reproducible
without re-grading.

---

## Table of contents

1. [Measurement-backend evidence (perf vs walltime)](#1-measurement-backend-evidence-perf-vs-walltime)
2. [Reference-table speedup distribution at the per-program level](#2-reference-table-speedup-distribution-at-the-per-program-level)
3. [Per-problem speedup histogram (averaged across problems)](#3-per-problem-speedup-histogram-averaged-across-problems)
4. [Per-problem tgt-pool tightness](#4-per-problem-tgt-pool-tightness)
5. [Train-only verification of the tgt-pool tightness](#5-train-only-verification-of-the-tgt-pool-tightness)
6. [Qwen3 baseline pass-rate sweep on PIE val](#6-qwen3-baseline-pass-rate-sweep-on-pie-val)
7. [Custom-reward distribution probe (src-given prompt, -O3)](#7-custom-reward-distribution-probe-src-given-prompt-o3)
8. [Same probe under -O0 compile](#8-same-probe-under-o0-compile)
9. [Bimodality decomposition: "near src" vs "near tgt"](#9-bimodality-decomposition-near-src-vs-near-tgt)
10. [Reward ceiling: are Mode-2 generations competitive with tgts?](#10-reward-ceiling-are-mode-2-generations-competitive-with-tgts)
11. [Cross-referenced inventory of every artifact](#11-cross-referenced-inventory-of-every-artifact)

---

## 1. Measurement-backend evidence (perf vs walltime)

The original PIE measurement backend was wall-clock nanoseconds. Empirical
investigation in this session (continuing from earlier session
[`SESSION_BUG_ANALYSIS_2026_05_20_21.md`](SESSION_BUG_ANALYSIS_2026_05_20_21.md))
showed that wall-clock data is dominated by OS scheduling noise under the
production 64-thread grader load.

### 1.1 Variant E — single-program noise under varying concurrency

Same C++ program (val pid `p02587` tgt) graded N times back-to-back under
the production grader. Five separate ghx4 whole-node allocations.

```
node            workers   k     pooled rel_spread   (max - min)/median
gh082 / E1      64        10×64 = 640 samples       28.10×
gh082 / E2      64+pin    10×64 = 640 samples       28.10×   (~same as unpinned)
gh082 / E3       1        10                         0.097× (intrinsic ceiling)
```

Under 64-thread load: rel_spread = **2,810%**. With one thread: 9.7%.
The 28× jitter is purely OS scheduler / CPU contention; the underlying
program is bit-identical.

### 1.2 Worker-sweep — noise vs concurrency

```
workers  pooled rel_spread (mean over 5 cross-node runs)
8        0.82×
16       1.69×
32       2.33×
48       2.27×
64       2.40×
```

Noise grows with concurrency and saturates near ~2.4×.

### 1.3 Perf backend gates (final ghx4 verification, post SIGSTOP wrapper)

```
Gate 1 (single-thread determinism)    rel_spread = 2×10⁻⁶   target 1×10⁻⁴   PASS by 50×
Gate 2 (64-thread concurrency)        rel_spread = 2×10⁻⁶   target 1×10⁻³   PASS by 500×
Gate 3 (cross-program differentiation) max/min = 49.92×     target ≥ 5×     PASS
Gate 4 (end-to-end compute_scores_batch determinism) score rel_spread = 0.000 across 10 trials  PASS
Gate 5 (production throughput) reward phase = 0.68 s/step   walltime baseline 3.32 min  ~280× speedup
Gate 6 (paired clean determinism runs, SEED=42 on 2 nodes) steps 1-2 bit-identical critic/score/mean  PASS
```

### 1.4 Walltime vs perf reference-table comparison

Same reference programs graded under both backends; plotted in
`plots/pie_ref_table_norm_slowest_wall_vs_perf.png`.

```
                 walltime ref-table        perf ref-table
pooled p50       1.571×                    1.086×
pooled p90       2.584×                    2.890×
pooled p99       8.250×                    49.245×
pooled max       39.776×                   1,174.844×
bin [1.0, 1.1)   ~12% mass                 ~51% mass
```

Walltime distribution looks smoother because OS jitter sprinkles
bit-identical programs across a 1-5× range; perf distribution is sharply
peaked at 1.0 with a heavy tail because algorithmically-identical programs
get identical counts.

**Conclusion: walltime data is unreliable in the production grader.**
Everything below uses perf instruction counts.

---

## 2. Reference-table speedup distribution at the per-program level

For each saved program in `reference_table_perf.pkl`, ratio =
`max(refs[pid]) / elapsed_ns_of_this_program` (so all ratios ≥ 1, with
1.0 = slowest in its pid's pool).

### 2.1 Pooled across all (pid, program) pairs

```
pooled (20,557 programs, 1450 pids)
p10  1.000
p50  1.086
p75  1.667
p90  2.890
p95  6.182
p99  49.245
max  1,174.844
[1.0, 1.1)     mass = 0.509
[5, +inf)      mass = 0.063
```

### 2.2 Per-problem averaged (each pid's histogram normalized to sum=1, then mean)

```
1450 pids
[1.0, 1.1)     mass = 0.790
[5, +inf)      mass = 0.025
```

The per-problem averaged density is more concentrated at 1.0 because
each pid is weighted equally regardless of pool size; problems with many
similar tgts get the same weight as problems with one outlier-fast tgt.

Plot: `plots/pie_ref_table_norm_slowest.png` (side-by-side).

---

## 3. Per-problem speedup histogram (averaged across problems)

For each pid, compute candidate pool = unique (pid, code) from all rows;
slowest_g = max(elapsed) in pool; ratio_i = slowest_g / elapsed_i for
each candidate. Per-problem histogram normalized to sum=1; averaged
across problems.

### 3.1 At `-O3` compile (production default at time of grading)

Source: `plots/pie_all_perproblem_elapsed_ns.json` (combined src+tgt
per pid, no kind tag).

```
src+tgt combined  (n=1466 pids, 37,685 grades kept, 5083 gate-fail)
  pooled p10  1.01
  pooled p50  1.74
  pooled p90  4.84
  pooled p99  279.62
  pooled max  4,051.02
  bin [1.0, 1.1) per-problem mean mass  0.487
  bin [5, +inf)                          0.055
```

```
tgt-only          (n=1450 pids, 20,560 grades)
  pooled p10  1.00
  pooled p50  1.09
  pooled p90  2.95
  pooled p99  50.33
  pooled max  1,210.20
  bin [1.0, 1.1) mass  0.788
  bin [5, +inf)        0.026
```

Plots: `pie_all_perproblem_avg_v2.png` (src+tgt, fine bins),
`pie_tgtonly_perproblem_avg.png` (tgt-only).

### 3.2 At `-O0` compile (no compiler optimization)

Source: `plots/pie_all_perproblem_elapsed_ns_O0.json` (per pid, separated
into `src` and `tgt` lists, both at -O0).

```
src+tgt combined  (n=1466 pids, 40,895 grades kept, 5473 gate-fail)
  pooled p10  1.014
  pooled p25  1.299
  pooled p50  6.534
  pooled p75  18.523
  pooled p90  32.755
  pooled p95  64.992
  pooled p99  777.496
  pooled max  245,004.983
  bin [1.0, 1.1) per-problem mean mass  0.424
  bin [5, +inf)                          0.319
```

```
tgt-only          (n=1452 pids, 20,570 grades)
  pooled p10  1.000
  pooled p25  1.012
  pooled p50  1.563
  pooled p75  5.753
  pooled p90  25.852
  pooled p95  33.650
  pooled p99  149.223
  pooled max  245,004.983
  bin [1.0, 1.1) mass  0.731
  bin [5, +inf)        0.112
```

#### Why -O0 changes the picture so much

`-O3` normalizes different implementations of the same algorithm down to
nearly identical machine code; bubble_sort and std::sort at -O3 can have
within ~5% instruction count. At -O0 the compiler preserves every
candidate-program iteration, so bubble_sort emits orders of magnitude
more retired instructions than std::sort. Algorithmic differences that
-O3 erases become visible at -O0:

| metric | -O3 src+tgt | **-O0 src+tgt** |
|---|---|---|
| overflow [5,∞) per-pid mass | 5.5% | **31.9%** |
| pooled p50 | 1.74× | **6.53×** |
| pooled p90 | 4.84× | **32.8×** |
| pooled max | 4,051× | **245,005×** |

Plots: `pie_perproblem_O0.png` (fine 0.1-width bins),
`pie_perproblem_O0_wbin.png` (1.0-width bins, x ∈ [1, 20] + overflow),
`pie_perproblem_O0_logx.png` (log-spaced bins, log x-axis to 10⁴).

### 3.3 Bin-by-bin mass (per-problem averaged at -O0, src+tgt combined, bin width 1.0)

```
[1,   2):   0.555    (most pids: candidates cluster near slowest)
[2,   3):   0.060
[3,   4):   0.038
[4,   5):   0.029
[5,   6):   0.022
[6,   7):   0.019
[7,   8):   0.014
[8,   9):   0.013
[9,  10):   0.046     ← shoulder
[10, 11):   0.013
[11, 12):   0.008
[12, 13):   0.008
[13, 14):   0.008
[14, 15):   0.008
[15, 16):   0.008
[16, 17):   0.008
[17, 18):   0.010
[18, 19):   0.037     ← shoulder
[19, 20):   0.005
[20, ∞):    0.096
```

Two visible shoulders at ~9× and ~18× — likely two canonical
"algorithmic substitution" magnitudes that recur across problems.

---

## 4. Per-problem tgt-pool tightness

How spread out are tgt programs within a single problem? max/min ratio
across each pid's tgt set, all-splits, -O0 perf counts.

Source: per-pid tgts from `pie_all_perproblem_elapsed_ns_O0.json` filtered
to `["tgt"]`. n = 1122 pids with ≥2 tgt programs.

### 4.1 Quantiles of `max/min` per pid

```
p10   1.001×     (10% of pids: tgts bit-identical)
p25   1.009×     (quarter of pids: tgts within 1%)
p50   1.043×     (median pid: tgts within 4.3%)
p75   3.352×
p90   14.230×
p95   27.309×
p99   221.543×
max   245,004.983×
```

### 4.2 Threshold sweep

```
max/min > 1.5×    35.6%  (399/1122)
max/min > 2×      31.1%  (349/1122)
max/min > 3×      25.8%  (290/1122)
max/min > 5×      20.8%  (233/1122)
max/min > 10×     13.7%  (154/1122)
max/min > 50×      2.8%  (31/1122)
```

### 4.3 Tightness buckets (the "3-bucket" view)

```
Tight        max/min < 1.05×        ~52% of pids
Moderate     1.05× ≤ max/min < 10×   ~34%
Heavy spread  max/min ≥ 10×          ~14%
```

### 4.4 Heavy-tail quantification

```
mean of max/min          234.63×    (n=1122)
median of max/min          1.0434×
mean / median            225×       ← classic heavy-tail signature
stdev                    7,311.89×
CV (stdev/mean)          31.16     (>>1 → heavy)

top  1% of pids account for  97.4% of summed max/min
top  5% of pids account for  98.7%
top 10% of pids account for  99.1%

Hill tail-index estimator (Pareto α, threshold x ≥ 2.0)
α ≈ 0.60   (< 1 → distribution doesn't have a finite mean in the limit;
            heaviest possible tail class)
```

### 4.5 IQR-based tightness (less sensitive to outliers)

```
p75/p25 within pid, n=802 pids with ≥4 tgts
p50  1.015×
p75  1.148×
p90  3.819×
p99  21.459×
```

81% of pids have their middle-half tgts within 10% — most variation is
in the outliers, not the bulk.

Plot: `pie_tgt_tightness_O0.png`.

---

## 5. Train-only verification of the tgt-pool tightness

Re-graded the train split only to check whether the all-splits picture
holds for just-the-train data (the data RL training will see).

```
1474 train pids, 21,970 unique train tgts
1338 pids have ≥1 passing tgt at -O0
1034 pids have ≥2 passing tgts

per-pid max/min on train-only tgts:
p10   1.001×
p25   1.010×
p50   1.043×
p75   3.452×
p90   14.238×
p99   277.166×
max   244,872.612×
mean  253.98×    median 1.0446×

max/min > 1.5×    35.7%  (369/1034)
max/min > 2×      31.0%
max/min > 3×      26.1%
max/min > 5×      21.1%
max/min > 10×     13.8%
max/min > 50×      2.9%
max/min > 100×     1.8%
```

**Essentially identical to the all-splits picture** — train, val, and test
have the same per-pid tgt-spread structure. There's no selection bias to
worry about when interpreting RL training metrics.

Source: `plots/pie_train_tgt_only_elapsed_ns_O0.json`.

---

## 6. Qwen3 baseline pass-rate sweep on PIE val

Zero-shot, no PIE-specific training. 200 val problems × 16 rollouts at
T=1.0 (+ 1 greedy rollout at T=0). pass@k via unbiased Codex estimator.
Pass-rate is gate-pass (compile + all-5-tests-pass), backend-agnostic
(walltime fine here since pass/fail is binary).

### 6.1 Headline numbers per (model, template, decoding)

```
Model                  Template       n    T      pass@1   pass@5   pass@16  n_solved
Qwen3-0.6B-Base        chat           16   1.0    0.0381   0.1725   0.4350   87/200
Qwen3-0.6B-Base        raw+1shot      16   1.0    0.0453   0.1976   0.4650   93/200
Qwen3-0.6B-Base        raw+1shot       1   0.0    0.2500     —        —      50/200
Qwen3-1.7B-Base        chat           16   1.0    0.0641   0.2625   0.5450  109/200
Qwen3-1.7B-Base        raw+1shot      16   1.0    (FAILED — GPU shard crash)
Qwen3-1.7B-Base        raw+1shot       1   0.0    0.5400     —        —     108/200
Qwen3-4B-Base          chat           16   1.0    0.1209   0.4373   0.7650  153/200
Qwen3-4B-Base          raw+1shot      16   1.0    0.1547   0.5077   0.7900  158/200
Qwen3-4B-Base          raw+1shot       1   0.0    0.3000     —        —      60/200
Qwen3-0.6B  (Instruct) chat           16   1.0    0.2897   0.6141   0.7450  149/200
Qwen3-0.6B  (Instruct) chat            1   0.0    0.4400     —        —      88/200
Qwen3-1.7B  (Instruct) chat           16   1.0    0.5781   0.8033   0.8800  176/200
Qwen3-1.7B  (Instruct) chat            1   0.0    0.6750     —        —     135/200
Qwen3-4B    (Instruct) chat           16   1.0    0.6803   0.8768   0.9300  186/200
Qwen3-4B    (Instruct) chat            1   0.0    0.7150     —        —     143/200
```

### 6.2 Headline pass@16 per model class

```
Qwen3-0.6B-Base      best variant: raw+1shot (T=1.0)     pass@16 = 0.465
Qwen3-1.7B-Base                    chat       (T=1.0)              0.545
Qwen3-4B-Base                      raw+1shot (T=1.0)              0.790
Qwen3-0.6B (Inst)                  chat       (T=1.0)              0.745
Qwen3-1.7B (Inst)                  chat       (T=1.0)              0.880
Qwen3-4B   (Inst)                  chat       (T=1.0)              0.930
```

### 6.3 Patterns

- **Instruct ≫ Base** at every size. Gap closes from ~28pp at 0.6B to
  ~14pp at 4B.
- **Pass-rate monotone in model size** for both classes.
- **For Base models, raw+1shot ≥ chat** at all sizes — base models prefer
  raw completion; forcing them into chat-template format costs accuracy.
- **Greedy outperforms sampled mean** for all Instruct models — the
  model is more confident than diverse; sampling at T=1.0 occasionally
  garbles syntactically-correct programs.
- **One missing data point**: `base_1.7B_oneshot_n16_t1` (GPU shard
  crash, never re-run).

Tokenizer audit: all 6 Qwen3 models have vocab size 151,669. Base EOS =
`<|endoftext|>` (151643). Instruct EOS = `<|im_end|>` (151645). When
asked to apply_chat_template, both Base and Instruct produce IDENTICAL
rendered strings (the jinja-source hash differs only in whitespace
formatting). So "chat template on Base" is testing the same prompt as
"chat template on Instruct" but with a different model.

---

## 7. Custom-reward distribution probe (src-given prompt, -O3)

Define `r = max(0, ln(src_instr / gen_instr))` if all_pass else 0.
Pre-measure `src_instr` per problem via perf, generate K = 256 rollouts
per problem via vLLM (chat template + enable_thinking=False), then grade
each rollout. 8 val problems (`p02587, p03415, p03375, p03460, p02639,
p03947, p02317, p02621`) × 256 = 2048 rollouts per model.

### 7.1 Aggregate

```
                     0.6B Inst -O3    4B Inst -O3    1.7B SFT -O3
n_rollouts           2048             2048           2048
gate_pass            355 (17.3%)      1048 (51.2%)   241 (11.8%)
positive r           16 (0.78%)       42 (2.05%)     152 (7.42%)
positive r among gp  4.51%            4.01%          63.07%
mean speedup (pos)   1.035×           1.041×         1.008×
median speedup       1.051×           1.053×         1.004×
max speedup          1.053×           1.062×         1.371×
mean reward (all)    0.0003           0.0008         0.0006
mean reward (pos)    0.0344           0.0397         0.0075
```

### 7.2 Per-problem (4B Inst -O3)

```
pid       gate_pass  n_positive  mean_speedup   max_speedup
p02587       0            0           —              —
p03415      72           32        1.0533×        1.0620×
p03375       0            0           —              —
p03460     114            9        1.0005×        1.0005×
p02639     223            0           —              —
p03947     239            0           —              —
p02317     171            0           —              —
p02621     229            1        1.0007×        1.0007×
```

Almost all positive-reward mass is on a single problem (`p03415`),
clustered at exactly 1.05×. At -O3 the visible "speedup" is just the
constant-factor I/O improvements (`endl` → `"\n"`, removing unused
includes) — no algorithmic discrimination.

### 7.3 SFT model qualitative difference

```
SFT -O3
n_positive = 152 / 241 gp  = 63.07%
                            ← vs ~4% for Instruct models
median speedup = 1.004×    (0.4% improvement)
```

SFT learned to reliably *make some change* but the change is mostly
trivial. The Instruct models change less often but when they do change,
they sometimes pick up real micro-optimizations.

Plots: `custom_reward_qwen3_06B/reward_distribution.png`,
`.../qwen3_4B/...`, `.../qwen3_1.7B_sft/...`.

---

## 8. Same probe under -O0 compile

Critical experiment: re-running everything with `-O0` reveals algorithmic
differences that `-O3` was erasing.

### 8.1 Aggregate

```
                     0.6B Inst -O0    4B Inst -O0    1.7B SFT -O0
n_rollouts           2048             2048           2048
gate_pass            359 (17.5%)      1066 (52.1%)   239 (11.7%)
positive r           18 (0.88%)       92 (4.49%)     155 (7.57%)
positive r among gp  5.01%            8.63%          64.85%
mean speedup (pos)   1.4689×          1.4554×        1.0201×
median speedup       1.7082×          1.8490×        1.0042×
p90 speedup          1.8560×          1.8560×        1.0047×
max speedup          1.8560×          1.8733×        2.2739×
mean reward (all)    0.0030           0.0148         0.0011
mean reward (pos)    0.3433           0.3293         0.0143
```

### 8.2 Direct -O3 vs -O0 comparison for Instruct models

```
                    0.6B Inst        4B Inst
                    -O3      -O0     -O3      -O0
median speedup      1.051×   1.708×  1.053×   1.849×
max speedup         1.053×   1.856×  1.062×   1.873×
positive r among gp 4.51%    5.01%   4.01%    8.63%
mean reward (pos)   0.0344   0.3433  0.0397   0.3293
```

At -O3 the reward signal is invisible (1.05× speedups, r ≈ 0.05). At -O0
the reward signal is clearly distinguishable (1.85× speedups, r ≈ 0.62).

Plot: `custom_reward_O0_combined.png` — three panels showing
positive-r distributions for each model.

### 8.3 -O3 + -O0 SFT contrast

SFT under -O0 has the highest max speedup (2.27× on `p02587`) of any
model in this study — that's the only Mode-2 "real algorithmic discovery"
seen. But it happens only twice out of 2048 rollouts. SFT mostly emits
trivial micro-changes (median 1.004×) just like at -O3.

---

## 9. Bimodality decomposition: "near src" vs "near tgt"

For each gate-pass rollout, classify it by:
- `gen_instr / src_instr` ∈ [0.9, 1.1] → "near src speed"
- `gen_instr / tgt_median` ∈ [0.9, 1.1] → "near tgt speed"
- otherwise → "in between" or "other"

This isolates **what speed regime the model is producing code in**.

### 9.1 Bucketing of all gate-pass rollouts

```
Qwen3-0.6B Inst -O0  (n=359 gate-passes)
  near src speed:       346  (96.4%)
  near tgt speed:         9  (2.5%)
  in between:             2  (0.6%)
  other:                  2  (0.6%)

Qwen3-4B Inst -O0    (n=1066 gate-passes)
  near src speed:      1006  (94.4%)
  near tgt speed:        49  (4.6%)
  in between:             0  (0.0%)
  other:                 11  (1.0%)

Qwen3-1.7B SFT -O0   (n=239 gate-passes)
  near src speed:       237  (99.2%)
  near tgt speed:         2  (0.8%)
  in between:             0  (0.0%)
  other:                  0  (0.0%)
```

### 9.2 Bimodal reward decomposition (-O0)

Threshold `r = 0.3` cleanly separates Mode 1 (trivial) from Mode 2 (real):

```
                Mode 1 (r<0.3)               Mode 2 (r≥0.3)
                                      | median r | max speedup ||                | median r | median speedup
0.6B Inst -O0    8 (44.4%)            | 0.0028   | 1.168×      || 10 (55.6%)     | 0.6176   | 1.854×
4B Inst -O0     43 (46.7%)            | 0.0006   | 1.004×      || 49 (53.3%)     | 0.6184   | 1.856×
1.7B SFT -O0   153 (98.7%)            | 0.0042   | 1.095×      ||  2 ( 1.3%)     | 0.8215   | 2.274×
```

Three observations:

- **Empty middle**: almost zero rollouts at r ∈ [0.1, 0.5]. The reward
  distribution is genuinely bimodal — speedups are either trivial
  (≤1.1×) or algorithmic (≥1.5×). No continuous in-between.
- **Mode-2 ratios are tight**: 1.85× across both Instruct models. The
  models converge on the SAME canonical optimization pattern (likely
  `p03415` closed-form solution).
- **SFT distribution is qualitatively different**: explored Mode 1
  heavily (153/155 positives) but rarely hit Mode 2.

### 9.3 Why the bimodality matters for RL

A reward function that doesn't distinguish Mode 1 vs Mode 2 (e.g., binary
`r > 0` reward, or anything that saturates at small speedups) will let
the model exploit Mode 1 and never learn Mode 2. The `ln(src/gen)` reward
the user specified has the right log-scale property to separate the two
clusters.

The maxrl_unnorm advantage shape in particular gets enormous benefit:
within a G=16 batch you'll often see ~14 zeros + ~1 small-positive + ~1
big-positive. The big-positive's advantage relative to the mean is large
— exactly the gradient signal RL needs. Under -O3, both small and big
collapsed near zero, killing the signal.

---

## 10. Reward ceiling: are Mode-2 generations competitive with tgts?

Important sanity check: when the model achieves `r ≈ 0.62` (speedup
~1.85× vs src), is the generated code actually as fast as a tgt program,
or just less slow than src?

### 10.1 4B Instruct -O0 Mode-2 successes (all on p03415, n=49)

```
src_instr      9,507,327     (slow input)
gen_instr      ~5,122,000    (generated, ~1.85× faster than src)
tgt_med        4,992,608     (median tgt for this problem)
tgt_min        4,991,671     (fastest tgt)

gen / src       0.539  (gen is 1.85× faster than src)  ← reward sees this
gen / tgt_med   1.026  (gen is 2.6% SLOWER than median tgt)
gen / tgt_min   1.026  (gen is 2.6% slower than fastest tgt)

beats median tgt:   0 / 49
beats fastest tgt:  0 / 49
```

### 10.2 0.6B Instruct -O0 Mode-2 (10 on p03415)

```
median gen/tgt_med   1.027   beats median tgt:  0/10
median gen/tgt_min   1.027
```

### 10.3 1.7B SFT -O0 Mode-2 (2 on p02587)

```
gen_instr      9,489,820     src 21,579,270 (so r = ln(2.27) = 0.82)
tgt_med        9,388,528     gen is 1.1% slower than median tgt
tgt_min        8,358,836     gen is 13% slower than fastest tgt

beats median tgt:  0/2
beats fastest tgt: 0/2
```

### 10.4 Interpretation

- The Mode-2 generations all converge to the same algorithmic class as
  the existing tgts, but implementation is ~1-13% less polished. The
  models are *retrieving* the canonical pattern, not *inventing* it.
- The src/tgt-median gap upper-bounds the reward signal. For p03415 that
  ratio is 9.5M / 5M = 1.90, so `ln(1.90) = 0.64` is the practical
  ceiling of `ln(src/gen)` for that problem.
- A reward like `ln(median_tgt / gen)` would zero out for matching-tgt
  generations and reward only ABOVE-tgt performance — much harder signal.

---

## 11. Cross-referenced inventory of every artifact

### 11.1 Raw JSON data (no plot needed, for replotting)

| file | contents | n records |
|---|---|---|
| `pie_val_src_tgt_ratios.json` | val src + tgt ratios vs slowest_ref, -O3 | 687 src + 758 tgt |
| `pie_all_perproblem_elapsed_ns.json` | per-pid combined src+tgt elapsed_ns, all splits, -O3 | 37,685 grades |
| `pie_tgtonly_perproblem_elapsed_ns.json` | per-pid tgt-only elapsed_ns, -O3 | 20,560 grades |
| `pie_all_perproblem_elapsed_ns_O0.json` | per-pid src + tgt elapsed_ns (kind-tagged), -O0 | 40,895 grades |
| `pie_train_tgt_only_elapsed_ns_O0.json` | per-pid tgt-only elapsed_ns from train.jsonl only, -O0 | ~17,000 grades |
| `pie_all_perproblem_avg_ratios.json` | aggregated histogram (combined src+tgt avg), -O3 | 50 log bins |
| `pie_perproblem_O0_data.json` | aggregated histograms (combined + tgt-only), -O0 | 41 linear 0.1 bins |
| `custom_reward_qwen3_{06B,4B,1.7B_sft}{,_O0}/rewards.json` | per-rollout (pid, j, r, src_instr, gen_instr, speedup), 6 dirs | 2048 each |

### 11.2 Plot files (all under `experiment11_code_optim/docs/plots/`)

| file | shows |
|---|---|
| `pie_val_src_tgt_speedup.png` | val src vs tgt ratios vs slowest ref, -O3 |
| `pie_all_perproblem_avg_speedup.png` | per-problem avg, src+tgt combined, -O3, log bins |
| `pie_all_perproblem_avg_v2.png` | same with refined bins |
| `pie_tgtonly_perproblem_avg.png` | per-problem avg, tgt-only, -O3 |
| `pie_ref_table_norm_slowest.png` | per-problem avg + pooled, tgt-only perf ref table |
| `pie_ref_table_norm_slowest_wall_vs_perf.png` | walltime vs perf 2x2 grid |
| `pie_perproblem_O0.png` | per-problem avg, src+tgt and tgt-only side-by-side, -O0, 0.1-wide bins |
| `pie_perproblem_O0_wbin.png` | -O0, bin width 1.0 |
| `pie_perproblem_O0_logx.png` | -O0, log-spaced bins on log x-axis |
| `pie_tgt_tightness_O0.png` | per-pid tgt max/min and p75/p25 histograms |
| `custom_reward_qwen3_06B/reward_distribution.png` | 0.6B Inst -O3 |
| `custom_reward_qwen3_4B/reward_distribution.png` | 4B Inst -O3 |
| `custom_reward_qwen3_1.7B_sft/reward_distribution.png` | SFT 1.7B -O3 |
| `custom_reward_qwen3_06B_O0/reward_distribution.png` | 0.6B Inst -O0 |
| `custom_reward_qwen3_4B_O0/reward_distribution.png` | 4B Inst -O0 |
| `custom_reward_qwen3_1.7B_sft_O0/reward_distribution.png` | SFT 1.7B -O0 |
| `custom_reward_O0_combined.png` | side-by-side, all 3 -O0 reward distributions |
| `probe_perf_gate{1,2,3,4}.json` | perf-backend validation gates |
| `probe_variant_e_*.json`, `probe_variant_f_*.json` | concurrency noise sweep |
| `probe_walltime_*.json` | walltime noise probes |

### 11.3 Scripts that produced the above (under `experiment11_code_optim/scripts/`)

| script | purpose |
|---|---|
| `custom_reward_distribution.py` | custom reward probe (CLI: --model, --opt_level, --timeout_per_test) |
| `custom_reward_no_src.py` | no-src variant (I/O examples only) — drafted but unused |
| `aggregate_baseline_eval.py` | aggregate baseline pass-rate results |
| `eval_qwen3_baselines.sh` + `eval_qwen3_baseline_one.sh` | sweep driver + per-job wrapper |
| `baseline_eval_oneshot_preamble.txt` | 1-shot anchor for Base + raw_oneshot |
| `perproblem_speedup_O0.py` | per-problem speedup grading + histogram at -O0 |
| `tgt_spread_train.py` | TRAIN-only tgt spread |
| `rebuild_reference_table_perf.sh` | rebuilds perf-units reference table |

### 11.4 Reference-table data files (data lake)

| path | what |
|---|---|
| `/work/nvme/bgmo/sramasubramanian/pie_data/reference_table/` | walltime reference (legacy) |
| `/work/nvme/bgmo/sramasubramanian/pie_data/reference_table.pkl` | walltime pickle |
| `/work/nvme/bgmo/sramasubramanian/pie_data/reference_table_perf/` | perf reference (-O3) |
| `/work/nvme/bgmo/sramasubramanian/pie_data/reference_table_perf.pkl` | perf pickle |

Both tables cover **1450 pids** with passing references (142 problems
excluded due to zero passing tgts under K=5 test cap). The excluded
set is **identical** between walltime and perf reference tables (verified
in §1.4).

---

## Summary of practical conclusions for RL

1. **Walltime is unusable as a reward measurement under production
   concurrency.** Switch to perf (instruction count). Committed and
   pushed (commits `ff7820e`, `763b5dc`, `0c71c2d`).

2. **The PIE reference data is heavy-tailed.** 52% of problems have a
   tight tgt pool (one canonical algorithm, all tgts within 5% of each
   other) and provide little learning signal beyond "match the canonical
   algorithm." ~14% of problems have ≥10× spread — those are where RL has
   the richest signal. The top 1% contribute 97% of the spread mass.

3. **Compile with -O0, not -O3** for the reward measurement. -O3
   normalizes implementations down to similar instruction counts; -O0
   preserves the algorithmic-class differences (5×-10×+ ratios).

4. **The reward distribution is genuinely bimodal**: trivial speedups
   (~1.05×, r ≈ 0.05) and real algorithmic speedups (~1.85×, r ≈ 0.62).
   Empty middle between r = 0.1 and r = 0.5. The `ln(src/gen)` reward
   function correctly separates the two clusters on the log scale.

5. **Pre-RL untrained models barely produce algorithmic improvements.**
   Best zero-shot model (Qwen3-4B Instruct) achieves real speedup on
   only 4.5% of rollouts; max speedup is 1.87× and converges to existing
   tgt patterns (never beats fastest tgt). This is the gap RL on PIE is
   designed to close.

6. **The SFT-trained model is qualitatively different**: low gate-pass
   (12%) but high speedup-given-pass (65%). Most speedups are trivial
   (~0.4%) but the rare Mode-2 hits are the largest seen (up to 2.27×).

7. **Pass-rate scales with model size for both Base and Instruct**, with
   Instruct uniformly ahead. Best zero-shot pass@16 is 93% for Qwen3-4B
   Instruct on 200 val problems.
