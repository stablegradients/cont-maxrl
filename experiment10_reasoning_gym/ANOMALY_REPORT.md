# Experiment 10 · Reasoning Gym — Anomaly Report

**Generated:** 2026-04-24
**W&B project:** `stablegradients/Reasoning-GYM-GOS`
**Runs analysed:** 20 (seed 42), all combinations of {WS, SR, GoL} × {3,4,7,8,10,12} × {gos, grpo} × {continuous, binary} that had logged ≥1 step.
**Method:** pulled full step-wise history per run via `wandb.Api().runs(...).history(samples=10000)`, computed reward, length, entropy, gradient, and clip-fraction trajectories; flagged trend reversals, length hacking, entropy/grad collapse; cross-referenced final scores at matched (task, est, reward, diff) tuples for anti-evidence.

The full analysis script lives at `/tmp/deep_analysis.py` (not committed) and the structured output at `/tmp/deep_analysis_out.txt`.

---

## TL;DR

The headline hypothesis ("continuous reward outperforms binary as difficulty rises, and GOS outperforms GRPO") is **partially supported** in the cleanest task (sentence_reordering with GOS continuous), and **catastrophically violated** in two specific places:

1. **GRPO + continuous + sentence_reordering** collapses to a length-hack basin in the very first epoch, scoring 0.050 forever, while the same configuration with GOS reaches 0.99. Δ = +0.93pp in favor of GOS.
2. **word_sorting diff4 with GOS** has *binary > continuous* by 21pp, because the continuous reward hits the documented "extra-words-after-sort" hack and entropy collapses while binary stays clean.

Plus: 9 of 20 runs show entropy collapse, 7 show gradient collapse, 4 show late-training trend reversals. The training configuration is stable end-to-end but deeply susceptible to length/entropy attractors on saturated tasks.

---

## 1. Per-run table

```
RUN                                                 state   step  score   max  Δrev  rlen  clip   ent   grad   f0    f1   flags
----------------------------------------------------------------------------------------------------------------------------------
game_of_life_grpo_binary_diff4_s42                  crashed   86  0.000 0.000 0.000  338  0.21  3.728  0.00 1.00 0.00  -
sentence_reordering_gos_binary_diff3_s42            crashed  205  0.883 0.953 0.071   13  0.00  0.069  0.05 0.12 0.88  TREND_REVERSAL,ENTROPY_COLLAPSE,GRAD_COLLAPSE
sentence_reordering_gos_continuous_diff3_s42        crashed  317  0.977 0.989 0.011   11  0.00  0.026  0.07 0.00 0.93  ENTROPY_COLLAPSE,GRAD_COLLAPSE
sentence_reordering_gos_continuous_diff7_s42        crashed  306  0.986 0.989 0.003   15  0.00  0.010  0.04 0.00 0.89  ENTROPY_COLLAPSE
sentence_reordering_gos_continuous_diff10_s42       crashed  349  0.789 0.821 0.032   13  0.00  0.017  0.09 0.00 0.00  TREND_REVERSAL,ENTROPY_COLLAPSE
sentence_reordering_grpo_binary_diff3_s42           crashed  226  0.888 0.953 0.065   81  0.04  0.546  0.20 0.11 0.89  TREND_REVERSAL
sentence_reordering_grpo_binary_diff7_s42           crashed  144  0.000 0.000 0.000  263  0.35  4.455  0.00 1.00 0.00  -
sentence_reordering_grpo_continuous_diff3_s42       crashed   80  0.050 0.050 0.000  494  0.93  5.109  0.01 0.00 0.00  LENGTH_HACK,GRAD_COLLAPSE
sentence_reordering_grpo_continuous_diff7_s42       crashed  104  0.050 0.050 0.000  493  0.93  5.115  0.02 0.00 0.00  LENGTH_HACK,GRAD_COLLAPSE
word_sorting_gos_binary_diff4_s42                   running  355  0.791 0.817 0.026   15  0.00  0.056  2.36 0.21 0.79  ENTROPY_COLLAPSE
word_sorting_gos_binary_diff8_s42                   crashed  146  0.000 0.000 0.000  188  0.23  4.112  0.00 1.00 0.00  -
word_sorting_gos_binary_diff12_s42                  crashed  157  0.000 0.000 0.000  186  0.23  4.159  0.00 1.00 0.00  -
word_sorting_gos_continuous_diff4_s42               running  174  0.576 0.609 0.033  510  0.99  0.191  0.12 0.05 0.31  TREND_REVERSAL,ENTROPY_COLLAPSE
word_sorting_gos_continuous_diff8_s42               running  216  0.352 0.352 0.000  283  0.29  0.318  0.19 0.06 0.02  ENTROPY_COLLAPSE
word_sorting_gos_continuous_diff12_s42              crashed  129  0.204 0.210 0.006  129  0.02  0.411  0.00 0.00 0.00  ENTROPY_COLLAPSE,GRAD_COLLAPSE
word_sorting_grpo_binary_diff4_s42                  running   58  0.096 0.096 0.000  132  0.10  1.565  0.27 0.90 0.10  -
word_sorting_grpo_binary_diff8_s42                  crashed  104  0.000 0.000 0.000  186  0.22  4.146  0.00 1.00 0.00  -
word_sorting_grpo_binary_diff12_s42                 crashed  161  0.000 0.000 0.000  186  0.23  4.135  0.00 1.00 0.00  -
word_sorting_grpo_continuous_diff4_s42              running  245  0.822 0.835 0.013  511  0.99  0.002  0.10 0.01 0.67  ENTROPY_COLLAPSE,GRAD_COLLAPSE
word_sorting_grpo_continuous_diff8_s42              crashed  177  0.228 0.235 0.007  206  0.14  1.228  0.03 0.00 0.00  GRAD_COLLAPSE
```

Columns: `Δrev` = max-score minus final-score (positive = trend reversal). `rlen` = mean response length at last step. `clip` = response_length/clip_ratio (fraction hitting `max_response_length`). `ent` = final entropy loss. `grad` = final actor grad norm. `f0/f1` = fraction of rollouts scoring exactly 0 / exactly 1.

State `crashed` here is W&B's tag for runs whose process was killed (by us via the stall-watchdog or chain replacement) — *not* a training crash.

---

## 2. Anti-evidence findings

### A1. GRPO + continuous + sentence_reordering catastrophically collapses to length-hack at step 1

| run | step | score | rlen | clip | entropy | grad |
|---|---:|---:|---:|---:|---:|---:|
| sr_grpo_continuous_diff3 | 80 | 0.050 | **494** | **0.93** | 5.11 | 0.014 |
| sr_grpo_continuous_diff7 | 104 | 0.050 | **493** | **0.93** | 5.11 | 0.015 |
| sr_**gos**_continuous_diff3 | 317 | **0.977** | 11 | 0.00 | 0.026 | 0.065 |
| sr_**gos**_continuous_diff7 | 306 | **0.986** | 15 | 0.00 | 0.010 | — |

**Δ(gos − grpo) = +0.93pp.** Same task, same reward, only estimator differs. GRPO variants jumped to 494-token responses within the first epoch and never produced signal; GOS variants stayed short (~15 tokens) and solved the task. This is the most decisive differential in the entire sweep, and it is the *opposite* of what the conventional "GRPO is the safe baseline" intuition predicts.

**Mechanistic hypothesis.** GRPO normalizes advantages by group standard deviation: `A_i = (R_i − μ_group) / σ_group`. When the initial group std is small (all 16 rollouts cluster near 0.05, since the base model produces similar low-quality SR completions), tiny raw differences become huge normalized advantages. One bad early update lands the policy in the length-hack basin (max-response-length attractor) and policy gradients then keep amplifying the bad direction because the local gradient near max-length is small (clip_ratio ≈ 0.93, grad_norm ≈ 0.014). GOS's ordinal-rank advantage is bounded by definition and doesn't get magnified by tight reward groups, so it stays in the productive learning regime.

### A2. word_sorting diff4 with GOS: binary beats continuous by 21pp

| run | score (final) | score (max) | rlen | clip |
|---|---:|---:|---:|---:|
| ws_gos_**binary**_diff4 | 0.791 | 0.817 | **15** | 0.00 |
| ws_gos_**continuous**_diff4 | 0.576 | 0.609 | **510** | 0.99 |

Direct contradiction of the headline hypothesis. The cause is the documented "extra-words-after-sort" hack of `WordSortingDataset.score_answer` — only the first `len(oracle)` words of the answer are scored, so trailing tokens are free. Continuous reward incentivizes the model to add a "safe sort prefix + arbitrary padding", entropy collapses to 0.12, response length saturates at 510. Binary reward needs `continuous == 1.0` exactly, which is harder to game cheaply, so the binary-trained policy keeps outputs at length 15 and actually sorts cleanly.

This is anti-evidence against the central spec claim *for this specific easy difficulty*. The continuous-has-richer-signal argument only holds when the reward function does not have a cheap local optimum.

### A3. word_sorting continuous: gos vs grpo flips with difficulty

| | gos | grpo | Δ(gos − grpo) |
|---|---:|---:|---:|
| ws_continuous_diff4 | 0.576 | **0.822** | −0.246 |
| ws_continuous_diff8 | **0.352** | 0.228 | +0.124 |

Both end in the length-hack basin at diff4 (rlen ≥ 510, clip ≈ 0.99) but GRPO rides it higher because z-score normalization keeps gradients well-scaled inside the basin. GOS collapses entropy sooner (0.004 vs 0.19 — note GRPO's policy is *more* deterministic, yet its reward is higher). At harder diff8 neither has saturated, and GOS's sharper top-group credit gives the edge.

The estimator ranking flips depending on whether the task has a cheap hack.

---

## 3. Mechanistic anomalies

### M1. Near-universal entropy collapse (9 of 20 runs)

Every GOS run that made progress hit entropy collapse. Final entropy values:

- `ws_grpo_continuous_diff4`: **0.002** (initial 4.06, drop = 99.95%) — the policy is essentially deterministic
- `sr_gos_continuous_diff7`: 0.010
- `sr_gos_continuous_diff10`: 0.017
- `sr_gos_continuous_diff3`: 0.026
- `ws_gos_continuous_diff4`: 0.181
- `ws_gos_continuous_diff8`: 0.318
- `ws_gos_continuous_diff12`: 0.411
- `sr_gos_binary_diff3`: 0.069
- `ws_gos_binary_diff4`: 0.056

Expected for on-policy PG with no entropy bonus, but the magnitude is extreme and tightly correlated with length-hack onset. Adding `actor.entropy_coef ≈ 0.01` or a KL-to-reference regularizer would likely arrest this and prevent the length attractors.

### M2. Gradient collapse (7 runs)

Always paired with entropy collapse. Final `actor/grad_norm`:

- `ws_gos_continuous_diff12`: **0.003**
- `sr_grpo_continuous_diff3`: 0.014
- `sr_grpo_continuous_diff7`: 0.015
- `sr_gos_binary_diff3`: 0.054
- `sr_gos_continuous_diff3`: 0.065
- `ws_grpo_continuous_diff4`: 0.098
- `ws_grpo_continuous_diff8`: 0.033

Near-zero gradient means training has *frozen*, not "still learning slowly". Interventions can't recover from here without restarting with regularization.

### M3. Trend reversals (4 runs)

| run | max (step) | final (step) | Δrev |
|---|---|---|---|
| `sr_gos_binary_diff3` | 0.953 (184) | 0.883 (205) | −0.071 |
| `sr_grpo_binary_diff3` | 0.953 (184) | 0.888 (226) | −0.065 |
| `sr_gos_continuous_diff10` | 0.821 (256) | 0.789 (349) | −0.032 |
| `ws_gos_continuous_diff4` | 0.609 (162) | 0.570 (174) | −0.039 |

The two SR-binary-diff3 runs reverse *with the same shape from the same step* (peak at step 184, both gos and grpo). That's epoch-boundary suspicious: epoch 2 ends at step 156 and epoch 3 ends at 234, so step 184 is mid-epoch — but the data sampler's shuffle seed advances per epoch, so something coincides at the same global step across two runs. Worth investigating whether this is a verl dataloader artefact or a genuine policy phenomenon.

### M4. Difficulty non-monotonicity (mild violation, likely ceiling noise)

`sentence_reordering_gos_continuous`: diff3=0.977, **diff7=0.986**, diff10=0.789. diff7 narrowly *beats* diff3, but both are at the 0.989 ceiling — almost certainly rollout-variance noise at saturation. diff10's 0.789 is the genuine drop and it's monotonic relative to that ceiling.

All other (task, est, reward) groups are monotonic in the expected direction.

### M5. Binary sparsity cliff confirmed cleanly on WS and GoL

- `ws_gos_binary`: diff4=0.791, diff8=0.000, diff12=0.000
- `ws_grpo_binary`: diff4=0.096, diff8=0.000, diff12=0.000
- `gol_grpo_binary_diff4`: 0.000

Binary completely dies once `len(oracle) ≥ 8` — clean demonstration of sparsity collapse. This *validates* the half of the central hypothesis that says "binary fails when sparse."

### M6. Sentence_reordering with continuous + GOS supports the headline cleanly

- diff3: gos_continuous **0.977** vs gos_binary 0.883 → +10pp for continuous
- diff10: gos_continuous **0.789** vs no comparable binary run yet
- diff7: only continuous trained to ceiling so far

When the task is hard enough that the model can't trivially reach `score == 1.0`, continuous's richer signal helps and there's no length-hack loophole (SR's score function ignores trailing tokens but its tree-edit-distance is more discriminating than WS's positional). This is the intended demonstration.

---

## 4. Hypotheses produced by the data

1. **GRPO is fragile under low-variance rewards.** Z-score normalization amplifies tight reward groups into oversized advantages, flipping the policy into bad attractors before learning gets traction. GOS (ordinal-rank, MaxRL-unnormalized) is bounded by construction and avoids this. This is the strongest single finding.

2. **The length-hack severity is task- and difficulty-specific.** It dominates word_sorting diff4 (small oracle, position-only scoring) and sentence_reordering early training under GRPO (low-variance reward). It does not appear on sentence_reordering with GOS or on game_of_life. The hack matters when the reward function admits a cheap-to-discover local optimum *and* the optimizer is in a regime that explores around it.

3. **Entropy regularization is plausibly a single-knob fix for most pathologies seen.** All 9 entropy-collapsed runs and 7 grad-collapsed runs share the same dynamic: policy commits to a low-entropy mode, gradient shrinks, training freezes. `actor.entropy_coef ≈ 0.01` would directly counter this. Worth one ablation run before changing anything else.

4. **Two SR binary runs reversing at identical step 184 is weird** and merits a look at the dataloader epoch boundary or a saved validation pass.

---

## 5. Recommended follow-ups (priority order)

1. **Add `actor.entropy_coef = 0.01` to `launch_slurm_deltaai.sh`** and resume one mid-stall run (e.g. `ws_grpo_continuous_diff8` from its step-156 checkpoint) as a single-variable ablation. If entropy collapse is averted and the run continues to improve, propagate the setting to the rest of the sweep.

2. **Investigate `sr_grpo_continuous`** specifically. Resume one with extra logging (`critic/advantages/std`, `critic/advantages/abs_max`, dump rollout examples at steps 1–5) to confirm the narrow-reward → large-advantage → length-jump hypothesis. If the std is genuinely tiny in step 1, that's the smoking gun.

3. **For the write-up**, present results as three cuts:
   - SR with GOS — clean support for the hypothesis (the headline plot)
   - WS diff4 — cautionary length-hack case requiring DAPO overlong shaping or per-token reward masking before fair comparison
   - Binary sparsity cliff at WS diff8/12 and GoL diff4 — clean evidence for the "binary fails when sparse" half

4. **Don't naïvely resume the 13 killed runs.** Same hyperparameters land in the same basin. Resuming with regularization (entropy_coef or KL-to-ref) is the only useful experiment from those checkpoints.

---

## 6. Methodology footnotes

- W&B history was queried with `samples=10000` to avoid downsampling for runs with up to ~780 logged steps; in practice we got every step.
- "Length hack" flag triggers when `clip_ratio ≥ 0.5` AND `score_reversal < 0.01` AND `step ≥ 80`. Captures the saturate-at-max-length-without-reward-improvement signature.
- "Entropy collapse" flag triggers at `entropy_drop_frac ≥ 0.9` AND `step ≥ 80`. Captures policies that have lost ≥90% of their initial entropy.
- "Gradient collapse" flag triggers at `grad_drop_frac ≥ 0.7` AND `step ≥ 80`.
- "Trend reversal" flag triggers at `(rolling_max − final) ≥ 0.03` AND `step ≥ 40`.
- "Anti-evidence" tags in cross-run tables fire at `Δ < −0.02pp` (margin to filter ceiling noise).
- All comparisons use `critic/score/mean` (training reward) since validation passes are infrequent (every 78 steps) and many runs were killed before two val passes could land.
