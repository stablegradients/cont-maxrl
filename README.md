# Continuous Maximum Likelihood Reinforcement Learning

**Extending MaxRL to Continuous Rewards via Integrated Survival Functions**

---

## Overview

This repository contains the code for the bandit experiments in the working note *Continuous Maximum Likelihood Reinforcement Learning*. The paper generalizes binary MaxRL (which works on pass/fail rewards) to the case where rewards are continuous scalars in `[0, 1]`.

**The core idea in one sentence:** replace pass@k with `max_reward@k = E[max(r_1, ..., r_k)]`, and the entire MaxRL framework carries over — including the harmonic-series convergence argument, the closed-form estimator, and the integrated log-survival limit.

---

## Background

### Binary MaxRL (Tajwar et al., 2026)

Binary MaxRL is defined for rewards `r ∈ {0, 1}`. It constructs a family of objectives:

```
J_binary^(T) = sum_{k=1}^{T} (1/k) * pass@k
```

where `pass@k = 1 - (1-p)^k` and `p = P(r = 1)`. As `T → ∞`, the gradient converges to `∇ log p` — the maximum likelihood gradient — via the Maclaurin series of `-log(1-x)`.

### This Paper: Continuous MaxRL

For continuous rewards, the survival function `p_τ = P(r > τ)` plays the role of `p`. We define:

```
max_reward@k = E[max(r_1, ..., r_k)] = ∫_0^1 [1 - (1 - p_τ)^k] dτ
```

The continuous MaxRL objective is:

```
J_cont^(T) = sum_{k=1}^{T} (1/k) * max_reward@k
```

As `T → ∞`, the gradient converges to the **integrated log-survival objective**:

```
J_ML^cont = ∫_0^1 log P(r > τ) dτ
```

This asks not just "can you solve it?" but "how well can you solve it?" — maximizing `log P(r > τ)` averaged uniformly over all quality levels `τ`.

### The Estimator (Algorithm 1)

Given `N` rollouts with rewards `r_1, ..., r_N`, the gradient estimator has a closed-form solution via **gap-over-survivors weighting**:

1. **Sort** rollouts: `r_(1) ≤ r_(2) ≤ ... ≤ r_(N)`, set `r_(0) = 0`
2. **Compute cumulative weights** (running sum):
   ```
   w_(0) = 0
   w_(i) = w_(i-1) + (r_(i) - r_(i-1)) / (N - i + 1)
   ```
3. **Center** weights: subtract the batch mean `w̄ = (1/N) Σ w_(i)`
4. Use centered `w_(i)` as the advantage for rollout `i`

**Complexity:** `O(N log N)` from the sort; `O(N)` for all remaining steps.

**Binary recovery:** When `r ∈ {0, 1}`, this reduces exactly to `w_(i) = 1/K` for successes and `0` for failures — the binary MaxRL estimator (where `K` is the number of successes).

**Intuition for the weights:** Each `w_(i)` accumulates credit from every reward gap that sample `i` "survives" (all gaps at rewards below `r_(i)`). Each gap is divided by the number of surviving samples. This means:
- Higher-reward samples get more weight (they survive more gaps)
- Gaps at high thresholds contribute more per unit (fewer survivors → `1/(N-j)` is large)

This is the continuous analog of the `1/K` upweighting in binary MaxRL.

---

## Key Results

### What continuous MaxRL does differently

At large batch size `N`, the gap-over-survivors weighting produces a **spike**: advantages are near zero for ~96% of samples, then rise sharply for the top samples. The highest-reward sample receives ~5× the weight it would get under GRPO's z-score normalization.

This amplification lets the method discover **rare high-reward strategies** that mean-seeking methods (GRPO, REINFORCE) never find.

### Experiment 1 — Moderate environment (50 arms, 5 Type B)

| N | Method | E[r] | Best-of-N | ILS | π(B) |
|---|--------|------|-----------|-----|------|
| 256 | GRPO | **0.806** | 0.962 | -0.770 | 0.0% |
| 256 | Cont. MaxRL (unnorm) | 0.766 | **0.987** | **-0.627** | **8.9%** |
| — | ILS-Optimal | 0.753 | 0.992 | -0.605 | 12.8% |

At N=256, unnormalized MaxRL recovers 70% of the ILS-optimal arm allocation and comes within 0.022 nats of the optimal ILS, while GRPO is 0.165 nats below.

### Experiment 2 — Hard environment (100 arms, 3 Type B, 10% tail probability)

| N | Method | E[r] | Best-of-16 | Best-of-N | ILS | π(B) |
|---|--------|------|-----------|-----------|-----|------|
| 1024 | GRPO | **0.504** | 0.764 | 0.911 | -2.155 | <0.1% |
| 1024 | Cont. MaxRL (unnorm) | 0.398 | **0.821** | **0.998** | **-1.581** | **31.5%** |

The method allocates 31.5% of policy mass to 3% of the arms (the ones with high ceiling despite low mean). GRPO never finds them regardless of training budget. The metric gaps are ~4× larger than in the moderate environment — the harder the setting, the more the method stands out.

### Experiment 3 — Contextual bandit with shared MLP (train/test split)

The MLP matches or slightly exceeds the tabular results (π(B) = 33.6% vs 29.1% at N=256), and generalizes perfectly to held-out prompts. Parameter sharing *helps*: a single tail event propagates through shared weights to inform the policy on all prompts simultaneously.

### The tradeoff

Continuous MaxRL does **not** dominate GRPO. In the hard environment the expected reward cost is ~21% at N=1024. A practitioner choosing this method must believe that best-of-N or pass@k at high thresholds is the right evaluation metric. If you care primarily about expected reward, use GRPO.

---

## Repository Contents

| File | Description |
|------|-------------|
| `experiment1_bandit.py` | Baseline synthetic bandit: easy/medium/hard tiers, 100 prompts, 10 arms per prompt |
| `experiment1b_hard_bandit.py` | Harder mean-vs-tail tradeoff with arm-type analysis and diagnostics |
| `experiment1c_scaling.py` | **Moderate environment** scaling sweep over `N` (paper Experiment 1) |
| `experiment1d_hard_bandit.py` | **Hard environment** scaling sweep over `N` (paper Experiment 2) |
| `experiment2_5_contextual.py` | **Contextual bandit** with shared MLP policy, train/test split (paper Experiment 3) |

The paper maps to code as follows:

| Paper section | Code file |
|---------------|-----------|
| §8 Exp 1 (Moderate Tradeoff) | `experiment1c_scaling.py` |
| §8 Exp 2 (Hard Discovery) | `experiment1d_hard_bandit.py` |
| §8 Exp 3 (Contextual Bandit) | `experiment2_5_contextual.py` |
| §8 Advantage Weight Analysis | Figure outputs from `experiment1d_hard_bandit.py` |

---

## Environment Setup

### Requirements

```
python >= 3.10
numpy
scipy
matplotlib
torch
```

### CPU-only

```bash
conda create -n cont-maxrl python=3.10 -y
conda activate cont-maxrl
pip install numpy scipy matplotlib torch
```

### GPU (CUDA)

```bash
conda create -n cont-maxrl-gpu python=3.10 -y
conda activate cont-maxrl-gpu
pip install numpy scipy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## Running Experiments

### Quick smoke test (CPU, ~5 seconds)

```bash
python experiment2_5_contextual.py \
  --device cpu --num-workers 1 --steps 1 --eval-every 1 \
  --n-values 4 --seeds 42 --output-dir results_smoke
```

### Experiment 1c — Moderate scaling (paper Experiment 1)

```bash
python experiment1c_scaling.py \
  --device cuda \
  --gpu-ids 0 1 2 3 \
  --num-workers 32 \
  --steps 2000 \
  --n-values 4 8 16 32 64 128 256 \
  --output-dir results_1c
```

Outputs:

- `results_1c/metrics.json` — full training histories
- `results_1c/metrics_summary.json` — aggregated final metrics by method and N
- `results_1c/fig1_scaling_curves.png` — main scaling figure (paper Fig. 1)
- `results_1c/fig2_learning_curves_by_N.png`
- `results_1c/fig3_advantage_distributions.png`
- `results_1c/fig4_weight_nonlinearity.png`

Regenerate figures without retraining:

```bash
python experiment1c_scaling.py \
  --plots-only \
  --output-dir results_1c \
  --metrics-path results_1c/metrics.json
```

### Experiment 1d — Hard scaling (paper Experiment 2)

```bash
python experiment1d_hard_bandit.py \
  --device cuda \
  --gpu-ids 0 1 2 3 \
  --num-workers 32 \
  --steps 3000 \
  --n-values 4 8 16 32 64 128 256 512 1024 \
  --output-dir results_1d
```

Outputs mirror `results_1c/` with the same figure naming. `fig4_weight_nonlinearity.png` is the advantage-vs-rank comparison showing the spike (paper Fig. on weight nonlinearity).

Regenerate figures:

```bash
python experiment1d_hard_bandit.py \
  --plots-only \
  --output-dir results_1d \
  --metrics-path results_1d/metrics.json
```

### Experiment 2.5 — Contextual bandit (paper Experiment 3)

```bash
python experiment2_5_contextual.py \
  --device cuda \
  --gpu-ids 0 1 2 3 \
  --num-workers 32 \
  --steps 5000 \
  --n-values 4 16 64 256 512 1024 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir results_2_5
```

Additional outputs beyond the scaling figures:

- `results_2_5/fig3_train_vs_test.png` — train/test comparison confirming generalization
- `results_2_5/fig4_per_prompt_analysis.png` — per-prompt π(B) vs best-of-16 scatter
- `results_2_5/checkpoints/{method}_N{N}_seed{seed}.pt` — saved policy checkpoints

### Experiment 1 and 1b (standalone, no scaling sweep)

```bash
python experiment1_bandit.py \
  --output-dir results \
  --steps 2000 \
  --log-every 50 \
  --eval-trials 100
```

```bash
python experiment1b_hard_bandit.py \
  --output-dir results_1b \
  --steps 2000 \
  --log-every 50 \
  --eval-trials 200
```

---

## Methods

All scaling experiments compare the following advantage-weighting schemes:

| Key | Label | Description |
|-----|-------|-------------|
| `grpo` | GRPO | `A_i = (r_i - mean) / std`, z-score normalization |
| `continuous_maxrl_unnormalized` | Cont. MaxRL (unnorm) | Gap-over-survivors weights, centered only |
| `continuous_maxrl_normalized` | Cont. MaxRL (norm) | Gap-over-survivors weights, centered and std-normalized |
| `binary_maxrl_*` | Binary MaxRL | Binary MaxRL at a fixed threshold (experiment-specific) |
| `reinforce_baseline` | REINFORCE+baseline | `A_i = r_i - mean` |

The **unnormalized** variant is theoretically cleaner and recommended. Dividing by the within-batch std compresses the weight spike, reducing the tail amplification that distinguishes MaxRL from GRPO. In the hard environment the normalized variant still works, but unnormalized should be the default.

---

## Metrics

| Metric | Definition |
|--------|------------|
| `expected_reward` | `E[r]` under the current policy |
| `best_of_16` | `E[max(r_1,...,r_16)]` from Monte Carlo rollouts |
| `best_of_n` | `E[max(r_1,...,r_N)]` where N is the training batch size |
| `integrated_log_survival` (ILS) | `∫ log P(r > τ) dτ` on a fixed τ grid; the population objective |
| `pass_at_05` / `pass_at_08` | `P(r > threshold)` at the specified threshold |
| `pi_type_a` / `pi_type_b` / `pi_type_c` | Policy mass allocated to each arm type |

Notes:
- `best_of_1 == expected_reward`
- Higher ILS is better (it is negative; closer to zero is better)
- `pi_type_b` is the key diagnostic for whether the method discovers tail arms. Uniform allocation gives `pi_type_b = (num Type B arms) / (total arms)`. A mean-seeking method drives it toward zero; continuous MaxRL pushes it well above baseline.

---

## Multi-GPU Usage

- `--num-workers` is the total number of worker processes across all selected GPUs
- Workers are distributed round-robin across `--gpu-ids`
- CUDA runs use the `spawn` multiprocessing context

Practical guidance: start with 4–8 workers per GPU and increase if stable. If you see OOM or contention, lower `--num-workers`.

---

## Reproducibility

- All experiments use fixed environment seeds and explicit per-run seeds
- Default seeds: `42 43 44 45 46`
- `metrics.json` stores the full config, environment parameters, and per-step histories
- `metrics_summary.json` (1c, 1d, 2.5 only) stores aggregated final metrics by method and N, suitable for tables

---

## Troubleshooting

**CUDA not detected**
- Confirm the env has a CUDA-enabled PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU visibility: `nvidia-smi`

**Plot legends missing**
- Regenerate with `--plots-only` using the existing `metrics.json`

**Large `metrics.json`**
- Use `metrics_summary.json` for reporting; it is auto-generated by the scaling experiments

**Disk quota**
- All results write to `/zfsauton/scratch/shrinivr/cont-maxrl/results*`
