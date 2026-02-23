# Continuous Maximum Likelihood Reinforcement Learning

**Extending MaxRL to Continuous Rewards via Integrated Survival Functions**

---

## Overview

This repository contains all code for the paper *Continuous Maximum Likelihood Reinforcement Learning*. It includes:

- **Bandit experiments** (Experiments 1-3): synthetic environments demonstrating the theoretical framework
- **LLM code generation** (Experiment 4): training Qwen2.5-Coder models on APPS, TACO-verified, and CodeContests with continuous rewards via [verl](https://github.com/volcengine/verl), evaluated on APPS, CodeContests, and LiveCodeBench

**The core idea in one sentence:** replace pass@k with `max_reward@k = E[max(r_1, ..., r_k)]`, and the entire MaxRL framework carries over — including the harmonic-series convergence argument, the closed-form estimator, and the integrated log-survival limit.

---

## Background

### Binary MaxRL (Tajwar et al., 2026)

Binary MaxRL is defined for rewards `r ∈ {0, 1}`. It constructs a family of objectives:

```
J_binary^(T) = sum_{k=1}^{T} (1/k) * pass@k
```

where `pass@k = 1 - (1-p)^k` and `p = P(r = 1)`. As `T → ∞`, the gradient converges to `∇ log p` — the maximum likelihood gradient.

### This Paper: Continuous MaxRL

For continuous rewards, the survival function `p_τ = P(r > τ)` plays the role of `p`. We define:

```
max_reward@k = E[max(r_1, ..., r_k)] = ∫_0^1 [1 - (1 - p_τ)^k] dτ
```

As `T → ∞`, the gradient converges to the **integrated log-survival objective**:

```
J_ML^cont = ∫_0^1 log P(r > τ) dτ
```

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

**Binary recovery:** When `r ∈ {0, 1}`, this reduces exactly to `w_(i) = 1/K` for successes and `0` for failures (where `K` is the number of successes).

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/stablegradients/cont-maxrl.git
cd cont-maxrl
```

### 2. Install the environment

The setup script creates a conda env with Python 3.10, CUDA 12.4 toolkit, PyTorch 2.6.0, flash-attention 2.7.4, vLLM 0.8.3, verl 0.2.0 (vendored with patches), and all dependencies.

```bash
bash experiment4_llm/setup.sh
conda activate cont-maxrl
```

This takes ~30 minutes (flash-attention builds from source for sm_86). If your GPUs are not A5000/A6000, edit `TORCH_CUDA_ARCH_LIST` in `setup.sh` before running (e.g. `"8.0"` for A100, `"9.0"` for H100).

### 3. Verify installation

```bash
python experiment4_llm/scripts/check_env.py
pytest experiment4_llm/tests/ -q
```

All 179 tests should pass.

### 4. Download and prepare datasets

```bash
bash experiment4_llm/scripts/download_data.sh
```

This downloads from HuggingFace, deduplicates, and creates parquet files in `data/`:

| Dataset | Train | Val/Test | Source |
|---------|-------|----------|--------|
| APPS | 1,980 (`apps_train.parquet`) | 3,393 (`apps_test.parquet`) | `codeparrot/apps` |
| TACO-verified | 5,187 (`taco_train.parquet`) | 576 (`taco_val.parquet`, 10% held out) | `likaixin/TACO-verified` |
| CodeContests | 11,530 (`code_contests_train.parquet`) | 162 (`code_contests_test.parquet`) | `deepmind/code_contests` |
| LiveCodeBench | — (eval-only) | 529 (`livecodebench_test.parquet`) | `livecodebench/code_generation_lite` |

### 5. Download model weights

```bash
bash experiment4_llm/scripts/download_models.sh
```

Pre-downloads Qwen2.5-Coder weights (1.5B, 7B, 32B) so training doesn't block on HF downloads.

### 6. Generate experiment scripts

```bash
python experiment4_llm/scripts/generate_run_scripts.py
```

Generates 123 training + 369 evaluation scripts in `experiment4_llm/scripts/runs/`.

**Important:** The generated scripts contain hardcoded paths (`REPO_ROOT`, `PYTHON`, `CUDA_HOME`). If running on a different machine, edit the three path variables at the top of `generate_run_scripts.py` and re-run it.

### 7. Run training

```bash
# Run locally (SBATCH lines are ignored)
bash experiment4_llm/scripts/runs/train_1.5b_taco_grpo_g4_s42.sh

# Or submit to SLURM
sbatch experiment4_llm/scripts/runs/train_1.5b_taco_grpo_g4_s42.sh

# Submit all 1.5B TACO experiments
for f in experiment4_llm/scripts/runs/train_1.5b_taco_*.sh; do sbatch "$f"; done
```

### 8. Evaluate (cross-benchmark)

After training, evaluate each checkpoint on all benchmarks:

```bash
# Evaluate TACO-trained model on all benchmarks
for f in experiment4_llm/scripts/runs/eval_1.5b_taco_grpo_g4_s42_on_*.sh; do
  sbatch "$f"
done

# Or run a single benchmark eval
bash experiment4_llm/scripts/runs/eval_1.5b_taco_grpo_g4_s42_on_livecodebench.sh
```

Each eval script generates 200 completions per test problem at temperature 0.8 and computes unbiased pass@k for k in {1, 2, 5, 10, 20, 50, 100}.

---

## Experiment 4: LLM Code Generation

### Methods

| Method | Config key | Description |
|--------|-----------|-------------|
| GRPO | `grpo` | `A_i = (r_i - mean) / std` (z-score, mean-seeking) |
| MaxRL (unnorm) | `maxrl_unnorm` | Gap-over-survivors, mean-centered (tail-seeking) |
| MaxRL (norm) | `maxrl_norm` | Gap-over-survivors, mean-centered + std-normalized |
| REINFORCE+baseline | `reinforce_baseline` | `A_i = r_i - mean` |

### Experiment Matrix

| Model | Methods | Group sizes (G) | Seeds | Train scripts |
|-------|---------|-----------------|-------|---------------|
| 1.5B (`Qwen2.5-Coder-1.5B-Instruct`) | all 4 | 4, 16, 32, 64 | 42 | 48 |
| 7B (`Qwen2.5-Coder-7B-Instruct`) | all 4 | 4, 16, 32, 64 | 42 | 48 |
| 32B (`Qwen2.5-Coder-32B-Instruct`) | grpo, maxrl_unnorm, maxrl_norm | 16, 32, 64 | 42 | 27 |

Each trained model is evaluated on **3 benchmarks** (APPS, CodeContests, LiveCodeBench) for cross-dataset generalization.

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Total steps | 150 |
| PPO epochs | 1 |
| Learning rate | 1e-6 |
| KL coefficient (beta) | 0.001 |
| Clip ratio | 0.2 / 0.28 |
| Temperature (train) | 1.0 |
| Temperature (eval) | 0.8 |
| Max prompt length | 1024 |
| Max response length | 1024 |
| Checkpoint frequency | every 25 steps |
| Validation frequency | every 20 steps |
| FSDP | full sharding (`fsdp_size=-1`) |

Batch sizes depend on G (group size):
- G=4: batch_size=1024, 4096 rollouts/step
- G=16: batch_size=128, 2048 rollouts/step
- G=32: batch_size=64, 2048 rollouts/step
- G=64: batch_size=32, 2048 rollouts/step

### Reward Function

The reward for each generated solution is **continuous** in `[0, 1]`:

```
reward = (number of test cases passed) / (total test cases)
```

Each test case is executed in a sandbox with a 1-second timeout and 512MB memory limit.

### Inline Validation

During training (every 20 steps), the model generates G completions per validation problem with sampling (temperature 1.0). Metrics logged to wandb:

- `val-core/{dataset}/reward/mean@G` — mean reward across G completions
- `val-core/{dataset}/reward/best@G/mean` — mean best-of-G reward (best@k)

### Post-Training Evaluation

After training, standalone eval scripts generate 200 completions per test problem at temperature 0.8 and compute unbiased pass@k for k in {1, 2, 5, 10, 20, 50, 100}.

### Script Naming Convention

```
experiment4_llm/scripts/runs/
├── train_{model}_{dataset}_{method}_g{G}_s{seed}.sh
└── eval_{model}_{dataset}_{method}_g{G}_s{seed}_on_{benchmark}.sh
```

Examples:
- `train_1.5b_taco_maxrl_unnorm_g16_s42.sh` — train on TACO
- `eval_1.5b_taco_maxrl_unnorm_g16_s42_on_livecodebench.sh` — eval TACO-trained model on LiveCodeBench

### Results Directory

```
results/
└── 1.5b_taco_maxrl_unnorm_g16_s42/
    ├── global_step_25/               # Checkpoint (model + optimizer shards)
    ├── global_step_50/
    ├── ...
    ├── latest_checkpointed_iteration.txt
    ├── slurm_*.out / slurm_*.err
    ├── eval_apps/
    │   ├── completions.json
    │   └── eval_results.json
    ├── eval_code_contests/
    └── eval_livecodebench/
```

---

## Repository Structure

```
cont-maxrl/
├── README.md
├── pyproject.toml
│
├── experiment1_bandit.py              # Baseline synthetic bandit
├── experiment1b_hard_bandit.py        # Hard mean-vs-tail tradeoff
├── experiment1c_scaling.py            # Moderate environment N-scaling (paper Exp 1)
├── experiment1d_hard_bandit.py        # Hard environment N-scaling (paper Exp 2)
├── experiment2_5_contextual.py        # Contextual bandit with MLP (paper Exp 3)
│
├── experiment4_llm/                   # LLM code generation (paper Exp 4)
│   ├── src/
│   │   ├── advantages.py             # Core advantage estimators
│   │   ├── reward_function.py         # Continuous reward (fraction of tests passed)
│   │   ├── verl_register.py           # verl integration (estimator dispatch table)
│   │   ├── train.py                   # Hydra entrypoint for training
│   │   ├── evaluate.py                # Evaluation + pass@k computation
│   │   ├── data_prep.py               # Dataset preparation with deduplication
│   │   └── sandbox.py                 # Sandboxed code execution (1s timeout)
│   ├── scripts/
│   │   ├── setup.sh                   # Full environment setup (conda + CUDA + all deps)
│   │   ├── download_data.sh           # Download and prepare all datasets
│   │   ├── download_models.sh         # Pre-download model weights
│   │   ├── generate_run_scripts.py    # Generate all experiment scripts
│   │   ├── check_env.py               # Verify installation
│   │   ├── dry_run.sh                 # Quick 3-step test (0.5B model, 2 GPUs)
│   │   ├── create_dry_run_data.py     # Tiny parquet dataset for testing
│   │   └── runs/                      # Generated bash+SLURM scripts
│   └── tests/                         # 179 tests
│
├── vendor/verl/                       # Vendored verl 0.2.0.dev (patched)
├── data/                              # Training/eval data (parquet, not in git)
└── results/                           # Training outputs (not in git)
```

---

## Bandit Experiments (Experiments 1-3)

### Requirements (CPU-only)

```bash
conda create -n cont-maxrl-bandits python=3.10 -y
conda activate cont-maxrl-bandits
pip install numpy scipy matplotlib torch
```

### Running

```bash
# Quick smoke test (~5 seconds)
python experiment2_5_contextual.py \
  --device cpu --num-workers 1 --steps 1 --eval-every 1 \
  --n-values 4 --seeds 42 --output-dir results_smoke

# Experiment 1 — Moderate scaling (paper Exp 1)
python experiment1c_scaling.py \
  --device cuda --gpu-ids 0 1 2 3 --num-workers 32 \
  --steps 2000 --n-values 4 8 16 32 64 128 256 \
  --output-dir results_1c

# Experiment 2 — Hard scaling (paper Exp 2)
python experiment1d_hard_bandit.py \
  --device cuda --gpu-ids 0 1 2 3 --num-workers 32 \
  --steps 3000 --n-values 4 8 16 32 64 128 256 512 1024 \
  --output-dir results_1d

# Experiment 3 — Contextual bandit (paper Exp 3)
python experiment2_5_contextual.py \
  --device cuda --gpu-ids 0 1 2 3 --num-workers 32 \
  --steps 5000 --n-values 4 16 64 256 512 1024 \
  --batch-size 32 --lr 1e-3 --output-dir results_2_5
```

### Outputs

Each experiment produces:
- `metrics.json` — full training histories
- `metrics_summary.json` — aggregated final metrics by method and N
- `fig1_scaling_curves.png` through `fig4_*.png` — paper figures

---

## Vendored verl Patches

verl 0.2.0.dev is vendored at `vendor/verl/` with the following patches:

| File | Patch |
|------|-------|
| `verl/trainer/ppo/ray_trainer.py` | Added `MAXRL_UNNORM`, `MAXRL_NORM`, `REINFORCE_BASELINE` to `AdvantageEstimator` enum; dispatch to custom estimators in `experiment4_llm.src.verl_register` |
| `verl/workers/fsdp_workers.py` | `AutoModelForVision2Seq` → `AutoModelForImageTextToText` (transformers compat); guard `reshard(True)` calls with `uses_sharded_strategy` check (prevents crash with NO_SHARD) |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | Same AutoModel rename |

---

## Troubleshooting

**`No available memory for the cache blocks` (vLLM OOM)**
- Reduce `gpu_memory_utilization` in the script or ensure GPUs have enough free memory.
- Check `nvidia-smi` — other processes may be using GPU memory.

**`AssertionError: Expects sharded strategy` (FSDP reshard)**
- Already fixed in vendored verl. The `reshard(True)` calls are guarded with `uses_sharded_strategy`.

**CUDA OOM during `optimizer.step()` for 1.5B on 24GB GPUs**
- `fsdp_size=1` (NO_SHARD) is impossible on 24GB GPUs for 1.5B — model + optimizer + gradients = 24GB with no room for activations.
- Use `fsdp_size=-1` (FULL_SHARD). This is already the default.

**`Ray AF_UNIX socket path too long`**
- Already fixed: scripts use `RAY_TMPDIR=/tmp/ray_${USER}` to avoid long NFS paths.

**`ImportError: cannot import name 'experiment4_llm'`**
- Ensure `PYTHONPATH` includes the repo root. All generated scripts set this automatically.

**Flash-attention build fails**
- Ensure `CUDA_HOME` points to conda's CUDA 12.4, not the system CUDA. The setup script handles this.
- If your GPUs are not sm_86, edit `TORCH_CUDA_ARCH_LIST` in `setup.sh`.

**Training resumes from old checkpoint instead of starting fresh**
- verl auto-resumes from `results/{experiment}/latest_checkpointed_iteration.txt`. Delete or rename the results directory to start fresh.

---

## Tests

```bash
# Run all 179 tests
pytest experiment4_llm/tests/ -q

# Test categories:
#   test_advantages_core.py          - Worked examples, edge cases, algebraic properties
#   test_advantages_cross_method.py  - Relationships between methods
#   test_advantages_batch.py         - Batch-level behavior
#   test_reward.py                   - Code extraction, partial scoring
#   test_sandbox.py                  - Sandboxed execution
#   test_integration_critical.py     - PPO gradient direction, verl wrapper
#   test_verl_integration.py         - Custom estimator enum + dispatch
#   test_data_prep.py                - Dataset preparation + deduplication
#   test_generate_scripts.py         - Script generation
```
