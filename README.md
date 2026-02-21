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
│   │   ├── data_prep.py               # Multi-dataset preparation (APPS, TACO, CodeContests, LiveCodeBench)
│   │   └── sandbox.py                 # Sandboxed code execution
│   ├── scripts/
│   │   ├── dry_run.sh                 # Quick 3-step test (0.5B model, 2 GPUs)
│   │   ├── generate_run_scripts.py    # Generate all experiment scripts (252 train + 756 eval)
│   │   ├── runs/                      # Generated bash+SLURM scripts
│   │   ├── create_dry_run_data.py     # Tiny parquet dataset for testing
│   │   └── check_env.py               # Verify installation
│   └── tests/                         # 126 tests covering math, integration, gradients
│
├── vendor/verl/                       # Vendored verl (git submodule, patched)
├── data/                              # Training data (parquet)
└── results/                           # Output from training runs
```

---

## Quick Start

### 1. Clone and set up the environment

```bash
git clone --recurse-submodules <repo-url>
cd cont-maxrl

# Create conda env with Python 3.12
conda create -n cont-maxrl python=3.12 -y
conda activate cont-maxrl

# Install PyTorch (CUDA 12.8)
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install vllm + verl
pip install vllm==0.12.0
pip install -e vendor/verl

# Install project
pip install -e .

# Install remaining deps
pip install ray==2.54.0 "transformers>=4.45,<5" datasets hydra-core wandb
```

### 2. Verify installation

```bash
python experiment4_llm/scripts/check_env.py
pytest experiment4_llm/tests/ -q
```

### 3. Run the dry run (2 GPUs, ~20 minutes)

```bash
# Create tiny test data
python experiment4_llm/scripts/create_dry_run_data.py

# Run 3 training steps with 0.5B model
bash experiment4_llm/scripts/dry_run.sh
```

### 4. Prepare datasets

```bash
# Prepare all datasets (APPS, TACO, CodeContests, LiveCodeBench)
python experiment4_llm/src/data_prep.py --dataset all --output_dir data/

# Or prepare individually
python experiment4_llm/src/data_prep.py --dataset apps --output_dir data/
python experiment4_llm/src/data_prep.py --dataset taco --output_dir data/
python experiment4_llm/src/data_prep.py --dataset code_contests --output_dir data/
python experiment4_llm/src/data_prep.py --dataset livecodebench --output_dir data/
```

This downloads from HuggingFace and creates parquet files:

| Dataset | Train | Test | Source |
|---------|-------|------|--------|
| APPS | `apps_train.parquet` | `apps_test.parquet` | `codeparrot/apps` |
| TACO-verified | `taco_train.parquet` | `taco_val.parquet` (10% held out) | `likaixin/TACO-verified` |
| CodeContests | `code_contests_train.parquet` | `code_contests_test.parquet` | `deepmind/code_contests` |
| LiveCodeBench | — (eval-only) | `livecodebench_test.parquet` | `livecodebench/code_generation_lite` |

### 5. Run experiments

```bash
# Generate all scripts (252 train + 756 cross-benchmark eval)
python experiment4_llm/scripts/generate_run_scripts.py

# Run locally
bash experiment4_llm/scripts/runs/train_1.5b_apps_maxrl_unnorm_g16_s42.sh

# Or submit to SLURM (SBATCH headers are embedded in each script)
sbatch experiment4_llm/scripts/runs/train_1.5b_apps_maxrl_unnorm_g16_s42.sh

# Submit all 1.5B APPS experiments
for f in experiment4_llm/scripts/runs/train_1.5b_apps_*.sh; do sbatch "$f"; done

# Submit all 1.5B experiments (all datasets)
for f in experiment4_llm/scripts/runs/train_1.5b_*.sh; do sbatch "$f"; done
```

### 6. Evaluate (cross-benchmark)

Each trained model gets eval scripts for **every** benchmark:

```bash
# Evaluate APPS-trained model on all benchmarks
for f in experiment4_llm/scripts/runs/eval_1.5b_apps_maxrl_unnorm_g16_s42_on_*.sh; do
  sbatch "$f"
done

# Or run a single benchmark eval
bash experiment4_llm/scripts/runs/eval_1.5b_apps_maxrl_unnorm_g16_s42_on_livecodebench.sh
```

---

## Experiment 4: LLM Code Generation

### Overview

Train Qwen2.5-Coder models on programming problems using continuous rewards (fraction of test cases passed) instead of binary pass/fail across multiple datasets. Compare four advantage estimators:

| Method | Config key | Description |
|--------|-----------|-------------|
| GRPO | `grpo` | `A_i = (r_i - mean) / std` (z-score, mean-seeking) |
| MaxRL (unnorm) | `maxrl_unnorm` | Gap-over-survivors, mean-centered (tail-seeking) |
| MaxRL (norm) | `maxrl_norm` | Gap-over-survivors, mean-centered + std-normalized |
| REINFORCE+baseline | `reinforce_baseline` | `A_i = r_i - mean` |

### Datasets

| Dataset | Problems (train) | Problems (test) | Source | Use |
|---------|-----------------|-----------------|--------|-----|
| APPS | ~3K | ~2K | `codeparrot/apps` | Train + eval |
| TACO-verified | ~7.5K | — | `likaixin/TACO-verified` | Train only |
| CodeContests | ~3.7K | ~165 | `deepmind/code_contests` | Train + eval |
| LiveCodeBench | — | ~880 | `livecodebench/code_generation_lite` | Eval only |

### Experiment Matrix

Training: 3 datasets x (models x methods x G x seeds):

| Model | Methods | Group sizes (G) | Seeds | Train scripts |
|-------|---------|-----------------|-------|---------------|
| 1.5B (`Qwen2.5-Coder-1.5B-Instruct`) | all 4 | 4, 16, 32 | 42, 43, 44 | 108 |
| 7B (`Qwen2.5-Coder-7B-Instruct`) | all 4 | 4, 16, 32 | 42, 43, 44 | 108 |
| 32B (`Qwen2.5-Coder-32B-Instruct`) | grpo, maxrl_unnorm, maxrl_norm | 16, 32 | 42, 43 | 36 |

Each trained model is evaluated on **3 benchmarks** (APPS, CodeContests, LiveCodeBench).

**Total: 252 training + 756 evaluation = 1008 scripts**

### Training Hyperparameters (fixed across experiments)

| Parameter | Value |
|-----------|-------|
| Total steps | 300 |
| Batch size | 128 |
| Mini batch | 32 |
| PPO epochs | 1 |
| Learning rate | 1e-6 |
| KL coefficient | 0.001 |
| Clip ratio | 0.2 / 0.28 |
| Temperature (train) | 1.0 |
| Temperature (eval) | 0.8 |
| Max prompt length | 1024 |
| Max response length | 2048 |
| Checkpoint frequency | every 50 steps |

### Reward Function

The reward for each generated solution is **continuous** in `[0, 1]`:

```
reward = (number of test cases passed) / (total test cases)
```

Each test case is executed in a sandbox with a 10-second timeout and 512MB memory limit. This provides a richer training signal than binary pass/fail.

### Evaluation

After training, each checkpoint is evaluated by:

1. Generating 200 completions per test problem at temperature 0.8
2. Executing each completion against all test cases (binary: all pass or not)
3. Computing unbiased pass@k for k in {1, 2, 5, 10, 20, 50, 100}

### Compute Requirements

| Model | GPUs | SLURM memory | Time estimate |
|-------|------|-------------|---------------|
| 1.5B | 4 | 64 GB | ~8 hours |
| 7B | 4 | 128 GB | ~20 hours |
| 32B | 8 | 384 GB | ~36 hours |

### Generated Script Naming Convention

```
experiment4_llm/scripts/runs/
├── train_{model}_{dataset}_{method}_g{G}_s{seed}.sh
└── eval_{model}_{dataset}_{method}_g{G}_s{seed}_on_{benchmark}.sh
```

Examples:
- `train_1.5b_apps_maxrl_unnorm_g16_s42.sh` — train on APPS
- `train_1.5b_taco_maxrl_unnorm_g16_s42.sh` — train on TACO
- `eval_1.5b_apps_maxrl_unnorm_g16_s42_on_livecodebench.sh` — eval APPS-trained model on LiveCodeBench

Each script works both as a standalone bash script (`bash script.sh`) and as a SLURM job (`sbatch script.sh`). SLURM directives are embedded as `#SBATCH` comments at the top.

### Results Directory Structure

```
results/
└── 1.5b_apps_maxrl_unnorm_g16_s42/
    ├── global_step_50/               # Checkpoint at step 50
    ├── global_step_100/              # Checkpoint at step 100
    ├── ...
    ├── slurm_12345.out               # SLURM stdout
    ├── eval_apps/                    # Same-dataset eval
    │   ├── completions.json
    │   └── eval_results.json
    ├── eval_code_contests/           # Cross-dataset eval on CodeContests
    └── eval_livecodebench/           # Cross-dataset eval on LiveCodeBench
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

Regenerate figures without retraining:

```bash
python experiment1c_scaling.py --plots-only --output-dir results_1c --metrics-path results_1c/metrics.json
```

---

## Key Results (Bandit Experiments)

### Experiment 1 — Moderate environment (50 arms, 5 Type B)

| N | Method | E[r] | Best-of-N | ILS | π(B) |
|---|--------|------|-----------|-----|------|
| 256 | GRPO | **0.806** | 0.962 | -0.770 | 0.0% |
| 256 | Cont. MaxRL (unnorm) | 0.766 | **0.987** | **-0.627** | **8.9%** |
| — | ILS-Optimal | 0.753 | 0.992 | -0.605 | 12.8% |

### Experiment 2 — Hard environment (100 arms, 3 Type B)

| N | Method | E[r] | Best-of-16 | Best-of-N | ILS | π(B) |
|---|--------|------|-----------|-----------|-----|------|
| 1024 | GRPO | **0.504** | 0.764 | 0.911 | -2.155 | <0.1% |
| 1024 | MaxRL (unnorm) | 0.398 | **0.821** | **0.998** | **-1.581** | **31.5%** |

---

## Methods

| Key | Label | Description |
|-----|-------|-------------|
| `grpo` | GRPO | `A_i = (r_i - mean) / std`, z-score normalization |
| `maxrl_unnorm` | Cont. MaxRL (unnorm) | Gap-over-survivors weights, centered only |
| `maxrl_norm` | Cont. MaxRL (norm) | Gap-over-survivors weights, centered and std-normalized |
| `reinforce_baseline` | REINFORCE+baseline | `A_i = r_i - mean` |

The **unnormalized** variant is theoretically cleaner and recommended. Dividing by the within-batch std compresses the weight spike, reducing the tail amplification that distinguishes MaxRL from GRPO.

---

## Tests

```bash
# Run all 126 tests
pytest experiment4_llm/tests/ -q

# Test categories:
#   test_advantages_core.py          - Worked examples, edge cases, algebraic properties
#   test_advantages_cross_method.py  - Relationships between methods
#   test_advantages_batch.py         - Batch-level behavior
#   test_reward.py                   - Code extraction, partial scoring
#   test_sandbox.py                  - Sandboxed execution
#   test_integration_critical.py     - PPO gradient direction, verl wrapper, dry run verification
#   test_verl_integration.py         - Custom estimator enum + dispatch
```

---

## Technical Notes

### verl Integration

Custom advantage estimators are integrated directly into the vendored verl source (`vendor/verl/verl/trainer/ppo/ray_trainer.py`). This is necessary because verl uses Ray actors, and monkey-patching in the driver process does not propagate to worker processes.

The dispatch table lives in `experiment4_llm/src/verl_register.py` and is imported by `ray_trainer.py` at compute-advantage time. The `PYTHONPATH` must include the repo root for this import to work (all generated scripts set this).

### Environment Requirements

- The generated scripts use the full Python path (`/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python`) rather than `conda activate` to avoid PATH contamination.
- `RAY_TMPDIR` and `TMPDIR` are set to `/tmp/` (local disk) to avoid NFS socket path limits and cleanup issues.
- `PYTHONNOUSERSITE=1` prevents user-site packages from interfering.

### Data Format

verl expects **parquet** files with columns:
- `data_source`: string identifier (e.g., `"apps"`)
- `prompt`: list of chat message dicts (`[{"role": "system", ...}, {"role": "user", ...}]`)
- `reward_model`: dict with `"ground_truth"` key containing JSON test cases

The `data_prep.py` script handles the conversion from HuggingFace datasets (APPS, TACO, CodeContests, LiveCodeBench) to this format.

---

## Troubleshooting

**`No available memory for the cache blocks` (vLLM OOM)**
- Reduce `gpu_memory_utilization` in the script or ensure GPUs have enough free memory.
- Check `nvidia-smi` — other processes may be using GPU memory.

**`Ray AF_UNIX socket path too long`**
- Already fixed: scripts use `RAY_TMPDIR=/tmp/ray_${USER}` to avoid long NFS paths.

**`ImportError: cannot import name 'experiment4_llm'`**
- Ensure `PYTHONPATH` includes the repo root. All generated scripts set this automatically.

**`NotImplementedError` in `compute_advantage`**
- The custom estimators must be patched into `vendor/verl/verl/trainer/ppo/ray_trainer.py`. Run `git submodule update --init` and verify the patches exist.

**NFS `.nfs*` cleanup errors during shutdown**
- Harmless. Set `TMPDIR=/tmp/${USER}_verl_tmp` to avoid (already done in generated scripts).
