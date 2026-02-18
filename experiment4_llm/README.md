# Experiment 4: Continuous MaxRL for LLM Code Generation

This directory implements the core software needed to run the Experiment 4 specification:

- Continuous reward (`fraction of test cases passed`) for code tasks.
- Four advantage estimators:
  - `grpo`
  - `maxrl_unnorm`
  - `maxrl_norm`
  - `reinforce_baseline`
- Pass@k evaluation with the unbiased combinatorial estimator.
- APPS dataset conversion utilities for verl-style JSONL.

## Directory Layout

- `configs/`: Training templates for Qwen2.5-Coder 1.5B, 7B, 32B.
  - 1.5B/7B include `grpo`, `maxrl_unnorm`, `maxrl_norm`, `reinforce`.
  - 32B intentionally includes `grpo`, `maxrl_unnorm`, `maxrl_norm` only.
- `src/advantages.py`: Advantage functions and grouping helpers.
- `src/sandbox.py`: Best-effort local sandbox runner (timeout + memory caps).
- `src/reward_function.py`: Continuous reward implementation.
- `src/data_prep.py`: APPS conversion/filtering utilities.
- `src/evaluate.py`: Pass@k aggregation and summary generation.
- `src/evaluate_completions.py`: Raw completion execution + aggregation into problem-level rows.
- `src/custom_adv_estimator.py`: Token-level advantage wiring for trainer integration.
- `scripts/`: Training/eval/figure entrypoints.
- `tests/`: Unit tests for advantages, reward computation, and sandbox behavior.

## Quick Start

### 0) Check environment dependencies

```bash
python experiment4_llm/scripts/check_env.py
```

This checks required modules (`torch`, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `datasets`, `transformers`, `accelerate`) and optional modules (`verl`, `vllm`, `pytest`).

### 1) Convert APPS to training JSONL

```bash
python -m experiment4_llm.src.data_prep \
  --split train \
  --difficulties introductory interview \
  --min-test-cases 5 \
  --output-path data/apps_train_intro_interview.jsonl
```

### 2) Run one training config (verl command can be overridden)

```bash
chmod +x experiment4_llm/scripts/run_training.sh
experiment4_llm/scripts/run_training.sh experiment4_llm/configs/qwen_1.5b_grpo.yaml
```

If your trainer entrypoint differs:

```bash
VERL_TRAIN_CMD="python -m verl.trainer.main_ppo" \
experiment4_llm/scripts/run_training.sh experiment4_llm/configs/qwen_1.5b_maxrl_unnorm.yaml
```

If `verl` is not installed, `run_training.sh` exits early with a clear error.

### 3) Run matrix launcher

```bash
chmod +x experiment4_llm/scripts/run_sweep.sh
# default is DRY_RUN=1
DRY_RUN=0 experiment4_llm/scripts/run_sweep.sh
```

`qwen_32b_reinforce.yaml` is intentionally not provided because the 32B sweep uses only the 3-method matrix (`grpo`, `maxrl_unnorm`, `maxrl_norm`).

### 4) Evaluate and plot pass@k curves

Problem-level input format (one row per problem per run):

```json
{
  "model": "qwen_7b",
  "method": "maxrl_unnorm",
  "seed": 42,
  "group_size": 16,
  "problem_id": "apps_1234",
  "binary_passes": [0, 0, 1, 0, 1],
  "continuous_rewards": [0.2, 0.4, 1.0, 0.1, 1.0]
}
```

Run from problem-level rows:

```bash
chmod +x experiment4_llm/scripts/run_evaluation.sh
experiment4_llm/scripts/run_evaluation.sh /path/to/completions.jsonl results_experiment4/eval
```

Raw completion-level input is also supported:

```json
{
  "model": "qwen_7b",
  "method": "maxrl_unnorm",
  "seed": 42,
  "group_size": 16,
  "problem_id": "apps_1234",
  "generated_code": "def solve(...): ...",
  "test_cases": {"inputs": [...], "outputs": [...], "fn_name": "solve"}
}
```

Run raw-completion evaluation:

```bash
experiment4_llm/scripts/run_evaluation.sh /path/to/raw_completions.jsonl results_experiment4/eval --raw-completions
```

If you have per-checkpoint evaluation logs, include them to generate Figure B:

```bash
experiment4_llm/scripts/run_evaluation.sh \
  /path/to/completions.jsonl \
  results_experiment4/eval \
  --training-metrics-jsonl /path/to/training_metrics.jsonl
```

Expected training-metrics JSONL rows include `model`, `method`, `seed`, `group_size`, a step field (`step` or `global_step`), and pass metrics (`pass_at_k` with keys `1` and `10`, or flat keys like `pass@1` / `pass@10`).

This executes each completion in the sandbox, computes continuous reward and binary pass, then aggregates pass@k summaries and figures.

Generated figures from `scripts/generate_figures.py` are:

- Figure A: pass@k curves with crossover stars.
- Figure B: training curves (pass@1 and pass@10 over step) when `--training-metrics-jsonl` is provided.
- Figure C: reward histograms.
- Figure D: MaxRL normalized vs unnormalized comparison.
- Figure E: scaling with group size `G` (pass@10 vs `G`).

## Advantage API

`experiment4_llm/src/advantages.py` is the canonical implementation.

- `grpo_advantage(rewards)`
- `maxrl_unnormalized_advantage(rewards)`
- `maxrl_normalized_advantage(rewards)`
- `reinforce_baseline_advantage(rewards)`

Each takes one prompt group `rewards: [G]` and returns `advantages: [G]`.

For batched/grouped usage:

- `compute_grouped_advantages(rewards: [num_prompts, G], method=...)`
- `compute_advantages_from_group_indices(rewards: [batch], group_indices: [batch], method=...)`

## Verifying Core Logic

Run unit tests:

```bash
python -m unittest discover -s experiment4_llm/tests -p "test_*.py"
```

Run full smoke checks (env check + tests + self-test + mini end-to-end pipeline):

```bash
chmod +x experiment4_llm/scripts/smoke_run.sh
PYTHON_BIN=python experiment4_llm/scripts/smoke_run.sh results_experiment4/smoke
```

Checks included:

- MaxRL gap-over-survivors weighting against known reference vectors.
- MaxRL implementation equivalence with bandit Experiment 1d functions.
- Zero-variance behavior (all methods return zero advantages).
- Continuous reward values for both function-call and stdin/stdout formats.
- Timeout/memory constraints in sandbox subprocess execution.

## Notes on Sandbox Security

The local subprocess sandbox is intentionally simple and intended for development.
For production-scale training, use a hardened runtime (e.g. sandbox-fusion, containerized judge).

## Mapping to Spec

- Continuous training reward: `src/reward_function.py`
- MaxRL estimator variants: `src/advantages.py`
- Per-sequence-to-token advantage broadcast: `src/custom_adv_estimator.py`
- Unbiased pass@k estimator: `src/evaluate.py`
- Raw completion execution for evaluation: `src/evaluate_completions.py`
