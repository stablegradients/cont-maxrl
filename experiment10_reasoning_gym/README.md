# experiment10_reasoning_gym

Train Qwen3-0.6B-Base on [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
tasks via verl, comparing GOS vs GRPO vs MaxRL vs RLOO across tasks and
difficulty levels. Implements spec `spec_reasoning_gym.md`.

Scientific question: *does the performance gap between continuous reward and
binary reward grow as task difficulty increases and binary reward becomes
sparser?*

---

## Table of contents

1. [Layout](#layout)
2. [Installation](#installation) — what you need on top of an existing `experiment4_llm` env
3. [Quick start](#quick-start) — end-to-end one-task workflow
4. [Generating data](#generating-data)
5. [Reward functions](#reward-functions)
6. [Pass@k baseline](#passk-baseline)
7. [Training](#training)
   - [Single run](#single-run)
   - [Estimator name mapping](#estimator-name-mapping)
   - [Per-task lengths](#per-task-lengths)
   - [Tunable env vars](#tunable-env-vars)
   - [Hydra overrides](#hydra-overrides)
   - [Cross-difficulty validation](#cross-difficulty-validation)
   - [Outputs and W&B](#outputs-and-wb)
8. [Multi-run dispatchers](#multi-run-dispatchers)
   - [`launch_all.sh` — Sweeps A & B](#launch_allsh--sweeps-a--b)
   - [`dispatch_wssr.sh` — WS+SR waves](#dispatch_wssrsh--wssr-waves)
   - [`launch_slurm.sh` — SLURM wrapper](#launch_slurmsh--slurm-wrapper)
9. [Benchmark scripts](#benchmark-scripts)
10. [Plots](#plots)
11. [Tests](#tests)
12. [Adding a new task or difficulty](#adding-a-new-task-or-difficulty)
13. [Troubleshooting](#troubleshooting)
14. [Deviations from spec and open questions](#deviations-from-spec-and-open-questions)

---

## Layout

```
experiment10_reasoning_gym/
├── README.md
├── src/
│   ├── generate_data.py            # produce train/test parquets per (task, difficulty)
│   ├── reward_manager.py           # compute_score_continuous / _binary (+ K&K fixed reward)
│   ├── compute_pass_at_k.py        # vLLM pass@k for one (task, difficulty)
│   ├── pass_at_k_sweep.py          # multi-config pass@k that reuses one vLLM engine
│   ├── plot_difficulty_scaling.py  # Sweep A plot
│   └── plot_reward_ablation.py     # Sweep B plot
├── scripts/
│   ├── launch.sh                   # single-run entry point (see §Training)
│   ├── launch_all.sh               # batch launcher for Sweep A + Sweep B
│   ├── launch_slurm.sh             # SLURM wrapper for one run (GHX4 / H200)
│   ├── dispatch_wssr.sh            # 8-GPU wave dispatcher for the WS+SR sweep
│   ├── benchmark_1gpu8x.sh         # 8 single-GPU runs in parallel
│   ├── benchmark_2per_gpu.sh       # 16 runs, 2 per GPU, MPS-shared
│   ├── benchmark_parallel.sh       # 4 runs, 2 GPUs each
│   └── run_tests.sh                # pytest wrapper
├── tests/                          # 66 unit + integration tests
├── data/                           # generated parquets (gitignored)
│   └── {task}_diff{N}/{train,test}.parquet
└── results/
    └── pass_at_k/                  # pass@k JSON per (task, difficulty)
```

---

## Installation

**Assumption:** you already have a working `experiment4_llm` environment (the
shared conda env on this stack — usually called `cont-maxrl`, `maxrl`, or
similar). It already provides torch 2.6.0+cu124, vllm 0.8.4, ray 2.54.0,
transformers 4.57.6, flash-attn, wandb, and an editable verl checkout. Those
are *all* of the heavy dependencies experiment 10 needs.

Marginal install on top is just one package:

```bash
conda activate <your-exp4-env>           # e.g. cont-maxrl, maxrl, rgym
pip install 'reasoning_gym>=0.1.19'      # 0.1.19 was used for the spec; 0.1.25 also works
```

Verify:

```bash
python -c "import reasoning_gym, verl, vllm, torch; \
           print('rg', reasoning_gym.__version__, '| verl', verl.__file__, \
                 '| vllm', vllm.__version__, '| torch', torch.__version__)"
```

You should see `rg 0.1.19` (or `0.1.25`), verl pointing at your `vendor/verl`
checkout (or wherever your exp4 env loads it from), vllm `0.8.4`, torch
`2.6.0+cu124`.

Then put the repo on the Python path (every entry point assumes this):

```bash
export PYTHONPATH=/path/to/cont-maxrl:$PYTHONPATH
```

`launch.sh` exports it for you, but you'll want it in your shell to run the
data generators, tests, and pass@k scripts directly.

**Optional:** if you keep an `.env` at repo root with `WANDB_API_KEY`,
`HF_TOKEN`, `WANDB_ENTITY`, `launch.sh` will source it automatically. Without
`WANDB_API_KEY`, set `WANDB_MODE=offline` to skip cloud logging.

---

## Quick start

End-to-end, single task, 1 GPU, ~30 minutes (excluding data gen):

```bash
# 0. Activate env and put repo on path
conda activate <your-exp4-env>
export PYTHONPATH=/path/to/cont-maxrl:$PYTHONPATH
cd /path/to/cont-maxrl

# 1. Generate one (task, difficulty) pair
python -m experiment10_reasoning_gym.src.generate_data \
    --task word_sorting --difficulty 8 \
    --train_size 20000 --test_size 1000 --seed 42

# 2. (Optional) sanity-check base-model sparsity at this difficulty
python -m experiment10_reasoning_gym.src.compute_pass_at_k \
    --task word_sorting --difficulty 8 \
    --num_prompts 100 --num_samples 32 --k_values 1,4,16,32

# 3. Train one estimator+reward combo
WANDB_MODE=offline \
bash experiment10_reasoning_gym/scripts/launch.sh \
    word_sorting 8 gos continuous 42

# 4. Watch it
tail -f checkpoints/reasoning_gym_word_sorting/word_sorting_gos_continuous_diff8_s42/*.log
# (or, if launched directly, just watch stdout)
```

Outputs land at:

- Parquets: `experiment10_reasoning_gym/data/word_sorting_diff8/{train,test}.parquet`
- Pass@k JSON: `experiment10_reasoning_gym/results/pass_at_k/word_sorting_diff8.json`
- Checkpoints: `checkpoints/reasoning_gym_word_sorting/word_sorting_gos_continuous_diff8_s42/`
- W&B run: project `reasoning_gym_word_sorting`, name `word_sorting_gos_continuous_diff8_s42`

---

## Generating data

```bash
python -m experiment10_reasoning_gym.src.generate_data \
    --task <word_sorting|sentence_reordering|knights_knaves|game_of_life> \
    --difficulty <int> \
    --train_size 20000 --test_size 1000 --seed 42
```

Difficulty maps to one of these reasoning_gym kwargs (see
`src/generate_data.py::_difficulty_kwargs`):

| task                | difficulty controls                                          |
|---------------------|--------------------------------------------------------------|
| word_sorting        | `min_words = max_words = difficulty`                         |
| sentence_reordering | `min_words_in_sentence = max_words_in_sentence = difficulty` |
| knights_knaves      | `n_people = difficulty`                                      |
| game_of_life        | `grid_size_x = grid_size_y = difficulty`                     |

Output lands at `experiment10_reasoning_gym/data/{task}_diff{N}/{train,test}.parquet`.
Each row has the verl schema:

```python
{
  "data_source": "reasoning_gym_{task}_diff{N}",   # diff-aware so verl splits per-diff val metrics
  "prompt":      [{"role": "user", "content": "<question> + Put your answer within <answer> </answer> tags."}],
  "reward_model": {"ground_truth": "<oracle answer string>"},
  "extra_info":  {
      "task": "<task>",
      "difficulty": <int>,
      "entry": "<json-encoded full reasoning_gym entry dict>",
  },
}
```

`entry` is JSON-encoded because nested dicts can be lossy through pyarrow↔pandas;
the reward function decodes it back before scoring. Train and test use disjoint
seeds (`seed` and `seed + 81`), so test prompts never overlap with train.

---

## Reward functions

`src/reward_manager.py` exposes two verl `custom_reward_function.name` entry
points:

- `compute_score_continuous` — `score_answer(...)` directly (partial credit).
- `compute_score_binary` — `1.0` iff continuous score is exactly `1.0` else `0.0`.

Both return the **same** dict of extras so W&B can compare continuous vs binary
metrics regardless of training mode:

```python
{"score": <mode-dependent>,
 "continuous": <float in [0,1]>,
 "binary": <0 or 1>,
 "fraction_perfect": <0 or 1>,
 "format_ok": <0 or 1, did response contain <answer> tag>}
```

### K&K fixed reward (spec §3.2)

`reasoning_gym.KnightsKnavesDataset.score_answer` awards `0.3 + 0.7 * match/total`
for partial credit. We implement the spec's simpler `match / total` with a
strict length check: `len(pred) != len(oracle)` ⇒ 0.0. This blocks the
enumerate-all-assignments hack observed in prior GRPO runs. See
`_knk_fixed_score` in `reward_manager.py` and the tests in
`test_knk_reward.py`.

---

## Pass@k baseline

Run vLLM on the base model to measure binary sparsity *before* training. Use
this to validate the spec claim that pass@1 drops with difficulty.

**Single (task, difficulty):**

```bash
python -m experiment10_reasoning_gym.src.compute_pass_at_k \
    --task word_sorting --difficulty 10 \
    --num_prompts 200 --num_samples 64 --k_values 1,4,16,64 \
    --temperature 1.0
```

**Multi-config sweep (one vLLM engine, amortizes the ~40 s model load):**

```bash
python -m experiment10_reasoning_gym.src.pass_at_k_sweep \
    --configs 'word_sorting:4,word_sorting:8,word_sorting:12,knights_knaves:3' \
    --num_prompts 100 --num_samples 32 \
    --max_response_tokens 1024
```

Both write `experiment10_reasoning_gym/results/pass_at_k/{task}_diff{N}.json`
containing `pass@k` values, `mean_continuous_reward`, and `per_prompt_correct`.

---

## Training

### Single run

```bash
bash experiment10_reasoning_gym/scripts/launch.sh \
    <task> <difficulty> <estimator> <reward_mode> [seed] [hydra_overrides...]
```

Positional args:

- `<task>` — `word_sorting | knights_knaves | sentence_reordering | game_of_life`
- `<difficulty>` — integer; the matching parquet must already exist
- `<estimator>` — `gos | maxrl | rloo | grpo` (mapped below)
- `<reward_mode>` — `continuous | binary`
- `[seed]` — defaults to 42

Anything after the seed that contains `=` is forwarded verbatim as a Hydra
override, e.g.:

```bash
bash launch.sh word_sorting 8 gos continuous 42 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.total_epochs=10
```

### Estimator name mapping

Spec labels differ from verl's registered names. `launch.sh` accepts both:

| short    | verl `adv_estimator` | description                          |
|----------|----------------------|--------------------------------------|
| `gos`    | `maxrl_unnorm`       | GOS / ordinal cross-entropy (ours)   |
| `maxrl`  | `binary_maxrl`       | Tajwar et al. binary MaxRL           |
| `rloo`   | `rloo`               | leave-one-out baseline               |
| `grpo`   | `grpo`               | Z-score normalization baseline       |

W&B experiment names retain the short spec labels for readability.

### Per-task lengths (spec §6.2)

Set automatically by `launch.sh` based on `<task>`:

| task                | max_prompt | max_response |
|---------------------|-----------:|-------------:|
| word_sorting        | 512        | 512          |
| sentence_reordering | 512        | 512          |
| game_of_life        | 1024       | 1024         |
| knights_knaves      | 1024       | 2048         |

### Tunable env vars

`launch.sh` reads these from the environment (with defaults):

| var                 | default                                | purpose                                            |
|---------------------|----------------------------------------|----------------------------------------------------|
| `REPO_ROOT`         | `/root/shrinivas/cont-maxrl`           | repo path (override on a different host)           |
| `N_GPUS`            | `1`                                    | `trainer.n_gpus_per_node`                          |
| `TOTAL_EPOCHS`      | `30`                                   | `trainer.total_epochs`                             |
| `SAVE_FREQ`         | `780`                                  | steps between checkpoints                          |
| `MAX_CKPT_KEEP`     | `2`                                    | rolling checkpoint cap                             |
| `TEST_FREQ`         | `25`                                   | steps between val passes                           |
| `VAL_DIFFS`         | training difficulty                    | comma-separated diffs for cross-diff val (below)   |
| `PROJECT_NAME`      | `reasoning_gym_<task>`                 | W&B project                                        |
| `EXPERIMENT_NAME`   | `<task>_<est>_<reward>_diff<N>_s<seed>`| W&B run name + checkpoint dir                      |
| `WANDB_MODE`        | `online`                               | `offline` to skip cloud logging                    |
| `WANDB_ENTITY`      | `stablegradients`                      | W&B team/user                                      |
| `VERL_REWARD_WORKERS` | `64`                                 | reward fn parallelism                              |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN`                      | rollout backend                                    |
| `HF_HOME`           | `${REPO_ROOT}/.cache/huggingface`      | HF model cache                                     |
| `PYTHON`            | `python`                               | interpreter                                        |
| `TRAIN_ENTRY`       | `experiment4_llm/src/train.py`         | verl trainer entry point                           |

### Hydra overrides

Anything containing `=` after the positional args is appended to the python
command. The defaults set by `launch.sh` track spec §6.1: batch 256, G=16
rollouts, lr=1e-6, AdamW, clip=0.2, KL disabled, temperature 1.0, on-policy.
Common overrides:

```bash
# Lower vLLM memory share (run multiple processes per GPU)
actor_rollout_ref.rollout.gpu_memory_utilization=0.5

# Different rollout count G
actor_rollout_ref.rollout.n=8

# Force ref policy weights off-GPU
actor_rollout_ref.ref.fsdp_config.param_offload=True

# Cap total optimizer steps instead of epochs
trainer.total_training_steps=200
```

### Cross-difficulty validation

Set `VAL_DIFFS` to a comma-separated list of difficulty integers; each is
resolved to `data/{task}_diff{N}/test.parquet`. All listed parquets must
exist. verl reports per-(task, difficulty) val metrics because each parquet's
`data_source` is `reasoning_gym_{task}_diff{N}`.

```bash
# Train on diff=8, validate on {4, 8, 12}
VAL_DIFFS=4,8,12 bash launch.sh word_sorting 8 gos continuous
```

### Outputs and W&B

- **Checkpoints:** `${REPO_ROOT}/checkpoints/<PROJECT_NAME>/<EXPERIMENT_NAME>/`,
  rolling `MAX_CKPT_KEEP` (default 2) saved every `SAVE_FREQ` steps.
- **W&B:** project = `<PROJECT_NAME>`, run = `<EXPERIMENT_NAME>`, group =
  `<PROJECT_NAME>`. Metrics to watch:
  - `val/test_score/reward` — main spec metric
  - `val/test_score/continuous`, `.../binary`, `.../format_ok` — comparison
    metrics that are populated regardless of training reward mode
  - `perf/time_per_step`, `perf/mfu/actor` — used by benchmark scripts

---

## Multi-run dispatchers

### `launch_all.sh` — Sweeps A & B

Sweep A (estimator compare): GOS+continuous vs MaxRL+binary on
`word_sorting × {4,6,8,10,12}` and `knights_knaves × {3,5,7}` = 16 runs.
Sweep B (reward ablation): GOS+continuous vs GOS+binary on the same grid; the
GOS+continuous half is shared with Sweep A so only +8 GOS+binary runs.
**Total: 24 distinct runs.**

```bash
# Direct bash on the same node
SUBMIT=bash bash experiment10_reasoning_gym/scripts/launch_all.sh

# SLURM submission (one job per run, uses launch_slurm.sh)
SUBMIT=sbatch bash experiment10_reasoning_gym/scripts/launch_all.sh

# Print commands without executing
DRY_RUN=1 bash experiment10_reasoning_gym/scripts/launch_all.sh
```

### `dispatch_wssr.sh` — WS+SR waves

24 runs (3 word_sorting diffs × 4 combos + 3 sentence_reordering diffs × 4
combos), launched in **3 waves of 8 concurrent runs**, one run per GPU on an
8-GPU box. Each run trains 20 epochs and validates on all 3 difficulties of its
task. Logs land in `logs/prod/{tag}.log`.

```bash
bash experiment10_reasoning_gym/scripts/dispatch_wssr.sh
```

Edit the `WAVE1`/`WAVE2`/`WAVE3` arrays in the script if you need to retarget
runs. Each spec is `"task diff estimator reward val_diffs"`.

### `launch_slurm.sh` — SLURM wrapper

Wraps `launch.sh` with an `#SBATCH` header (`ghx4` partition, 8 GPUs, 24 h,
288 cpus, account `bgmo-dtai-gh`). It also activates conda, exports
`PYTHONPATH`/`VLLM_ATTENTION_BACKEND`, and isolates Ray tmpdirs per job:

```bash
sbatch experiment10_reasoning_gym/scripts/launch_slurm.sh \
    word_sorting 8 gos continuous 42
```

Override `CONDA_ENV` (default `rgym`) if your env name differs.

---

## Benchmark scripts

Three throughput benchmarks. All run 10 steps of WS@4 GOS+continuous in
offline-W&B mode and dump per-run `time_per_step` / `mfu/actor` to stdout when
done.

| script                  | layout                              | use to measure                        |
|-------------------------|-------------------------------------|---------------------------------------|
| `benchmark_1gpu8x.sh`   | 8 runs, 1 GPU each                  | clean per-GPU baseline                |
| `benchmark_2per_gpu.sh` | 16 runs, 2 per GPU (NVIDIA MPS on)  | MPS sharing gain                      |
| `benchmark_parallel.sh` | 4 runs, 2 GPUs each                 | scale-up vs `n_gpus_per_node=2`       |

Logs land in `logs/bench_*/`. Each script ends with a per-run summary
(`time_per_step` and `mfu/actor` extracted from the latest training step).

---

## Plots

Both plots pull metrics from W&B by run name. Tune the `--metric` flag if your
chosen key differs from `val/test_score/reward`.

```bash
# Sweep A: difficulty vs pass@1, GOS+continuous vs MaxRL+binary
python -m experiment10_reasoning_gym.src.plot_difficulty_scaling \
    --task word_sorting \
    --output figures/sweep_a_word_sorting.png

# Sweep B: reward-type ablation (GOS+continuous vs GOS+binary)
python -m experiment10_reasoning_gym.src.plot_reward_ablation \
    --task word_sorting \
    --output figures/sweep_b_word_sorting.png
```

Both expect run names in the form
`<estimator>_<reward>_diff<N>_G16_batch256_s<seed>`; if your custom
`EXPERIMENT_NAME` differs, edit the format strings inside the plot scripts.

---

## Tests

```bash
# Convenient wrapper (sources conda env automatically if /data/miniconda exists)
bash experiment10_reasoning_gym/scripts/run_tests.sh

# Or directly
PYTHONPATH=/path/to/cont-maxrl:$PYTHONPATH \
    python -m pytest experiment10_reasoning_gym/tests/ -v
```

66 tests cover:

- `<answer>` tag extraction (12) — case-insensitivity, truncation, multiple
  blocks, multi-line, `None`/empty, CoT-before-answer.
- K&K fixed reward (13) — oracle match, anti-hack (too many / too few), exact
  fractional partial credit, order invariance, punctuation tolerance,
  set-semantics on duplicates, binary mode equivalence.
- Cross-task reward behaviour (15) — perfect-answer scoring, partial credit on
  word_sorting, invalid JSON on game_of_life, length-mismatch penalty on
  sentence_reordering, binary-mode indicator.
- `generate_data` (21) — difficulty kwargs mapping, parquet schema
  (`data_source`, `prompt`, `reward_model`, `extra_info`), answer instruction
  appended, oracle round-trips through generate→score for all four tasks,
  reproducibility under a fixed seed, train/test disjointness, pandas parquet
  round-trip.
- verl integration (3) — `RLHFDataset` loads our parquet with the
  Qwen3-0.6B-Base tokenizer, `extra_info` survives and feeds `compute_score`,
  binary vs continuous indicator consistency.

---

## Adding a new task or difficulty

1. **Confirm the task exists in reasoning_gym** and find its difficulty knobs:

   ```python
   import reasoning_gym; ds = reasoning_gym.create_dataset("<task>", size=1, seed=0)
   print(ds[0])
   ```

2. **Map difficulty to kwargs** in `src/generate_data.py::_difficulty_kwargs`.
3. **Add per-task lengths** in `scripts/launch.sh` (`MAX_PROMPT_LENGTH` /
   `MAX_RESPONSE_LENGTH` case statement).
4. **Verify the reward path** — for tasks beyond the four built-in,
   `_score_continuous` falls back to reasoning_gym's `score_answer`. Add a
   custom branch (like `_knk_fixed_score`) if you need to override partial
   credit or block hacks.
5. **Generate data** then **add a smoke test** in `tests/test_generate_data.py`
   covering the oracle round-trip for the new task.
6. **Run a single launch** with `WANDB_MODE=offline` and a short
   `trainer.total_training_steps=10` to confirm everything wires up.

For a new difficulty on an existing task, just `generate_data` with the new
difficulty integer — no code changes needed.

---

## Troubleshooting

- **`ERROR: missing train parquet`** — run `generate_data.py` for that
  `(task, difficulty)`. Same for missing val parquets when `VAL_DIFFS` is set.
- **`extra_info missing 'task' or 'entry'`** — your parquet was generated by
  an older script; regenerate with the current `generate_data.py`.
- **`WARNING: WANDB_API_KEY not set`** — set `WANDB_MODE=offline` to skip, or
  drop `WANDB_API_KEY` into `.env` at repo root.
- **CUDA OOM on a single GPU** — lower
  `actor_rollout_ref.rollout.gpu_memory_utilization` (try `0.5`), enable
  `actor_rollout_ref.ref.fsdp_config.param_offload=True`, or reduce
  `actor_rollout_ref.rollout.n` (G).
- **Multiple Ray clusters on one node** — give each its own
  `RAY_TMPDIR=/tmp/ray_<unique>` and `TMPDIR=/tmp/tmp_<unique>`. The benchmark
  scripts and `dispatch_wssr.sh` show the pattern.
- **Two procs per GPU without speedup** — without NVIDIA MPS, CUDA
  time-slices and you lose all the gain. `benchmark_2per_gpu.sh` starts
  `nvidia-cuda-mps-control -d` first; copy that pattern.
- **W&B run name collision across tasks sharing a project** — `launch.sh`
  bakes `<task>` into the default `EXPERIMENT_NAME` to avoid this; set
  `PROJECT_NAME` per-sweep if you want runs grouped together.
- **verl points at the wrong checkout** — check
  `python -c "import verl; print(verl.__file__)"`. If it resolves to a stale
  copy (e.g. `/data/repos/maxrl_idk/verl/`) instead of `vendor/verl`, do an
  editable reinstall from `vendor/verl` inside your env.

---

## Deviations from spec and open questions

1. **Reward *manager* vs reward *function***. Spec §5 asks us to register a
   `ReasoningGymRewardManager` class in verl's reward_manager registry. verl
   upstream does not expose a public registry (see
   `vendor/verl/verl/trainer/ppo/reward.py::load_reward_manager`); the
   hard-coded ladder only accepts `naive|prime|batch|dapo`. Following the
   pattern used by `experiment6_maze` and `experiment4_llm`, we instead use
   the `custom_reward_function.path/name` mechanism with the stock `naive`
   reward manager. Functionally identical; avoids patching vendor/verl. The
   `compute_score_continuous` / `compute_score_binary` functions in
   `src/reward_manager.py` are the two callable entry points.

2. **Estimator name mapping**. Spec labels the estimators
   `gos/p_normalization/grpo/rloo`. The registered verl values (see
   `vendor/verl/verl/trainer/ppo/ray_trainer.py::AdvantageEstimator`) are
   `maxrl_unnorm/binary_maxrl/grpo/rloo`. `launch.sh` accepts both the short
   names (`gos`, `maxrl`) and their verl equivalents. W&B experiment names
   retain the short spec labels for readability.

3. **word_sorting extra-word hack (noted; not fixed per spec §3.1)**.
   `reasoning_gym.WordSortingDataset.score_answer` gives
   `correct_positions / len(oracle)`, so the model can append garbage after
   the correct sorted list and still score 1.0. Spec §3.1 explicitly says
   "No format reward component. The reward is purely the score_answer
   output", so we do **not** add a length-match anti-hack here. If this
   becomes a measurable issue during training, revisit.

4. **K&K parsing**. We reuse reasoning_gym's `_normalize_answer` (which
   lowercases, strips punctuation, and extracts `(name, role)` pairs) for
   consistency with the official scorer. This matches Fahim's "set
   intersection of (character, assignment) pairs" description but has not
   been cross-validated against his exact `fraction_reward_naive`
   implementation — spec §3.2 asks for that coordination.

5. **Sentence reordering & game-of-life difficulty knobs**. Verified present
   in reasoning_gym ≥0.1.19: `min/max_words_in_sentence` for SR,
   `grid_size_x/grid_size_y` for GoL. Sweep A in the spec only targets
   word_sorting and knights_knaves, so launching these is optional.
