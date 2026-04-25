# Experiment 10 · Reasoning Gym Sweep — Protocol

**Generated:** 2026-04-24
**Dispatch script:** `scripts/dispatch_sweep_deltaai.sh`
**Per-run SLURM template:** `scripts/launch_slurm_deltaai.sh`
**Single-run entry point:** `scripts/launch.sh` (unchanged upstream)
**Watchdog:** `scripts/watchdog_kill_dead_binary.py`

---

## 1. Scientific question

Does the **performance gap between continuous and binary reward widen** as task difficulty grows and binary signal becomes sparser? Prior MaxRL work claims partial-credit reward gives richer signal when correct-response density is low. We test this across three distinct task families.

---

## 2. Experiment matrix — 36 runs

| factor | values | count |
|---|---|--:|
| task | `word_sorting`, `sentence_reordering`, `game_of_life` | 3 |
| difficulty | `WS: {4,8,12}` · `SR: {3,7,10}` · `GoL: {4,7,10}` | 3 each |
| advantage estimator | `gos` (MaxRL-unnorm) vs `grpo` (Z-score baseline) | 2 |
| reward mode | `continuous` vs `binary` | 2 |
| seed | `42` | 1 |

**3 × 3 × 2 × 2 × 1 = 36 runs.**

### 2.1 Task semantics and difficulty parameter

| task | difficulty controls | partial-credit rule (continuous) | binary rule |
|---|---|---|---|
| `word_sorting` | `min_words = max_words = diff` | `correct_positions / len(oracle)` | 1 iff all positions correct |
| `sentence_reordering` | `min_words_in_sentence = max_words_in_sentence = diff` | tree-edit-distance (via `zss`); length mismatch → 0 | 1 iff sequences identical |
| `game_of_life` | `grid_size_x = grid_size_y = diff` (→ `diff²` cells) | fraction of cells correct in predicted next-state grid | 1 iff all cells correct |

Knights-Knaves from the original spec is excluded from this sweep.

---

## 3. Training configuration

| knob | value | note |
|---|---|---|
| model | `Qwen/Qwen3-0.6B-Base` | spec §4 |
| precision | bfloat16 rollouts + actor | |
| `data.train_batch_size` | 256 | spec §6.1 |
| `rollout.n` (rollouts per prompt) | **G = 16** | spec §6.1 |
| `ppo_mini_batch_size` | 256 | on-policy (1 PPO epoch) |
| `ppo_micro_batch_size_per_gpu` | 16 | actor / ref / log_prob |
| optimizer | AdamW, `lr=1e-6`, `grad_clip=1.0` | spec §6.1 |
| clip ratios | `low = high = 0.2` | |
| KL | **disabled** (`use_kl_loss=False`, `use_kl_in_reward=False`, `kl_coef=0`) | spec §6.1 |
| epochs | **10** (780 steps @ 78 steps/epoch) | |
| rollout backend | vLLM 0.8.3 · `enforce_eager=True` · `free_cache_engine=True` · `gpu_memory_utilization=0.85` | Qwen3 patch requires eager |
| torch.compile | **disabled** on actor + ref | aarch64 triton `ImportError: triton_key` |
| `remove_padding` | True | |
| gradient checkpointing | True | |
| ref policy | `param_offload=True` | to fit on 1 GPU |

### 3.1 Per-task sequence lengths

| task | `max_prompt` | `max_response` |
|---|---:|---:|
| `word_sorting` | 512 | 512 |
| `sentence_reordering` | 512 | 512 |
| `game_of_life` | 1024 | 1024 |

### 3.2 Data

- **Train set:** 20,000 prompts per (task, difficulty). Seed 42.
- **Test set:** 1,000 prompts per (task, difficulty). Seed `42 + 81` (disjoint from train).
- Parquets: `experiment10_reasoning_gym/data/{task}_diff{N}/{train,test}.parquet`
- `data_source` per row is `reasoning_gym_{task}_diff{N}` → verl reports per-difficulty val metrics.
- Oracle answer in `reward_model.ground_truth`; full entry in `extra_info.entry` (JSON-encoded).

### 3.3 Advantage estimator mapping

| short name | verl `adv_estimator` | description |
|---|---|---|
| `gos` | `maxrl_unnorm` | MaxRL-unnormalized (ordinal cross-entropy) |
| `grpo` | `grpo` | GRPO Z-score normalization (baseline) |

### 3.4 Reward functions (`src/reward_manager.py`)

- `compute_score_continuous` — `reasoning_gym.score_answer(...)` directly ∈ [0, 1]
- `compute_score_binary` — `1.0` iff continuous score is exactly `1.0`, else `0.0`

Both emit the same extras dict so W&B captures the "other" signal even when training on the opposite mode:

```
{ score, continuous, binary, fraction_perfect, format_ok }
```

→ For any `binary`-trained run we can still plot the `continuous` reward trajectory post-hoc.

---

## 4. Evaluation protocol

### 4.1 Validation set

Each run is trained on **one** difficulty but validated on **all three difficulties of its task** (cross-difficulty generalization):

| train task | val difficulties |
|---|---|
| `word_sorting` | `4, 8, 12` |
| `sentence_reordering` | `3, 7, 10` |
| `game_of_life` | `4, 7, 10` |

verl logs per-difficulty metrics under `val/test_score/reasoning_gym_{task}_diff{N}/*`.

### 4.2 Validation cadence and sampling

- `val_before_train=False` — no pre-training baseline eval (saves ~5 min; step-1 rollout acts as de-facto baseline).
- `test_freq=78` — eval at the end of each epoch (steps 78, 156, 234, …, 780).
- Validation rollouts: **4 samples per prompt** (`val_kwargs.n=4`), `temperature=1.0`, `top_p=1.0`, `top_k=-1`, `do_sample=True` → 4,000 completions per eval.

### 4.3 Metrics logged per eval

| key | meaning |
|---|---|
| `val/test_score/{data_source}/reward/mean` | headline reward mean (matches training mode) |
| `val/test_score/{data_source}/continuous/mean` | mean partial-credit score (always) |
| `val/test_score/{data_source}/binary/mean` | mean exact-match rate |
| `val/test_score/{data_source}/fraction_perfect/mean` | fraction of completions scoring 1.0 |
| `val/test_score/{data_source}/format_ok/mean` | fraction with a well-formed `<answer>…</answer>` tag |
| `val/test_score/{data_source}/{std, min, max}` | distributional |

**Training-time reward** (logged every step):

| key | meaning |
|---|---|
| `critic/score/mean` | mean reward over this step's 4,096 rollouts |
| `critic/score/fraction_{zero,partial,one}` | reward distribution shape (key for collapse detection) |
| `actor/{pg_loss, entropy_loss, grad_norm, pg_clipfrac}` | optimizer health |

### 4.4 W&B layout

- **Project:** `stablegradients/Reasoning-GYM-GOS` — single project for the whole sweep
- **Run name:** `{task}_{est}_{reward}_diff{N}_s{seed}`
  e.g. `word_sorting_gos_binary_diff8_s42`
- **Group:** `{task}_{est}_{reward}_diff{N}` — no seed, so future multi-seed runs collapse into the same group

### 4.5 Checkpointing and resume

| setting | value |
|---|---|
| `SAVE_FREQ` | 78 steps (= 1 epoch) |
| `MAX_CKPT_KEEP` | 3 rolling |
| checkpoint root | `/work/nvme/bgmo/sramasubramanian/cont-maxrl/checkpoints/Reasoning-GYM-GOS/{run_name}/` |
| `trainer.resume_mode` | `auto` (child jobs resume from latest rolling ckpt) |

---

## 5. Infrastructure

| aspect | value |
|---|---|
| platform | NCSA DeltaAI (aarch64, 4× GH200 per `ghx4` node) |
| GPU per run | 1 × GH200 120 GB |
| CPUs per run | 72 (of 288 per node) |
| SLURM wall per job | 47 h |
| chain depth | 1 pre-submitted dependent (`--dependency=afterany:<parent>`) = 94 h headroom |
| accounts used | `bgmo, bgtv, bgtu, bgts, bgyb, bgyc, bgyd` (round-robin, 8-GPU cap each → 56 concurrent slots) |
| measured step time | ~270 s/step at batch=256, G=16 |
| per-run wall (10 epochs) | ~58 h |
| sweep wall (queue tail + training) | ~3 days best case, ~3.5 days realistic |

### 5.1 Critical environment set in `launch_slurm_deltaai.sh`

```bash
# Conda CUDA (avoid system CUDA 12.9 / conda CUDA 12.4 mismatch)
export CUDA_HOME=$CONDA_PREFIX
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"

# NCCL — disable system plugin (incompatible with conda NCCL on this cluster)
unset NCCL_NET NCCL_NET_PLUGIN
export NCCL_PLUGIN_P2P=0

# Verl / Ray / vLLM
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VERL_REWARD_WORKERS=64        # prevents fork-overcommit SIGSEGV on aarch64
export RAY_OBJECT_STORE_MEMORY=32000000000
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID   # short; Linux AF_UNIX socket limit is 108 bytes
```

### 5.2 Hydra overrides passed to `launch.sh`

```
trainer.total_epochs=10
trainer.resume_mode=auto
trainer.val_before_train=False
ray_init.num_cpus=64                       # avoid Ray hang under SLURM cgroup
reward_model.launch_reward_fn_async=True   # async reward dispatch
data.enable_thinking=False                 # base model, no <think>
actor_rollout_ref.model.trust_remote_code=True
actor_rollout_ref.actor.use_torch_compile=False
actor_rollout_ref.ref.use_torch_compile=False
actor_rollout_ref.rollout.enforce_eager=True
actor_rollout_ref.rollout.free_cache_engine=True
actor_rollout_ref.rollout.dtype=bfloat16
actor_rollout_ref.rollout.max_num_batched_tokens=32768
actor_rollout_ref.rollout.max_num_seqs=1024
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.gpu_memory_utilization=0.85
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

---

## 6. Early-kill watchdog (binary-only)

**Script:** `experiment10_reasoning_gym/scripts/watchdog_kill_dead_binary.py`

### Policy

For each run in `stablegradients/Reasoning-GYM-GOS`:

1. Must be `state == "running"`.
2. Run name must contain `_binary_`.
3. Pull `run.history(keys=["critic/score/mean", "_step"], samples=500)`.
4. `baseline` = earliest-step `critic/score/mean`.
5. `latest` = latest-step `critic/score/mean`.
6. If `latest._step ≥ 100` and `(latest_score − baseline) < 0.04` → **`scancel --name=rgym_<run_name>`** (kills parent + its pre-submitted dependent chain via job-name match).

### Rationale

Absolute thresholds (e.g. "score ≤ 1%") fail when baseline is already > 1%. Relative 4 pp improvement over ~100 steps (~7½ h of training) is the threshold for "has the model started learning at all" on binary sparse reward. Continuous-reward runs always continue regardless.

### Defaults / CLI

```
--min-improvement 0.04   # 4 percentage-point gain required
--min-step 100           # only kill after step >= 100
--interval 900           # 15-minute poll
--dry-run                # log decisions, don't scancel
--once                   # single pass then exit
```

---

## 7. Planned analyses

All plots can be regenerated from W&B alone — no new scoring pass needed.

### 7.1 Sweep A · "reward-type gap vs difficulty" (headline)

- **x:** difficulty per task (low / mid / high)
- **y:** final `val/test_score/.../binary/mean` at matched difficulty (train-diff = val-diff)
- **curves:** `gos+continuous` vs `gos+binary` per task
- **hypothesis:** gap widens as difficulty rises (MaxRL continuous-reward advantage under sparsity)

### 7.2 Sweep B · estimator comparison

- Same axes; curves `gos+continuous` vs `grpo+continuous` and `gos+binary` vs `grpo+binary`
- Tests whether the gain is **method-specific** (GOS) vs **reward-signal-specific** (continuous-vs-binary)

### 7.3 Cross-difficulty generalization heatmaps (secondary)

- Per (task, est, reward): rows = val difficulty, cols = train difficulty, cell = `binary` score
- Reveals whether training at one difficulty transfers to harder/easier problems within the same task

---

## 8. Known deviations from the original spec

1. **Reward manager.** verl upstream does not expose a public reward-manager registry; we use verl's stock `naive` manager with `custom_reward_function.{path,name}` instead of a custom `ReasoningGymRewardManager`. Functionally identical.
2. **GoL substituted for K&K.** Original Sweep A targets `word_sorting + knights_knaves`; we swapped in `game_of_life` per user choice.
3. **Single seed.** Only seed `42`. Multi-seed can be added later sharing the same W&B group.
4. **word_sorting length-mismatch hack (unchanged).** `reasoning_gym.WordSortingDataset.score_answer` gives `correct_positions / len(oracle)`, so the model can append garbage after the sorted list and still score 1.0. Spec §3.1 explicitly mandates no format reward, so we keep this behaviour.

---

## 9. Quick-reference commands

```bash
# dispatch the sweep (idempotent; submits 36 × 2-chain jobs)
bash experiment10_reasoning_gym/scripts/dispatch_sweep_deltaai.sh

# dry-run the dispatcher
DRY_RUN=1 bash experiment10_reasoning_gym/scripts/dispatch_sweep_deltaai.sh

# queue status
squeue -u $USER -o '%.10i %.40j %.15a %.2t %.10M %.20R' | grep rgym_

# start-time estimates
squeue -u $USER --start | grep rgym_

# live per-run stdout
tail -f slurm_logs/rgym_<run_name>_<jobid>.out

# arm the watchdog (foreground; wraps with nohup / tmux yourself for persistence)
python experiment10_reasoning_gym/scripts/watchdog_kill_dead_binary.py

# watchdog dry-run
python experiment10_reasoning_gym/scripts/watchdog_kill_dead_binary.py --once --dry-run

# cancel the whole sweep
squeue -u $USER -h -o "%i %j" | grep rgym_ | awk '{print $1}' | xargs -r scancel
```
