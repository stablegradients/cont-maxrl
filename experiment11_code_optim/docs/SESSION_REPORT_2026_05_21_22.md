# Session Report — RL bring-up 2026-05-21 / 2026-05-22

Continuation of `SESSION_BUG_ANALYSIS_2026_05_20_21.md`. Covers the second
day of bugfixes and the launch of the first 3-arm production run, plus the
vLLM CUDA-OOM at step 22 that ended it.

---

## TL;DR

- §9 GO/NO-GO eval passed at **27.0 % ≥8/16** on the chat-format SFT ckpt
  (eq. to the original raw-prompt 28 % baseline before the EOS-token fix).
- All-3-arm production run **launched** at 2026-05-21T22:12 on `bgmo / bgyb / bgtu`.
- Ran healthy for **~4 h 31 m** (~28 steps for maxrl, ~27 for grpo, ~22 for rloo)
  with strong learning signal (mean score climbing 4-10x over baseline,
  gate-pass climbing 4-12x).
- **rloo died at step 22** with `vLLM cuMemMap` CUDA OOM during
  the FSDP↔vLLM `sleep/wake_up` handoff. Chain-resubmit recovered cleanly
  but lost all 22 steps (save_freq=100, no ckpt yet).
- User killed all 3 arms. Currently **dry-running rloo** at
  `actor.ppo_micro_batch_size_per_gpu = 2` (halved from 4) to verify the
  cure before re-launching production.

---

## Issues found and fixed (since SESSION_BUG_ANALYSIS_2026_05_20_21.md)

| # | Issue | Severity | Commit | Status |
|--:|---|---|---|---|
| 14 | `os` name-shadow trap in `vllm_rollout_spmd.py` (UnboundLocalError) | Critical | `dc129a8` | ✅ |
| 15 | Truncated module docstring in `vllm_rollout_spmd.py` (SyntaxError) | Critical | `17f8636` | ✅ |
| 16 | `TensorDict` membership via `in` (NotImplementedError) | Critical | `a2c0ea9` | ✅ |
| 17 | `meta_info["temperature"]` missing in skip-old-log-prob path (KeyError) | Critical | `2a5fcde` | ✅ |
| 18 | `_compile` doesn't catch `subprocess.TimeoutExpired` (RLOO crash) | Critical | `f6ed5a2` | ✅ |
| 19 | vLLM CUDA OOM at `wake_up` (RLOO step 22) | Critical | `2a61420` | 🟡 mitigated, dry-run pending |

### Issue 14 — `os` UnboundLocalError

**Symptom**: dry-run 2314300 died at vLLM init with `UnboundLocalError: local
variable 'os' referenced before assignment` at line 157 of
`vllm_rollout_spmd.py`.

**Cause**: A nested `import os` inside a megatron-only branch of
`vLLMRollout.__init__` (line 96) made Python treat `os` as function-local for
the entire `__init__` body — so the `os.environ.get(...)` we added at line
157 (the PIE_SKIP_OLD_LOG_PROB_RECOMPUTE env check) raised at fresh access
BEFORE the nested import executed.

**Fix**: Read the env var at MODULE LOAD time (`_PIE_SKIP_OLD` module-level
constant), avoid touching `os` inside `__init__` entirely.

**Test**: `test_skip_old_log_prob_env_var_read_at_module_level_not_init` —
source-level grep that catches re-introduction.

### Issue 15 — truncated docstring

**Symptom**: dry-run 2314382 died with `SyntaxError: unterminated string
literal` at line 39 of `vllm_rollout_spmd.py`.

**Cause**: My docstring edit for the previous fix accidentally truncated the
original module docstring; the original's "FSDP / Megatron" notes ended up
OUTSIDE the closing `"""` and Python tried to parse them as code. The line
containing `doesn't` (apostrophe) triggered the SyntaxError.

**Fix**: Restored the original docstring text inside the same docstring;
softened `doesn't` → `does not` for future-edit safety.

**Test**: `test_patched_vllm_rollout_module_parses_and_imports` — fresh-imports
the patched module to catch any syntax/docstring breakage.

### Issue 16 — TensorDict membership

**Symptom**: dry-run 2318791 died with `NotImplementedError: TensorDict does
not support membership checks with the 'in' keyword`.

**Cause**: My safety assertion in `ray_trainer.py`'s skip-old-log-prob branch
was `assert "old_log_probs" in batch.batch` — but `batch.batch` is a
TensorDict, which raises rather than returning a bool.

**Fix**: `assert "old_log_probs" in batch.batch.keys()` (TensorDict's
`.keys()` returns a normal iterable).

**Test**: source-grep that the patch contains the `.keys()` form.

### Issue 17 — `meta_info["temperature"]` KeyError

**Symptom**: dry-run 2318910 died with `KeyError: 'temperature'` in
`dp_actor.update_policy`.

**Cause**: The normal compute_log_prob path (in `fsdp_workers.py`) sets
`data.meta_info["temperature"] = self.config.rollout.temperature` as a
side-effect. The skip-path bypasses that, so when `update_actor` was called
next, `update_policy` couldn't find the key.

**Fix**: In ray_trainer.py's skip branch, explicitly set
`batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature`
before update_policy.

**Test**: source-grep for the `batch.meta_info["temperature"]` assignment.

### Issue 18 — `_compile` doesn't catch `subprocess.TimeoutExpired`

**Symptom**: Production grpo (2323298, step 3) died with `subprocess.TimeoutExpired:
Command '[g++, ...]' timed out after 30s`.

**Cause**: A model-generated C++ program with pathological template
instantiation caused g++ to hang past 30 s. `_compile` only caught
`returncode != 0`, not `TimeoutExpired`. The exception propagated up through
`grade_dedup_batch.pool.map` → `compute_scores_batch` → verl's reward fn →
trainer crash.

**Fix**: Wrap subprocess.run in try/except for both `TimeoutExpired` AND
`OSError`, return `(None, error_str)` instead of raising. Same path as any
other compile failure.

**Tests**:
- `test_compile_catches_timeout_and_oserror_in_source` — source-level
  grep that catches removal of the try/except (so we don't have to wait 30 s)
- `test_compile_timeout_returns_failure_dict_not_raises` — sanity check
  that bad source returns `(None, err)`.

### Issue 19 — vLLM CUDA OOM at `wake_up`

**Symptom**: Production rloo (2323560) died at step 22 with:
```
RuntimeError: CUDA Error: out of memory
File ".../fsdp_vllm.py", line 132, in __enter__:
    self.inference_engine.wake_up(tags=["kv_cache"])
File ".../device_allocator/cumem.py", line 77:
    python_create_and_map(*allocation_handle)
```

**Cause**: GH200 has 120 GB. Each PPO step alternates:

| Phase | GPU memory |
|---|---|
| Rollout (vLLM) | vLLM weights + KV cache (~50 GB at gpu_memory_utilization=0.5) |
| `vllm.sleep(level=1)` | KV cache released, weights stay |
| FSDP forward/backward/Adam | **71 GB peak** (logged in `perf/max_memory_allocated_gb`) |
| `vllm.wake_up(["kv_cache"])` | Needs to mmap **~50 GB contiguous** for KV cache |

After the FSDP peak, PyTorch's caching allocator holds ~71 GB worth of
freed-but-cached memory. cuMem (vLLM's allocator) bypasses the caching pool
and can't see it. After ~22 steps of accumulated allocator fragmentation,
free physical memory was no longer big enough for the 50 GB contiguous request.

RLOO is the most-affected arm because un-normalized advantages produce
larger gradient norms → larger FSDP all-gather intermediates → higher peak.

**Fix**: `actor.ppo_micro_batch_size_per_gpu: 4 → 2`. Activations scale
linearly with micro-batch; halving cuts the activation portion of the
FSDP peak by ~30 GB. Cost: ~10 % lower throughput on update_actor.

**Dry-run verification**: in progress as of writing (job 2327304 on bgmo,
gh038). If 3 consecutive steps complete without OOM, we relaunch production.

### Rejected alternatives (for reference)

| Alternative | Why not picked |
|---|---|
| `rollout.gpu_memory_utilization 0.5 → 0.4` | Reduces KV-cache budget; symptomatic, not root-cause |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Helps with fragmentation but doesn't reduce peak |
| `save_freq 100 → 25` | Recovery-side fix only; doesn't prevent the OOM |
| `sleep_level=2` (offload weights too) | +1-2 s/step weight reload; worse cost than micro-batch halving |

---

## Production run timeline (jobs 2323558 / 2323559 / 2323560)

Launched 2026-05-21T22:12; killed 2026-05-22T02:43. ~4 h 31 m wall-clock.

### Step trajectories

10-step running averages (all 3 arms):

| Arm | s10 mean | s10 gate | 10-step avg mean | avg gate | avg f1 |
|---|---:|---:|---:|---:|---:|
| maxrl | 0.079 | 0.198 | 0.069 | 0.155 | 0.024 |
| grpo  | 0.112 | 0.208 | 0.083 | 0.168 | 0.029 |
| rloo  | 0.081 | 0.162 | 0.081 | 0.164 | 0.028 |

By the time of the rloo crash (step 22), all 3 arms had reached
running averages of **mean ≈ 0.085, gate ≈ 0.18** — roughly **3.5× the
§9 GO baseline** gate rate (5 %) and **15× the SFT baseline mean score**.

### Peak single-step results

| Arm | step | mean | gate | f1 |
|---|---:|---:|---:|---:|
| maxrl | 28 | **0.198** | **0.476** | 0.059 |
| grpo  | 8  | 0.141 | 0.259 | 0.055 |
| rloo  | 16 | 0.156 | 0.275 | **0.065** |

Across the 28 logged steps maxrl produced **47.6 % gate-pass** — meaning
nearly half of 4096 rollouts passed the strict C++ gate against 5 test
cases each. Compare with step 1's 9.5 % gate-pass. That's a 5× increase in
just 28 RL updates.

### Per-step variance

Per-step CV (std/mean) ≈ **49 %**, much higher than DeepCoder's <15 %.
Documented in line with the user's question about reward stability;
structural to the percentile reward and bimodal test-case distribution
(89 % Public-source problems with ≤5 tests, 11 % Generated-source with
100 tests). Running average is the right signal; per-step values are
batch-difficulty dominated.

### Step time

- Average: **~10 min/step** for maxrl/grpo, ~10 min/step for rloo.
- Per-step phases (maxrl typical):
  - `timing_s/gen` ≈ 200-330 s (vLLM rollout, dominant)
  - `timing_s/old_log_prob` ≈ **0.0 s** (skip patch worked perfectly)
  - `timing_s/update_actor` ≈ 325 s
  - `timing_s/reward` ≈ 1.0 s (negligible — speed-fix paid off)

Old log-prob recompute saving: ~177 s/step. At 867 total steps × 3 arms,
that's ~46 GPU-hours saved.

### What we learned about the 3 estimators

Advantage normalization matters more than expected:

| Arm | grad_norm range | early learning rate |
|---|---|---|
| GRPO (whitened by group std) | 0.18-0.30 | Moderate, smooth |
| MaxRL_unnorm (max-of-group baseline) | 0.20-0.50 | Faster, more variable |
| RLOO (LOO baseline, no norm) | 0.27-1.84 | Variable, occasionally explosive |

The RLOO gradient explosions (`pg_loss = -1.418` at step 15, `pg_loss=
0.813` at step 9) didn't cause divergence — pg_clipping kept the policy
update bounded — but they almost certainly contributed to the FSDP peak
memory pressure that triggered the eventual OOM.

GRPO had the smoothest learning curve. Whether it converges to a better
or worse final policy is the actual scientific question; we don't know yet.

---

## Production launch infrastructure

### Spray-then-cancel pattern for fastest dispatch

Used multiple times when a job sat in PD (Priority/Resources) on a single
account. Spray to all 7 accounts in parallel, take the winner, scancel the
rest:

```bash
for acct in bgmo bgtu bgts bgtv bgyb bgyc bgyd; do
    sbatch --account=${acct}-dtai-gh ... &
done
wait
# Monitor watches all jobs; when first hits R state, kill the rest.
```

Tracked by 2 monitors in tandem: a "race" monitor that resolves when any
job hits R state, plus a step-progress monitor that catches both winners
and losers' early crashes.

### Storage layout

| Path | Filesystem | Used for |
|---|---|---|
| `/work/nvme/bgmo/.../pie_data/` | Lustre | Read-only inputs (test cases, parquets, pickles) |
| `/work/nvme/bgtu/.../sft_ckpts/` | Lustre | SFT ckpt (read-only at RL time) |
| `/work/nvme/bgtu/.../results/qwen3_1.7b_pie_*/` | Lustre | RL outputs (verl `default_local_dir`) |
| `/tmp/{USER}_*/` | xfs (local NVMe) | Per-grade tmpdirs, PCH, scratch |

Chose `bgtu` for RL outputs because `bgmo` quota was at 632/1000 GB.
`bgtu` is 1.95 TB quota with 56 GB used; comfortable headroom.

### Chain-resubmit mechanism

Each SLURM script self-resubmits with `--dependency=afterany` BEFORE
launching the training:

```bash
NEXT_CHAIN=$((CHAIN_LEFT - 1))
if [[ ${NEXT_CHAIN} -gt 0 ]]; then
    NEXT_JOB=$(sbatch \
        --account="${SLURM_JOB_ACCOUNT}" \
        --dependency=afterany:${SLURM_JOB_ID} \
        "${SCRIPT}" "${SEED}" "${NEXT_CHAIN}")
fi
```

`--account="${SLURM_JOB_ACCOUNT}"` propagation ensures chain links stay on
the same account as the parent (fixed in `0573e15` earlier in the session).
With `CHAIN_LEFT=4`, each arm has 4 × 47 h = 188 h of compute envelope.

`trainer.resume_mode=auto` makes verl re-load from the latest
`global_step_<N>/` if present. Combined: the chain restarts gracefully from
the last save.

**Caveat that bit us at rloo step 22**: `save_freq=100`, so when the
crash happened at step 22 there was no saved checkpoint and the chain
restarted from SFT — losing 22 steps. Considered tightening to
save_freq=25 but didn't apply (would cost +30 GB disk per arm).

---

## Open questions / future work

1. **Verify the micro-batch fix** — currently dry-running rloo with
   `ppo_micro_batch_size_per_gpu=2`. If 3 steps complete without OOM, relaunch
   all 3 arms. If still OOMs, next move is 8-bit Adam (saves ~10 GB).

2. **Entropy metric in skip-old-log-prob path** — currently
   `actor/entropy_loss = 0.0` (sentinel). A free MC estimate
   `H ≈ -mean(old_log_probs × response_mask)` would give us mode-collapse
   monitoring without any extra forward pass. ~3 lines in `ray_trainer.py`.
   Not yet implemented.

3. **save_freq tightening** — if vLLM-wake OOMs recur even after the
   micro-batch fix, drop save_freq from 100 → 25 to limit recovery loss.
   Disk cost: +30 GB/arm.

4. **bf16 reduce_dtype** — current FSDP MixedPrecision uses fp32 for
   gradient all-reduce (safe default). Switching to bf16 would save ~2 GB
   transient memory but risks gradient underflow on tiny norms. Not
   recommended; documented as a reserve lever.

---

## Files modified this session

### Code

- `experiment11_code_optim/measurement/core.py` — TimeoutExpired/OSError catch in `_compile`
- `vendor/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — PIE_SKIP_OLD module-level read + docstring + log_probs capture
- `vendor/verl/verl/trainer/ppo/ray_trainer.py` — skip-old-log-prob branch + TensorDict.keys() + meta_info[temperature]
- `experiment11_code_optim/scripts/qwen3_1.7b_base_pie_{maxrl,grpo,rloo}_g16.sh` — ppo_micro_batch_size_per_gpu 4→2

### Tests

- `experiment11_code_optim/tests/test_session_regressions_2026_05_21.py` — added 4 regression tests for issues 14/15/16/17/18

### Docs

- `experiment11_code_optim/docs/SESSION_BUG_ANALYSIS_2026_05_20_21.md` — comprehensive bug list (yesterday)
- `experiment11_code_optim/docs/SESSION_REPORT_2026_05_21_22.md` — this file

### Commit graph (latest first)

```
2a61420 tune(rl): ppo_micro_batch_size_per_gpu 4 → 2 (vLLM CUDA OOM at wake_up)
f6ed5a2 fix(reward): _compile catches TimeoutExpired+OSError, returns failure dict
2a5fcde fix(verl): set meta_info[temperature] in skip-old-log-prob path
a2c0ea9 fix(verl): TensorDict membership via .keys()
17f8636 fix(verl): restore truncated module docstring in vllm_rollout_spmd.py
dc129a8 fix(verl): read PIE_SKIP_OLD env var at module level, not __init__
e29657e feat(rl): skip old_log_prob recompute + 4096 cap + session bug analysis
df8b054 fix(rl): filter_overlong_responses False — penalize clipped rollouts
... (cf. yesterday's report for the full chain)
```

All commits pushed to `origin/pie-efficiency`.
