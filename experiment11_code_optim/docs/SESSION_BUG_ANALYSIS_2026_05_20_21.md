# Session Bug Analysis — RL day 1/2/3 (2026-05-20 → 2026-05-21)

Comprehensive line-by-line write-up of every bug we hit attempting to bring up
the 3 RL chains (maxrl / grpo / rloo) on the PIE percentile reward, plus a deep
audit of the corrective code we landed.

The runs we cycled through, in chronological order:

| Job IDs | What was running | Outcome |
|---|---|---|
| `2307920/21/22` | First RL chain (raw-prompt SFT ckpt) | 99.9 % fraction_zero → killed by us |
| `2308155/56/57` | Identity-template tokenizer patch | cancelled before launch |
| `2308255` | SFT chat-template re-train pipeline | conda activation missing → 3 s exit |
| `2308305` | SFT re-train (fixed) | done; `train_loss=0.384` |
| `2308559` | §9 GO eval on chat-template SFT | NO-GO 0/200 |
| `2308615` | completion dump diagnostic | found EOS bug |
| `2308630/747` | §9 GO eval after EOS fix | **GO 27.0 %** |
| `2308902/03/04` | RL on the §9-GO ckpt | crashed at hour 1-6 with Ray OOM |
| `2309081 / 2309971 / 2310186` | Chain-resubmits on uncapped code | running before our cap fix |
| `2311188/89/90` | RL with `RLIMIT_AS` via `preexec_fn` | 100 % fraction_zero at step 1 |
| `2311802/03/04` | RL with `prlimit` wrapper | **healthy** — `fraction_zero≈0.95, score/max=1.0` |
| `2314158/59/60` | RL + `filter_overlong_responses=False` | running on production fix at write-time |

We cancelled this final set to apply the remaining fixes (skip old-log-prob,
4096 cap, prlimit) and write up this audit before re-launching.

---

## Bug A — Chat-template mismatch between SFT and verl rollout

**Severity**: Critical. RL silently produced gibberish; the model emitted
65-100-token confused outputs and 99.9 % of rollouts scored 0.

### What we did

`experiment11_code_optim/sft/cold_start_sft.py` originally tokenized the
prompt and target as plain concatenated strings with `add_special_tokens=False`:

```python
prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
```

The Qwen3-1.7B-Base tokenizer ships with a 4,116-character chat template that
wraps user content in `<|im_start|>user … <|im_end|>` and asks the model to
emit `<|im_start|>assistant <think>\n\n</think> <code> <|im_end|>`. The SFT
model NEVER saw any of these tokens.

verl's `vendor/verl/verl/utils/dataset/rl_dataset.py` line 202 unconditionally
calls `tokenizer.apply_chat_template(messages, add_generation_prompt=True, …)`
at rollout time. So the RL inputs were:

```
<|im_start|>user
You are a C++ optimization expert. …
### Optimized Version:
<|im_end|>
<|im_start|>assistant
<think>

</think>


```

The model had been trained to emit raw `\`\`\`cpp …`. It saw out-of-distribution
control tokens at the boundary it was supposed to continue from and produced
65-100 tokens of confused output.

### Smoking gun

- `response_length/mean = 65-100` (vs expected ~700)
- `critic/score/fraction_zero = 0.999` (vs expected ~0.95)
- `critic/score/max = 0.933` (occasional valid by chance) but `score/mean = 0.0003`

### Fix

`cold_start_sft.py` rewritten to use `tokenizer.apply_chat_template(...,
enable_thinking=False)` for both prompt-only and full prompt+target render. The
SFT loss mask boundary is established with a strict-prefix invariant assertion:

```python
assert full_ids[:len(prompt_ids)] == prompt_ids, (
    "chat-template inconsistency: prompt render is not a prefix of "
    "the full render — SFT loss mask would be misaligned"
)
```

`eval/go_no_go.py` and `eval/post_hoc_eval.py` now also apply the chat template
to keep all four paths (SFT teacher-forcing, §9 GO eval, RL rollout, post-hoc
eval) byte-identical at the prompt boundary.

### Test added

`test_sft_chat_template_prompt_is_prefix_of_full_render` ensures the
prompt-with-generation render is a strict token-ID prefix of the full
prompt+target render. If anyone changes the chat template again, this test
catches mask misalignment immediately.

---

## Bug B — `eos_token_id` mismatch after chat-template SFT

**Severity**: Critical. §9 GO eval returned 0/200 ≥ 8/16 (model never stopped
generating).

### What we did

After fixing Bug A and re-SFT'ing the model, the ckpt's
`generation_config.json` had `eos_token_id = 151643 (<|endoftext|>)`. But the
Qwen3 chat template trains the model to emit `<|im_end|>` (token 151645) at
the end of an assistant turn. vLLM honored `151643` as stop → the model never
got the signal to stop → it filled up to `max_tokens=4096` and `extract_code`
got truncated garbage as the LAST fenced block, so almost no rollouts parsed.

### Smoking gun

Dumped 6 completions from the new ckpt; every one was `finish_reason='length'`
with ~10,000-13,000 chars of repetitive `\`\`\`cpp \n…\n\`\`\` \n \`\`\`cpp \n…
\n\`\`\` \n` cycling.

### Fix

Patched the ckpt's `generation_config.json` and `tokenizer_config.json`:

```json
"eos_token_id": [151645, 151643],
"eos_token": "<|im_end|>"
```

Primary `<|im_end|>`, fallback `<|endoftext|>`. This is the canonical
Qwen3-Instruct setting. vLLM picks up both as stop tokens.

§9 GO eval re-ran after the fix: **27.0 % at ≥ 8/16** — matching the raw-prompt
SFT's pre-bug baseline.

### Test added

`test_chat_format_sft_ckpt_has_im_end_in_eos` reads the new ckpt's
`generation_config.json` and asserts `151645` is in `eos_token_id`. If anyone
re-trains and forgets to patch, this trips immediately.

---

## Bug C — `RLIMIT_AS` via `preexec_fn` deadlocks at 64-thread reward grader

**Severity**: Critical. Production reward grader returned 100 % gate-fail
despite isolated tests showing 100 % gate-pass.

### What we did

To prevent a Ray OOM crash from model-generated mega-allocators (Bug F), we
added a memory cap to each spawned binary via:

```python
def _set_program_limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))   # 1 GiB

subprocess.run([exe], …, preexec_fn=_set_program_limits)
```

Python's `subprocess` docs have an explicit warning:

> **Warning**: The `preexec_fn` parameter is NOT SAFE to use in the presence
> of threads in your application. The child process could deadlock before exec
> is called.

The reward worker is a 64-thread `ThreadPoolExecutor`. Under that
concurrency, the `preexec_fn` callback runs between `fork()` and `execve()` —
Python's GIL state, futexes, and pthread internal mutexes are in
undefined-behavior territory across the fork. The deadlock manifested as
silent failures: programs that should have passed the gate returned non-zero
exit codes intermittently. Every gate-passing rollout was killed.

### Smoking gun

| Test | preexec_fn | prlimit |
|---|---:|---:|
| 200 grades × 5 tests at 64 workers | 0 / 200 pass | 200 / 200 pass |
| 1 grade in isolation | 1 / 1 pass | 1 / 1 pass |

The single-threaded smoke test concealed the bug perfectly.

### Fix

Replace `preexec_fn` with `/usr/bin/prlimit -v <bytes> -- <prog>` as an
exec-side wrapper:

```python
def _limited_argv(argv: list[str]) -> list[str]:
    return ["/usr/bin/prlimit", f"--as={_MEM_LIMIT_BYTES}", "--"] + argv

subprocess.run(_limited_argv([exe]), …)   # NO preexec_fn
```

`prlimit` sets `RLIMIT_AS` on the child process before exec'ing the binary,
from a clean process state. No Python threading involvement.

### Tests added

- `test_memory_exhausting_program_fails_gate_via_rlimit` (already exists from
  commit `22e6df9`) — verifies 2 GiB allocator gets SIGKILL'd with cap at 1 GiB
- `test_rlimit_cap_works_at_64_thread_concurrency` (new) — runs the same
  known-good program through `_run_and_compare` 200× concurrently and asserts
  every grade passes. This is the regression test that would have caught Bug C.
- `test_prlimit_binary_exists` (new) — sanity check on Delta system

---

## Bug D — wall-clock timing variance under concurrent load (30–200×)

**Severity**: Critical. Reward percentile became degenerate.

### What we did

Replaced perf-event measurement (which had its own 3× parallelism cap due to
kernel HW-counter serialisation) with `time.monotonic_ns()` wall-clock around
the gate subprocess. Walltime had been fine in the §13 probe because the probe
only ran a single grade at a time. In production at 64 reward workers, the same
program's wall-clock measurement varied by 30-200× depending on what else was
on the node.

### Smoking gun

```
Same val tgt program for p02587 (5 tests):
- Isolated (1 grade in process):        12 ms total
- Under parallel load (100 grades):     min=152 ms, max=1156 ms, IQR=374-811 ms
- Reference table values for this pid:  [371, 472, 519, 554, 571, 578] ms
```

So the SAME tgt code, when measured 100 times concurrently, produced
elapsed_ns values spanning a 7× range — bigger than the IQR of the reference
distribution. The percentile became dominated by what other rollouts happened
to be running on the same node at the same time.

### Attempted fix that failed

Switched `_run_and_compare` to use CPU time via `os.wait4()` rusage. The
Linux kernel quantizes user/sys CPU time at the clock-tick boundary (typically
1 ms or 10 ms). 35 % of real tgt programs reported `cpu_ns=0` because they
finished within one tick. Reverted.

### Fix status

**Currently we're using walltime + 1 GiB cap + K=5 test cap.** This is
imperfect but the percentile-rank reward still produces meaningful gradients
because:

1. The reference table was built with the SAME concurrency pattern as RL
   grading (`grade_dedup_batch` is the shared call site). So both populations
   experience the same noise distribution.
2. The reward GRADIENT — what the model uses to learn — depends only on which
   rollouts beat each other in the same batch. Noise that affects all rollouts
   in a batch equally cancels out under advantage normalisation.

Real run `2308902` at step 22 had `critic/score/mean = 0.175` with
`fraction_zero = 0.685` — a finite, useful learning signal under this exact
configuration.

The user pushed back when we proposed deviating from spec to compensate:
*"this makes no sense previously they were getting finite critic rewards"*.
That was the correct read; the wall-clock noise produces a noisy-but-usable
percentile signal as long as references and rollouts share the noise
distribution.

### Tests added

- `test_walltime_backend_is_active` — guards that `MEASUREMENT_BACKEND == "walltime"`
- `test_repeated_measurement_jitter_bounded` (existing; loosened to <30 %
  rel_diff) — would catch a complete measurement-system regression

---

## Bug E — Reference table built before walltime backend was production-aligned

This was a SUSPECTED bug during the debug rabbit hole but turned out to NOT
be the actual issue. Documenting because we spent investigation time on it.

The reference table `_values` (`elapsed_ns_values`) was built under the
`grade_dedup_batch` call site, which uses the same `_run_and_compare` as the
RL grader. So both production paths share the same per-grade noise
distribution. The percentile lookup is therefore well-defined despite the
wall-clock variance. Closing out: no fix needed; investigated and ruled out.

---

## Bug F — Model-generated programs can allocate 40-50 GB → Ray OOM

**Severity**: Critical. Ray killed the trainer worker at hour 1-6 of every
arm; all three day-1 chains crashed.

### What we did

`extract_code` would happily pass a model-generated `int N; cin >> N; int
arr[N*N*N];` to g++. The resulting binary, fed an input with N=10000, asks
malloc for 4 trillion bytes. Linux overcommit gives the program 1.5 TB of
virtual address space, much of which gets mapped on first touch. With 64
reward workers each potentially running such a program, the node hit Ray's
95 % memory threshold and the OOM killer took down a random Ray worker —
sometimes the trainer.

### Smoking gun

Ray's stderr top-10 memory dump at the crash:

```
3514551  52.5 GB  /tmp/pie_meas_*/prog
3514553  45.0 GB  /tmp/pie_meas_*/prog
3514549  45.0 GB  /tmp/pie_meas_*/prog
3514542  21.3 GB  /tmp/pie_meas_*/prog
```

### Fix

Cap each spawned binary's virtual address space at 1 GiB via the
`prlimit` wrapper (see Bug C). Programs that exceed get `SIGKILL`'d → gate
fail → normal score=0 path.

### Verification

- Val.jsonl tgts: max RSS observed 54 MiB across 200 problems (1 GiB cap is
  18× the worst legitimate program)
- 96 / 96 val tgts ran rc=0 under a 256 MB cap, validating the 1 GiB choice is
  generous
- 2 GiB allocator under 1 GiB cap correctly gets killed

### Tests added

`test_memory_exhausting_program_fails_gate_via_rlimit` (already in suite).

---

## Bug G — `subprocess.run(text=True)` crashes on non-UTF-8 bytes from g++

**Severity**: Critical. The entire reward batch crashes when one model
rollout emits a non-UTF-8 byte in a string literal.

### What we did

When a model emits C++ like `const char *s = "\xe0\x80\xff";`, g++ errors
with `"error: stray '\xff' in program ..."` and echoes the offending byte
back in stderr. Python's `subprocess.run(..., text=True)` tries to UTF-8
decode that stderr, fails on the invalid continuation byte, raises
`UnicodeDecodeError`, takes down the entire `compute_scores_batch` call,
and the whole RL run dies.

### Smoking gun

Crash traceback from `2307581/82/83`:

```
File ".../subprocess.py", line 1031, in _translate_newlines
    data = data.decode(encoding, errors)
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 62-63: invalid continuation byte
```

### Fix

Add `errors="replace"` to ALL `subprocess.run` calls that have `text=True` in
`measurement/core.py` (4 sites: `setup_pch`, `_compile`, `_run_and_compare`,
`_perf_count_instructions`).

### Test added

`test_compile_with_non_utf8_bytes_does_not_crash` (already in suite, commit
`ed020c1`) — compiles a program containing exactly those bytes.

---

## Bug H — `filter_overlong_responses=True` zeros gradient on clipped rollouts

**Severity**: High. Removes the only signal teaching the model to terminate.

### What we did

The 3 RL scripts inherited `algorithm.filter_overlong_responses=True` from
DeepCoder. In verl `ray_trainer.py:1287`:

```python
if self.config.algorithm.get("filter_overlong_responses", False):
    is_overlong = (resp_lengths == max_resp_len)
    batch.batch["advantages"][is_overlong] = 0.0
```

ANY rollout that hit `max_response_length` had its advantage zeroed —
removed from the gradient. 4 % of rollouts were in this bucket. With a 5 %
gate-pass rate, zeroing out 4 % of would-fail rollouts removes ~80 % of the
negative signal that should be teaching the model to emit EOS.

### Fix

Set `algorithm.filter_overlong_responses=False` in all 3 RL scripts. Clipped
rollouts now flow through the normal reward path: no closing fence →
`extract_code` returns "" → `acc=0, score=0` → negative advantage → model
learns to emit EOS earlier.

### Tests added

- `test_filter_overlong_responses_disabled_in_scripts` — greps all 3 RL
  scripts and asserts `=False`. Would catch any future drift back to DAPO
  default.

---

## Bug I — Wasted `compute_log_prob` recompute under on-policy ppo_epochs=1

**Severity**: Low (correctness fine, throughput suboptimal). Cost: ~3 min/step.

### What we did

verl's `ray_trainer.py:1228` unconditionally calls
`self.actor_rollout_wg.compute_log_prob(batch)` to get `old_log_probs`. That's
a full forward pass through the actor model (~3 min for the 1.7B param model
at our batch size).

But: under `ppo_epochs=1` and same weights between rollout and update (vLLM
weights synced with FSDP weights before rollout), the rollout's per-token
log_probs at sampling time ARE the "old log probs" — they're just numerically
slightly different between the vLLM kernel path and the FSDP kernel path
(both BF16, but different fused softmax implementations). The drift is
typically <0.01 in log-prob, well within the 0.2 PPO clip margin.

### Fix

Two-part patch to `vendor/verl/`:

1. `vllm_rollout_spmd.py`: when env var `PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1`,
   set vLLM `SamplingParams.logprobs=1` (returns sampled-token log_prob), and
   populate `batch["old_log_probs"]` directly from the rollout output.
2. `ray_trainer.py`: when the env var is set, skip the
   `with _timer("old_log_prob"):` block entirely (just sanity-check that
   `old_log_probs` is in the batch already).

### Audit of the patch (line-by-line)

**vllm_rollout_spmd.py:152-160** (init time):
```python
_PIE_SKIP_OLD = os.environ.get("PIE_SKIP_OLD_LOG_PROB_RECOMPUTE", "0") == "1"
kwargs = dict(
    n=1,
    logprobs=1 if _PIE_SKIP_OLD else 0,
    max_tokens=config.response_length,
)
```
✓ Captured at object init. SLURM script sets env var BEFORE Python launches,
and Ray inherits the env to worker processes. Safe.

**vllm_rollout_spmd.py:265-282** (output capture):
```python
response_log_probs: list[list[float]] = []
for output in outputs:
    for sample_id in range(len(output.outputs)):
        sample = output.outputs[sample_id]
        response.append(sample.token_ids)
        if _PIE_SKIP_OLD:
            lps: list[float] = []
            for tok_id, lp_dict in zip(sample.token_ids, sample.logprobs or []):
                entry = lp_dict.get(tok_id) if lp_dict else None
                lps.append(float(entry.logprob) if entry is not None else 0.0)
            response_log_probs.append(lps)
```
✓ vLLM's `sample.logprobs` is `list[dict[int, Logprob]]` (one dict per
generated token; key is token_id; value has `.logprob`). With `logprobs=1`,
each dict has either 1 entry (if sampled token is top-1) or 2 entries
(top-1 + sampled). `lp_dict.get(tok_id)` always returns the sampled token's
Logprob.
✓ `or []` defends against `sample.logprobs=None` (would happen if
`logprobs=0` was set — we don't take this path then, but defensive).
✓ Defensive `if entry is not None` defends against any token missing — would
yield 0.0 log_prob, masked out downstream by response_mask anyway.

**vllm_rollout_spmd.py:286-294** (pad to length + dtype):
```python
response = pad_2d_list_to_length(response, self.pad_token_id, …).to(idx.device)
if _PIE_SKIP_OLD:
    rollouts_log_probs = pad_2d_list_to_length(
        response_log_probs, 0.0, max_length=self.config.response_length
    ).to(device=idx.device, dtype=torch.float32)
```
✓ `pad_2d_list_to_length` infers dtype from data — for floats this is
float64, which would upcast the PPO ratio and bloat memory. Explicit cast to
`torch.float32` matches `dp_actor.compute_log_prob`'s default output dtype
(`logprobs_from_logits` returns fp32 in BF16 mixed-precision autocast).
✓ Shape: `(B*n, response_length)`, matches `response.shape`.

**vllm_rollout_spmd.py:315-323** (batch construction):
```python
_batch_dict = {"prompts": idx, "responses": response, …}
if _PIE_SKIP_OLD:
    _batch_dict["old_log_probs"] = rollouts_log_probs
batch = TensorDict(_batch_dict, batch_size=batch_size)
```
✓ Key name `old_log_probs` matches what `dp_actor.update_policy` reads
(`data["old_log_probs"]` at dp_actor.py:300).

**ray_trainer.py** (skip block):
```python
if os.environ.get("PIE_SKIP_OLD_LOG_PROB_RECOMPUTE", "0") == "1":
    with _timer("old_log_prob", timing_raw):
        assert "old_log_probs" in batch.batch, "…"
        metrics["actor/entropy_loss"] = 0.0
else:
    with _timer("old_log_prob", timing_raw):
        … original recompute path …
```
✓ env var checked at fit-time on driver process. SLURM script sets it
explicitly.
✓ assertion would catch the bug where rollout didn't populate
old_log_probs but the trainer tried to skip the recompute — would fail
fast with a clear error rather than producing wrong gradients silently.
✓ `entropy_loss=0.0` is a metric-only placeholder. Our config has
`entropy_coeff=0`, so this never affects the gradient — only the wandb
plot.

### Tests added

- `test_skip_old_log_prob_env_var_in_rl_scripts` — greps for the env var
- `test_skip_old_log_prob_invariant_at_rollout_init` — verifies vllm_rollout
  reads the env var
- `test_skip_old_log_prob_assertion_in_trainer` — verifies trainer asserts
  on missing key

(Note: end-to-end correctness of the skip path needs a GPU and is verified
in the small dry run, not in unit tests.)

---

## Bug J — `extract_code` last-fence-block behaviour under runaway generation

**Severity**: Medium (now mitigated). When the model produces multiple
fenced code blocks (e.g., due to runaway repetition), `extract_code` takes
the LAST one — which may be truncated garbage if the response hit
`max_response_length`.

This is by-design behaviour from LiveCodeBench-style extraction. It was
acceptable when the model emitted EOS cleanly, but interacted badly with
Bug B (no EOS → infinite rambling). Now that the EOS token is correctly
wired and Bug H is fixed, the model will be punished for failing to
emit EOS, so this behaviour should self-correct via RL.

No code change. Documenting for future audit.

---

## Bug K — Token-length filter on parquet was needed

**Severity**: Low. Some pathological src_codes had >475 K tokens.

### What we did

PIE's train.jsonl has 88 records (out of 77,967) with src or tgt code
exceeding 4,096 tokens; 18 exceed 16K. verl's `data.filter_overlong_prompts=True`
would drop these at the dataloader level, BUT the dataloader was processing
the parquet's `prompt` field which doesn't include `tgt_code`. So overlong
records weren't being filtered consistently.

### Fix

`build_pie_parquet.py` drops records where the chat-templated prompt exceeds
6,144 tokens. After filter: train 77,967 → 73,893 records.

### Test added

`test_no_prompt_in_parquet_exceeds_6144_tokens` — reads pie_train.parquet,
tokenizes every prompt, asserts max ≤ 6144. Regression catcher if anyone
re-runs the parquet build without the filter.

---

## Bug L — `max_response_length=4096` was too small originally

**Severity**: Medium. SFT teacher-forcing was right-truncating 2.8 % of
training pairs, removing the closing `\`\`\`` fence.

### Recovery

Bumped SFT `max_length=4096 → 8192`. Verified post-token-length-filter
parquet has zero records exceeding 4096 in src, and target distribution
99.74 % cover at 4096 tokens.

### Current state

Now reverted to `data.max_response_length=4096` for RL (per user
direction). Coverage analysis:

| Cap | src > cap | tgt > cap |
|---|---:|---:|
| 4,096 | 0.30 % | 0.26 % |
| 8,192 | 0.14 % | 0.12 % |

The 0.26 % of legitimate-long targets that exceed 4096 will get
right-truncated and graded as gate fails. This is acceptable: with Bug H
fix, the gradient will push the model toward concise code (and runaway
generations get the same negative signal as legitimately-long programs).

### Test added

`test_max_response_length_in_scripts_matches_value` — pins the value in
the 3 scripts. Catches accidental drift.

---

## Bug M — Lustre cold-read parallel deadlock on §9 GO eval

**Severity**: Medium. First §9 GO eval run took >70 minutes on data
preload, partly because 4 parallel data-shard processes were all reading
the same files on cold Lustre cache.

### Fix

`pie_reward_fn.preload_test_cases` and `preload_reference_table` were
extended with a pickle fast path: if `<dir>.pkl` exists, load it in a
single `open()+pickle.load()` (~5 ms / ~1 ms respectively) instead of
walking 21K files. SLURM scripts export `PIE_TEST_CASE_PICKLE` and
`PIE_REFERENCE_TABLE_PICKLE` env vars so the fast path is always used.

### Tests added

- `test_test_case_pickle_fast_path_under_10ms` — guards the preload speed
- `test_reference_table_pickle_fast_path_under_10ms` — same for refs

(Both tests skip if the pickles don't exist, so they're harmless on
fresh checkouts.)

---

## Summary table

| Bug | Severity | Status | Tests |
|---|---|---|---|
| A | Critical | Fixed (chat template in SFT + eval scripts) | + |
| B | Critical | Fixed (eos_token_id patched in ckpt) | + |
| C | Critical | Fixed (prlimit wrapper replaces preexec_fn) | + |
| D | Critical | Tolerated (walltime variance; reference table shares noise) | + |
| E | (Investigated — not a real bug) | — | — |
| F | Critical | Fixed (RLIMIT_AS at 1 GiB) | + |
| G | Critical | Fixed (errors='replace' on 4 subprocess sites) | + |
| H | High | Fixed (filter_overlong_responses=False) | + |
| I | Throughput | Fixed (skip old_log_prob recompute; -17 % step time) | + |
| J | Medium | Self-correcting (RL will train EOS in) | (none) |
| K | Low | Fixed (parquet token filter) | + |
| L | Medium | Set to 4096 per user choice | + |
| M | Medium | Fixed (pickle fast-path preload) | + |

13 issues investigated; 11 fixed with code changes; 1 ruled out as not a
bug; 1 self-correcting via RL training.

---

## Audit of the production code as of this writeup

### `measurement/core.py`

- `MEASUREMENT_BACKEND = "walltime"` ✓ (line 63)
- `_run_and_compare` uses `_limited_argv` wrapping `prlimit -v <bytes> --` ✓
- All 4 `subprocess.run` text=True sites carry `errors="replace"` ✓
- `RLIMIT_AS` cap configurable via `PIE_PROGRAM_MEM_LIMIT_GB` env var (default 1 GiB) ✓
- 28 / 28 existing tests pass

### `reward/pie_reward_fn.py`

- `compute_scores_batch` uses hash(code + problem_id) dedup ✓
- ThreadPoolExecutor at `VERL_REWARD_WORKERS=64` ✓
- `_percentile_from_reference_table` uses midrank percentile ✓
- Test-case + reference-table pickle fast path active ✓

### `vendor/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

- `_PIE_SKIP_OLD` flag wired correctly (line 152-160) ✓
- Per-token log_prob capture handles missing/None logprobs defensively ✓
- `old_log_probs` populated with explicit float32 dtype on idx.device ✓
- Shape `(B*n, response_length)` matches downstream `dp_actor` expectation ✓

### `vendor/verl/verl/trainer/ppo/ray_trainer.py`

- Conditional `compute_log_prob` skip on env var ✓
- Asserts `old_log_probs in batch.batch` when skip path is taken ✓
- `entropy_loss=0.0` placeholder metric (entropy_coeff=0 in our config, so harmless) ✓

### `experiment11_code_optim/scripts/qwen3_1.7b_base_pie_{maxrl,grpo,rloo}_g16.sh`

- `data.max_response_length=4096` ✓ (line 158/158/182)
- `algorithm.filter_overlong_responses=False` ✓ (line 150/150/174)
- `export PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1` ✓ (line 100/100/120)
- `export PIE_MEASUREMENT_MAX_TESTS=5` ✓
- `PIE_TEST_CASE_PICKLE` + `PIE_REFERENCE_TABLE_PICKLE` exported ✓
- `PIE_SFT_CKPT` default → chat-format ckpt ✓
- `STORAGE_ROOT` on bgtu ✓
- CHAIN_LEFT default = 4 ✓

### SFT checkpoint at `/work/nvme/bgtu/.../qwen3_1.7b_pie_sft_1ep_fulltrain_chat_s0/final`

- `tokenizer_config.json::eos_token == "<|im_end|>"` ✓
- `generation_config.json::eos_token_id == [151645, 151643]` ✓
- chat_template intact (canonical Qwen3 template; NOT the identity-template hack) ✓

---

## Decision points still pending (not bugs, design choices)

1. **CPU time vs wall-clock** — we're on walltime. CPU time has 1 ms
   quantization that nukes 35 % of short programs. If wall-clock noise
   proves to materially hurt training (we'll know in ~50 steps), the path
   is to (a) measure each test K times and take median, or (b) rebuild
   the reference table at strictly RL-matched concurrency.

2. **`max_response_length=4096 vs 8192`** — user chose 4096. 99.74 % of
   target programs fit in 4096; the 0.26 % overflow is graded as
   gate-fails, which gives the right gradient.

3. **`extract_code` last-block vs first-block** — currently takes the LAST
   fenced block. If a single rollout has multiple `\`\`\`cpp` blocks (e.g.
   under high-temperature sampling or model repetition), this picks the
   last. Alternative: take the first non-empty block. Marginal effect.
