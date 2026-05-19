# PARITY_CONFIG.md — PIE-vs-DeepCoder parity record

**Authoritative spec:** SPEC v9 (FINAL): PIE Instruction-Count Percentile Reward — GOS vs GRPO/RLOO on Code Optimization (delivered in the originating conversation; not committed to the repository).

**Build date:** 2026-05-19.

**Sources read at build time (verbatim):**

- `experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_maxrl_g16.sh` — DeepCoder GOS reference
- `experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_grpo_g16.sh` — DeepCoder GRPO reference
- `experiment4_llm/src/train.py` — verl Hydra entrypoint, target of the PIE 2-line wrapper (spec §2; confirmed ~34 lines)
- `experiment4_llm/src/verl_register.py` — `CUSTOM_ESTIMATORS` keys and the 3-arg `verl_advantage_fn` wrapper (spec §2 confirmed: NO `epsilon` arg; docstring at line 39 is stale and was not believed)

## §3.1 Verified two-line difference

The DeepCoder GOS and GRPO scripts differ in **exactly** these non-trivial lines:

| Line | maxrl_g16.sh | grpo_g16.sh |
|------|--------------|-------------|
| 2 (`#SBATCH --job-name`) | `q3_1.7b_maxrl_g16` | `q3_1.7b_grpo_g16` |
| 50 (`WANDB_RUN_GROUP`) | `qwen3-1.7b-base_deepcoder_maxrl_G16` | `qwen3-1.7b-base_deepcoder_grpo_G16` |
| 85 (`adv_estimator`) | `maxrl_unnorm` | `grpo` |
| 141 (`experiment_name`) | `qwen3_1.7b_deepcoder_maxrl_g16_s${SEED}` | `qwen3_1.7b_deepcoder_grpo_g16_s${SEED}` |
| (chain logic) | resubmits maxrl_g16.sh | resubmits grpo_g16.sh |

This **confirms spec §3.1 / §3.3**: the estimator-only design is realisable as exactly one algorithmic config knob plus run-identification strings. PIE inherits this discipline.

## §3.1 Verified parity: hparams transcribed and held identical across PIE/DeepCoder

All of the following values were read from the live DeepCoder GOS script and copied verbatim into all three PIE scripts (`scripts/qwen3_1.7b_base_pie_{maxrl,grpo,rloo}_g16.sh`):

**KL fully off (spec §3.1):**
- `algorithm.use_kl_in_reward=False`
- `algorithm.kl_ctrl.type=fixed`
- `algorithm.kl_ctrl.kl_coef=0.0`
- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.actor.kl_loss_coef=0.0`

**Optimization / clipping:**
- `actor_rollout_ref.actor.clip_ratio=0.2`
- `actor_rollout_ref.actor.clip_ratio_high=0.2`
- `actor_rollout_ref.actor.grad_clip=0.3`
- `actor_rollout_ref.actor.ppo_epochs=1`
- `actor_rollout_ref.actor.optim.lr=1e-6`

**Batching:**
- `data.train_batch_size=256`
- `actor_rollout_ref.actor.ppo_mini_batch_size=256`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`

**Rollout (training):**
- `actor_rollout_ref.rollout.n=16`
- `actor_rollout_ref.rollout.temperature=1.0`
- `actor_rollout_ref.rollout.top_p=1.0`

**Rollout (val):**
- `actor_rollout_ref.rollout.val_kwargs.n=64`
- `actor_rollout_ref.rollout.val_kwargs.temperature=0.6`
- `actor_rollout_ref.rollout.val_kwargs.top_p=0.95`
- `actor_rollout_ref.rollout.val_kwargs.top_k=-1`

**Data layout:**
- `data.max_prompt_length=1024`
- `data.max_response_length=4096` (PIE-permitted task-structural knob per spec §3.1; held at 4096 for now, revisit at §11 against staged C++ length distribution)
- `data.enable_thinking=False`
- `data.reward_fn_key=data_source`
- `data.filter_overlong_prompts=True`
- `algorithm.filter_overlong_responses=True`

**Model/FSDP/vLLM:**
- `actor_rollout_ref.model.enable_gradient_checkpointing=True`
- `actor_rollout_ref.model.trust_remote_code=True`
- `actor_rollout_ref.actor.strategy=fsdp`
- `actor_rollout_ref.actor.fsdp_config.fsdp_size=-1`
- `actor_rollout_ref.actor.use_torch_compile=False`
- `actor_rollout_ref.ref.use_torch_compile=False`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`
- `actor_rollout_ref.rollout.dtype=bfloat16`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1`
- `actor_rollout_ref.rollout.max_num_batched_tokens=32768`
- `actor_rollout_ref.rollout.max_num_seqs=1024`
- `actor_rollout_ref.rollout.enforce_eager=True`
- `actor_rollout_ref.rollout.free_cache_engine=True`
- `actor_rollout_ref.rollout.enable_chunked_prefill=False`

**Reward wiring:**
- `reward_model.enable=False`
- `reward_model.launch_reward_fn_async=True`
- `custom_reward_function.name=compute_score`

**Runtime / env:**
- `module load python/miniforge3_pytorch/2.10.0`
- `conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl`
- `VERL_REWARD_WORKERS=64`, `VLLM_USE_V1=0`, `RAY_OBJECT_STORE_MEMORY=32000000000`
- `unset NCCL_NET`, `unset NCCL_NET_PLUGIN`, `export NCCL_PLUGIN_P2P=0`
- `ray_init.num_cpus=280`
- SBATCH: `--partition=ghx4 --gpus-per-node=4 --time=47:00:00 --mem=0 --cpus-per-task=288 --nodes=1`
- Self-chaining via `sbatch --dependency=afterany`

## §3.1a Epoch arithmetic — COMPUTED, NOT TRANSCRIBED

**10-epoch total = Tajwar et al. (binary MaxRL) protocol convention.** This is a **convention match, not a compute/sample-budget match**. Steps-per-epoch is legitimately PIE-specific because the PIE train set size differs from DeepCoder's. A still-rising best@K/pass@K at epoch 10 is a reportable finding (PIE may need longer than the binary task), not a spec violation or a license to silently extend.

**Formula (spec §3.1a; agent must execute at §14 step 4 after PIE data verification):**

```
steps_per_epoch       = ceil(N_pie_train / train_batch_size)   # train_batch_size = 256 (parity-fixed)
total_training_steps  = 10 * steps_per_epoch                   # 10-epoch cap, replaces DeepCoder's 1000
eval_steps            = [k * steps_per_epoch for k in 1..10]   # replaces DeepCoder's [83,166,...,996]
save_freq             = steps_per_epoch                        # one checkpoint per epoch boundary
```

`N_pie_train` is the row count of the PIE train parquet after the dedup/filtering performed by `data_prep`. **Computation is gated on §14 step 4**: the agent verifies the staged JSONL, runs the data-prep, reads the row count, and fills concrete values back into this file (replacing the placeholders below) plus exporting them into the submission environment for the SLURM scripts.

**Concrete values (locked by user at GATE A, corrected 2026-05-19):**

- **RL training source: `train.jsonl`** (77,967 records, 1,474 unique problem_ids)
- **SFT cold-start source: `train_hq_only.jsonl`** (4,085 records, SAME 1,474 unique problem_ids; per-problem subset of train.jsonl) — matches spec §6 default `sft.mode=hq`

| Field | Value | Source |
|-------|-------|--------|
| `N_pie_train` (raw row count, train.jsonl) | **77,967** | `wc -l /work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl` |
| `N_pie_train` (post-dedup by `src_id`) | TBD at §14 step 4 (likely <77,967 because the same slow program appears with multiple fast targets; RL prompt = slow program → duplicate src_ids = redundant prompts; the cap of ~53 pairs/problem × 1474 problems suggests dedup could reduce N substantially) | data_prep |
| `N_unique_problems` (train) | **1,474** | empirical, problem_id set |
| `steps_per_epoch` | **305** (= `ceil(77967/256)`) | computed from raw |
| `total_training_steps` | **3,050** (= 10 × 305) | spec §3.1a 10-epoch cap |
| `eval_steps` | **[305, 610, 915, 1220, 1525, 1830, 2135, 2440, 2745, 3050]** | spec §3.1a |
| Projected wall-clock (DeepCoder rate ~7 min/step) | **~356 h ≈ 8 self-chained 47h segments** — matches the existing `CHAIN_LEFT=${2:-8}` default in the SLURM scripts; coincidence is fortunate | §13 step 5 probe refines |
| Projected wall-clock (if PIE C++ overhead 2×) | ~712 h ≈ **16 segments** — `CHAIN_LEFT` would need increase | §13 step 5 |
| Projected wall-clock (if PIE C++ overhead 3×) | ~1068 h ≈ **23 segments** | §13 step 5 |
| Comparison: DeepCoder | `total_training_steps=1000`, `total_epochs=15`, `save_freq=83` (= ceil(deepcoder_train_size/256)) — does NOT transfer | verified in scripts |

**Cross-split discipline (confirmed empirically, recorded for §10 isolation guard):**

| Overlap | Count | Status |
|---|---|---|
| `train ∩ val`  (problem_ids) | 0 | ✓ clean split |
| `train ∩ test` (problem_ids) | 0 | ✓ clean split |
| `val   ∩ test` (problem_ids) | 0 | ✓ clean split |
| `train_hq_only ⊆ train` | 1474 (full overlap) | ✓ SFT data is a per-problem subset of RL data, as designed |

**Concern recorded for §13 step 5 probe:** the existing `CHAIN_LEFT=${2:-8}` is correct ONLY if PIE per-step time matches DeepCoder's ~7 min/step. If C++ compile+grade is 2-3× the Python `exec` path, CHAIN_LEFT must be raised proportionally. The probe MUST report a real per-step wall-clock; the agent will recompute CHAIN_LEFT and flag for user override before the first multi-segment launch.

Alternative train sources (recorded for completeness; not chosen):

| Source | N | steps_per_epoch | total (10ep) | Wall-clock @7min/step | Role |
|--------|---|-----------------|--------------|----------------------|------|
| **`train.jsonl`** (chosen for RL) | **77,967** | **305** | **3,050** | **~356 h** | **RL training** |
| `train_hq_only.jsonl` (chosen for SFT) | 4,085 | (n/a — SFT uses HF Trainer not verl) | (1 epoch SFT default) | (small — minutes to hours) | SFT cold-start |
| `train_hq_and_synthetic.jsonl` | 5,570 | — | — | — | not used |
| `train_with_synthetic.jsonl` | (~very large; ~247 MB) | — | — | — | not used |

**Isolation guard (spec §10) anti-mis-transcription assertions** (built at §14 step 7):

```python
assert trainer.total_training_steps == 10 * trainer.save_freq, \
    "spec §3.1a: total_training_steps must equal 10 * save_freq"
assert trainer.save_freq == math.ceil(N_pie_train / 256), \
    "spec §3.1a: save_freq must equal ceil(N_pie_train / 256)"
```

The SLURM scripts already hard-fail (`:?...`) if `PIE_STEPS_PER_EPOCH` / `PIE_TOTAL_STEPS` / `PIE_EVAL_STEPS` are unset, so a run cannot accidentally inherit DeepCoder's 83/1000/996.

## §3.2 Decoupling caveat (recorded verbatim from spec)

> PIE hyperparameters transcribed from the DeepCoder GOS script at build time, now independent. If DeepCoder is retuned, PIE does NOT auto-track; re-transcribe and re-run. No runtime monitoring of the DeepCoder folder.

After this build (2026-05-19), the PIE SLURM scripts do not import, source, or otherwise read from `experiment4_llm/scripts/`. They reuse `experiment4_llm.src.train.main` via the 2-line `train.py` wrapper (spec §2) and `experiment4_llm.src.verl_register.CUSTOM_ESTIMATORS` via verl's dispatch (no PIE-specific code touches those modules). The estimator core is therefore shared; hyperparameters are not.

## §3.3 Estimator-only (LOCKED)

Three PIE SLURM scripts share one parametric body; they differ only by `algorithm.adv_estimator` and run-name strings:

| Script | `algorithm.adv_estimator` | Source |
|--------|---------------------------|--------|
| `qwen3_1.7b_base_pie_maxrl_g16.sh` | `maxrl_unnorm` | `CUSTOM_ESTIMATORS["maxrl_unnorm"]` (GOS, ×G post-mean-centering) |
| `qwen3_1.7b_base_pie_grpo_g16.sh` | `grpo` | verl-native `AdvantageEstimator.GRPO`, self-normalizing |
| `qwen3_1.7b_base_pie_rloo_g16.sh` | `rloo` | `CUSTOM_ESTIMATORS["rloo"]` |

No other algorithmic knob differs. Sweeps over the same arm differ only in seed (see §3.6 W&B grouping).

## §3.4 SFT cold-start defaults (flagged MUST-RETUNE per spec)

Recorded from spec §3.4 for use at §14 step 7 when the SFT framework is built:

| Hyperparameter | Default (from PIE paper) | Status |
|----------------|--------------------------|--------|
| Mode | full fine-tune | flagged MUST-RETUNE |
| Learning rate | 1e-5 | flagged MUST-RETUNE |
| Global batch size | 32 | flagged MUST-RETUNE |
| Micro batch size | 2 | flagged MUST-RETUNE |
| Epochs | 1 | flagged MUST-RETUNE |
| Schedule | cosine + warmup | flagged MUST-RETUNE |
| Precision | bf16 | flagged MUST-RETUNE |
| `train_on_inputs` | False | locked (parser/loss-masking discipline) |
| Sequence length | (validate against staged C++ length distribution at §11) | pending |
| Default `sft.mode` | `hq` (`train_hq_only.jsonl`, ~4,085 pairs, no perf-conditioning) | per spec §6 |

## §3.5 Security (mandatory)

W&B API key leak observed in the existing DeepCoder scripts. PIE scripts read `WANDB_API_KEY` from the submission environment only; they hard-fail if unset. See `SECURITY_NOTE.md` for the full report and the human-only remediation steps (rotate, history-scrub, force-push). The agent has NOT performed remediation.

## §3.6 W&B grouping (LOCKED)

```
WANDB_RUN_GROUP = f(model, "pie", estimator, G)          # no seed
experiment_name = WANDB_RUN_GROUP + "_s{seed}"
```

Concrete group strings the PIE scripts set:

| Arm | `WANDB_RUN_GROUP` | `experiment_name` |
|-----|-------------------|-------------------|
| GOS / maxrl_unnorm | `qwen3-1.7b-base_pie_maxrl_unnorm_G16` | `..._s${SEED}` |
| GRPO | `qwen3-1.7b-base_pie_grpo_G16` | `..._s${SEED}` |
| RLOO | `qwen3-1.7b-base_pie_rloo_G16` | `..._s${SEED}` |

All seeds within an arm share one group → clean grouped GOS-vs-GRPO-vs-RLOO view.

## Drift items (require human confirmation at GATE A)

### D1 — Account and storage paths

| Field | Live DeepCoder script | PIE | Reason |
|-------|-----------------------|-----|--------|
| `#SBATCH --account` | `bgtu-dtai-gh` | `bgmo-dtai-gh` | Spec §1 + §3.1: "adapt paths/account to `bgmo`" |
| `--output` / `--error` log dir | `/projects/bgtu/sramasubramanian/cont-maxrl/slurm_logs/` | `/projects/bgmo/sramasubramanian/cont-maxrl/slurm_logs/` | bgmo |
| `STORAGE_ROOT` | `/work/nvme/bgtu/sramasubramanian/cont-maxrl` | `/work/nvme/bgmo/sramasubramanian/cont-maxrl` | bgmo |

Confirm before running: does the user have `bgmo-dtai-gh` allocation on `ghx4`, and are `bgmo` paths writable from compute nodes?

### D2 — Epoch arithmetic

DeepCoder hardcodes `total_training_steps=1000`, `total_epochs=15`, `save_freq=83`, `eval_steps=[83,166,...,996]`. These are DeepCoder-derived (`83 = ceil(deepcoder_train_size/256)`) and do not transfer.

PIE values are **computed from `N_pie_train` after §14 step 4 data verification**; the formula is recorded above and asserted at training time by the §10 isolation guard. The SLURM scripts hard-fail if `PIE_STEPS_PER_EPOCH` / `PIE_TOTAL_STEPS` / `PIE_EVAL_STEPS` are unset, preventing a run from accidentally inheriting DeepCoder's numbers.

Confirm: 10-epoch cap is the Tajwar-et-al. protocol convention (not a budget match); a still-rising trajectory at epoch 10 is a reportable finding, not a license to extend training silently.

### D3 — Checkpoint policy

| Field | DeepCoder | PIE | Spec |
|-------|-----------|-----|------|
| `trainer.max_actor_ckpt_to_keep` | `-1` (keep ALL) | `1` (rolling full ckpt + save-time actor carve-out hook) | §8.3 |
| Persistent per-epoch tree | none | `OUTPUT_DIR/actor_ckpts/step_{k*spe}/` containing only sharded `model_path` + `huggingface/`; never `optim_path` or `extra_state_path` | §8.3 |

The save-time hook itself is a §14 step 7 deliverable (cannot be a pure post-training script — the rolling ckpt is overwritten every epoch). The SLURM scripts include a no-op placeholder invocation today, and emit a NOTE if the hook is absent. Spec §8.3's "sharded-weights merge path code-verified at build (§14 2b), stop-and-hand-back if unclear" is the immediate next thing to do after GATE A approval.

Confirm: this is the right trade-off (one rolling resume ckpt + ten persisted actor-only carve-outs), and the disk budget tolerates ten copies of the actor weights.

### D4 — W&B API key

DeepCoder hardcodes the real W&B key on line 48. PIE scripts read from env only; they hard-fail if `WANDB_API_KEY` is unset. See `SECURITY_NOTE.md`.

Confirm: human has rotated + scrubbed history per `SECURITY_NOTE.md` before any PIE run.

### D5 — In-training eval data source

| Field | DeepCoder | PIE | Spec |
|-------|-----------|-----|------|
| `insitu_eval.data_file` | `livecodebench_v5_test.parquet` | PIE 200-problem val subset (`pie_val_200.parquet`; sampled deterministically at §14 step 4 from `val.jsonl`) | §8.6 / §9 |
| `insitu_eval.extra_data_files` | `humaneval_plus`, `mbpp_plus` (Python) | NONE | PIE is C++; Python-eval extras dropped |
| `insitu_eval.n_completions` | 200 | 200 (held for parity; revisit at §14 step 10 smoke if step time excessive) | parity |

Confirm: the in-training signal lives on the same 200-problem set §9's GO/NO-GO gate uses (spec §8.6 verbatim).

### D6 — `data.max_response_length`

PIE-permitted task-structural knob per spec §3.1. Held at `4096` until §11 validates against the staged C++ length distribution. The PIE permitted range allows raising it; a hard increase changes the throughput probe in §13. Recording this as drift even though no change has been made yet, so that any later increase is visible.

Confirm: leave at 4096 for now; revisit at §11 / §14 step 7.

### D7 — RLOO arm added

DeepCoder has no live `qwen3_1.7b_base_deepcoder_rloo_g16.sh` script (the `rloo_g16_bgtv.sh` variant exists but uses different paths and is the only live RLOO precedent). PIE adds an RLOO arm as a third estimator under the §3.3 estimator-only locked structure. The estimator dispatch is via `CUSTOM_ESTIMATORS["rloo"]` (already present in `verl_register.py`; verified at build time).

Confirm: the comparison set is {GOS, GRPO, RLOO} as spec §0 states ("GOS=`maxrl_unnorm`, GRPO=`grpo`, RLOO=`rloo`, …"). No other estimators in the initial scope.

### D8 — Test-case archive: unified union of THREE sources (built at §14 step 2b on 2026-05-19)

The user's staged JSONLs use the `LearningOpt/pie` schema (`src_code`/`tgt_code`/`problem_id`, **no `tests` column**) for train/val; only `test.jsonl` (978 records, 41 unique problem_ids) carries `tests`. Cross-split overlap is empty: `train ∩ test = train_hq_only ∩ test = val ∩ test = ∅`, so the test-split `tests` indices cannot be inherited by problem_id join.

Empirically, no single archive provides rich multi-test grading for all PIE problems. After §14-step-2b reads we staged and unified THREE sources at `/work/nvme/bgmo/sramasubramanian/pie_data/`:

| Source | Where | Layout | Problems | Tests/problem | train_hq_only coverage |
|---|---|---|---|---|---|
| **IBM Project_CodeNet** | `Project_CodeNet/derived/input_output/data/{pid}/` (extracted from `Project_CodeNet.tar.gz`, 7.77 GB from `codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/`) | `input.txt` + `output.txt` (**singular** — the CodeNet README is the canon, not pie-perf's plural) | 3,867 | exactly 1 ("Only the first sample input and sample output... are included" — CodeNet README) | **99.8%** (1471/1474) |
| **pie-perf "Public test cases"** | `pie_perf_tests/codenet/public_test_cases/{pid}/` (Google Drive `1RcUpZMOR8L2xYYWDZx7I0tHFzFgg7COO`, 5.9 MB zip) | `input.<i>.txt` + `output.<i>.txt`, i contiguous from 0 | 1,141 | min 1, max 7, **mean 2.9** | 28.3% (417/1474) |
| **pie-perf "Generated test cases"** | `pie_perf_generated/codenet/generated_test_cases/{pid}/` (Google Drive `1migwX4wpED0gDDxn7gS6q55vWeXIDgId`, 173 MB zip; alphacode-sourced per pie-perf README) | same as Public | 247 | ~100 (`p02676` has 100 inputs; matches the `test.jsonl[0]` `n_tests=103` figure within noise) | (computed in MANIFEST.json) |

**Unified layout (built by `experiment11_code_optim/precompute/build_unified_test_cases.py`):**

```
/work/nvme/bgmo/sramasubramanian/pie_data/tests/
├── MANIFEST.json                                ← per-problem {source, n_tests} + per-split coverage
└── {problem_id}/
    ├── input.0.txt
    ├── output.0.txt
    ├── ...
    └── input.<n_tests-1>.txt
        output.<n_tests-1>.txt
```

Source precedence per problem (richest first): **Generated → Public → CodeNet**. The on-disk `tests/{problem_id}/` indices are always contiguous starting at 0 regardless of source. CodeNet's singular `input.txt`/`output.txt` are copied to `input.0.txt`/`output.0.txt` to homogenise.

**Reward fn contract (revises spec §8.1 ground_truth shape):**
- `ground_truth = json.dumps({"problem_id": "p02587", "split": "train"})` — `test_indices` field is **dropped** (spec §8.1's `{"problem_id","test_indices","split"}` collapses to `{"problem_id","split"}` because "all indices" is implied by the on-disk archive's contiguous-from-0 layout)
- Reward fn does `glob(test_case_dir/{problem_id}/input.*.txt)` to discover the per-problem test count, pairs by index with the matching `output.<i>.txt`, then calls `measure(code_str, [(read(in_i), read(out_i)) for i in range(n_tests)])`
- §10 isolation guard asserts: (a) `test_case_dir/{problem_id}/` exists for every `problem_id` in the train parquet (missing → fail run); (b) the `n_tests` recorded in `MANIFEST.json` matches `len(glob input.*.txt))` on disk (detects partial extractions or drift between builds)

This change to the ground_truth shape is internal to PIE (the verl-side schema `{data_source, prompt, reward_model:{ground_truth}}` is unchanged); only the JSON inside `ground_truth` differs from the literal spec §8.1 wording.

### D9 — insitu_eval redirect mechanism

Decided: `sys.modules`-shim gated by `PIE_MODE` env var, applied in `experiment11_code_optim/train.py` (process-local; no on-disk monkey-patching). Full design in the "§8.6 redirect mechanism" section above. Implementation deferred to §14 step 7. The PIE SLURM scripts will `export PIE_MODE=1` before launching `train.py`.

### D10 — Checkpoint contents / merger interface

`vendor/verl/scripts/model_merger.py` exists (confirmed at build, full `--help` not captured because the script imports torch+verl at startup and takes >30s; the interface is documented inline in `experiment4_llm/src/eval_checkpoints.py:51-58`):

```python
[sys.executable, "vendor/verl/scripts/model_merger.py",
 "--backend", "fsdp",
 "--hf_model_path", BASE_MODEL,    # e.g., Qwen/Qwen3-1.7B-Base (for tokenizer/config skeleton)
 "--local_dir",    actor_dir,       # FSDP-sharded actor checkpoint
 "--target_dir",   hf_dir]          # output: HF-format model dir
```

This is the merger the §8.3 carve-out + post-hoc eval pipeline relies on. **PIE keeps verl's default `checkpoint_contents = ["model", "optimizer", "extra"]`** (NOT adding `"hf_model"`), matching DeepCoder. The save-time carve-out copies only the sharded `model_world_size_*_rank_*.pt` files + tokenizer/config files into `actor_ckpts/step_{N}/`; the post-hoc §8.4 eval job invokes `model_merger.py` to produce a loadable HF model at eval time. Disk budget: 10 carved actor checkpoints × ~3.4 GB (Qwen3-1.7B in bf16) = ~34 GB on `/work/nvme`, comfortably fits.

**Carve-out hook implementation point (§14 step 7):** verl's `FSDPCheckpointManager.save_checkpoint` (verified at `vendor/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py:112-218`) does not provide a built-in post-save callback. The hook will be implemented by **subclassing `FSDPCheckpointManager` and overriding `save_checkpoint` to call the parent's save then mirror the actor-only files**, wired in via a PIE-side `worker_cls` override in the SLURM config (one-line change in the script). Confirmed by reading the FSDPCheckpointManager — it's straightforward inheritance.

## §8.6 redirect mechanism — DECIDED at §14 step 2b (2026-05-19)

**Mechanism: `sys.modules` shim, gated by `PIE_MODE` env var, applied in `experiment11_code_optim/train.py` BEFORE the first import of `experiment4_llm.src.train.main`.**

The shim is process-local (only the PIE training process sees it), is not on-disk monkey-patching, and never contaminates any concurrent DeepCoder run that uses `experiment4_llm/src/train.py` directly. Implementation (to be added at §14 step 7, design locked here):

```python
# experiment11_code_optim/train.py — VERY TOP, before any verl import
import os, sys
if os.environ.get("PIE_MODE") == "1":
    # Spec §8.6: insitu_eval.py uses
    #   from experiment4_llm.src.reward_function import compute_scores_batch, extract_code
    # at line 19 — a hardcoded module path. Replace the module in sys.modules
    # BEFORE insitu_eval is imported (insitu_eval is imported lazily inside verl's
    # training loop, well after this wrapper runs), so its imports resolve to PIE's
    # reward fn. The on-disk experiment4_llm.src.reward_function is untouched.
    import experiment11_code_optim.reward.pie_reward_fn as _pie_rf
    sys.modules["experiment4_llm.src.reward_function"] = _pie_rf
    # evaluate.pass_at_k is dataset-agnostic; do NOT redirect.

from experiment4_llm.src.train import main

if __name__ == "__main__":
    main()
```

The three PIE SLURM scripts set `export PIE_MODE=1` before launching `train.py`; DeepCoder's scripts do not set it, so DeepCoder's behavior is unchanged when running concurrently.

**Env-var path fallback (also DECIDED):** PIE reward fn must accept `reference_table_dir` / `test_case_dir` via either `**kwargs` (training-reward-manager dispatch through `custom_reward_function.reward_kwargs`) OR `PIE_REFERENCE_TABLE_DIR` / `PIE_TEST_CASE_DIR` env vars (insitu_eval dispatch, which does NOT pass kwargs). Both paths must resolve to identical values; the §10 isolation guard asserts equality when both are present. SLURM scripts already `export` these env vars.

**Silent-zero HARD STOP** (spec §8.6, locked): §14 step 10 smoke verifies insitu_eval returns a non-zero reward on a known-good C++ program that the §4 measurement core passes. If insitu_eval still yields all-zeros after the sys.modules shim is wired (likely cause: shim applied too late, or pie_reward_fn raises silently and returns 0), STOP and hand back; do NOT autonomously debug deeper.

## GATE A — RESOLVED 2026-05-19 (corrected after user re-confirmation)

User decisions captured:

1. **RL train source:** `train.jsonl` (77,967 records, 1,474 unique problem_ids). N_pie_train = 77,967 → steps_per_epoch = 305 → total_training_steps = 3,050. **SFT cold-start source: `train_hq_only.jsonl`** (4,085 records, same 1,474 problems, fewer pairs per problem). Wall-clock projection ~356h optimistic = 8 chained 47h segments (matches the existing `CHAIN_LEFT=${2:-8}` default — fortunate coincidence). Will be re-projected at the §13 step 5 probe using real PIE per-step time.
2. **Test-case archive:** resolved at §14 step 2b. Three sources unioned into `/work/nvme/bgmo/sramasubramanian/pie_data/tests/{problem_id}/input.<i>.txt`+`output.<i>.txt` with source precedence Generated → Public → CodeNet (richest first). Full layout + coverage table is in D8. No JSONL `tests` column required — the per-problem test set is determined by the on-disk archive (contiguous indices 0..n-1 regardless of source). PIE reward fn's `ground_truth` is `{"problem_id","split"}` (drops spec §8.1's `test_indices`; "all indices" is implied by the archive).
3. **Account:** `bgmo-dtai-gh` is the user's allocation on `ghx4`. Alternative accounts available if needed: `{bgtv, bgtu, bgts, bgyb, bgyc, bgyd}-dtai-gh`. The PIE scripts use `bgmo-dtai-gh`; if exhausted, the human can `sbatch --account=...` override or edit the SBATCH directive.
4. **D6 — `max_response_length=4096` held**, revisit at §11 against the staged C++ length distribution.
5. **GATE A approved** — agent proceeded to §14 step 2b.
