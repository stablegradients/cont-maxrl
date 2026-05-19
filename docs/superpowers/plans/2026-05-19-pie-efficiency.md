# PIE Efficiency RL Implementation Plan

> **For agentic workers:** Authoritative source is **SPEC v9 (FINAL): PIE Instruction-Count Percentile Reward — GOS vs GRPO/RLOO on Code Optimization** delivered in the originating conversation, NOT this document. This file is a tracking artifact for the gate-bounded chunks in spec §14. REQUIRED SUB-SKILL: `superpowers:executing-plans` (inline; the spec mandates stop-and-hand-back at each §14 gate, which precludes subagent-driven autonomous batch execution past a gate).

**Goal:** Train Qwen3-1.7B-Base on the PIE dataset with the instruction-count percentile reward, comparing advantage estimators (GOS=`maxrl_unnorm`, GRPO=verl-native `grpo`, RLOO=`rloo`) on one 4×GH200 node, per the strict §14 execution order with explicit hand-back at each gate.

**Architecture:** PIE-specific package at `experiment11_code_optim/`. Reuses verl + the unchanged estimator core in `experiment4_llm/src/` via a 2-line `train.py` wrapper (spec §2). PIE-specific reward fn matches the verified `compute_score`/`compute_scores_batch` interface and accepts `**kwargs` for path injection from `custom_reward_function.reward_kwargs` (spec §8.2). C++ measurement core (g++ 14.2.0 + `perf stat -e instructions`; PAPI fallback; never wall-clock) in-process under the conda runtime per §4.0 primary architecture, pending §13 item-0 probe confirmation. Save-time actor carve-out for per-epoch eval trajectory (§8.3, hook deferred to §14 step 7). Post-hoc idempotent self-chaining eval job for pass@K + best@K, K∈{1,4,16,64,256}, n=256, across all 10 per-epoch carved checkpoints (§8.4–§8.5; best@K is the exact expectation over size-K subsets of the n eval samples — NOT to be called "unbiased").

**Tech stack:** verl (vendored), FSDP, vLLM, Qwen3-1.7B-Base, conda env `/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl`, `module load python/miniforge3_pytorch/2.10.0`, ghx4 partition (2-day), single 4× GH200 node (`bgmo-dtai-gh`), g++ 14.2.0 + perf/PAPI for C++ retired-instruction measurement.

**Hard rules (spec §0):** No `git commit`/`push`. Secrets from env only. End by handing back at each §14 gate. Never autonomously launch full SFT / precompute / training past a gate.

---

## §14 execution order (LOCKED) — gate map

- [x] **Step 1**: Branch `pie-efficiency` from `deltaAI` + `experiment11_code_optim/` skeleton — **done in current chunk**
- [x] **Step 2**: Parity transcription (read DeepCoder GOS+GRPO scripts, write independent PIE config/scripts, `PARITY_CONFIG.md`, `SECURITY_NOTE.md`) — **done in current chunk**
- [ ] **→ GATE A → HUMAN CONFIRMATION REQUIRED before any further work**
- [ ] **Step 2b**: Read in full and conform — `data_prep.py` (parquet schema), `insitu_eval.py` (FULL read; confirm §8.6 redirect mechanism + env-var path fallback), `eval_checkpoints.py` (pass@K confirmed; add §8.5 best@K and use ITS `pass_at_k`, not `evaluate.py`'s), verl checkpoint-merge path (§8.3 sharded-weights). Stop+hand-back if any conflicts (especially verl merge path unclear, or insitu_eval cannot be cleanly redirected without contaminating the shared module).
- [ ] **Step 3**: §13 **item 0 first** (runtime-boundary probe: SIF or conda?) → fixes §4 architecture → Deliverable 2 (C++ measurement core + invocation: `measure()`, gate, perf/PAPI median-of-3, per-core lock).
- [ ] **Step 4**: Deliverable 1 — verify staged PIE data (1a JSONL schema detection, 1b test-case tree resolver, 1c `REQUIRED_DATA_MANIFEST.json` + SHA256 + `PROVENANCE.md`). Stop+hand-back if incomplete (no partial proceed).
- [ ] **Step 5**: Rest of §13 — calibrate.py, determinism, throughput, precompute extrapolation, step-budget vs DeepCoder ~7-min step, **computed `steps_per_epoch` + `total_training_steps=10*steps_per_epoch` + projected total wall-clock** → `PROBE_REPORT.md` with decision GO / GO-WITH-MITIGATION / NO-GO.
- [ ] **→ GATE B → HUMAN GO REQUIRED**
- [ ] **Step 6**: Deliverable 3 — per-split reference precompute (b-train-only, no cross-split contamination, no `MIN_REFS` gate) → `REFERENCE_TABLE_REPORT.md` (prominent `n_refs` distribution).
- [ ] **→ GATE C → HUMAN GO REQUIRED**
- [ ] **Step 7**: Deliverables 4a (SFT framework, standalone HF `Trainer`+FSDP, full-state consolidated checkpoint, tiny smoke + **vLLM ≥1-token load gate**), 4b (verl integration, `reward/pie_reward_fn.py` with `**kwargs`, save-time actor carve-out hook, idempotent post-hoc eval-job scaffold with `eval_done.json` manifest, best@K estimator), 5 (`guards/isolation.py`), 6 (SLURM scaffolding finalized), §11 (prompt parity + round-trip + base-model compliance probe).
- [ ] **Step 8**: Hand back for human-launched **full** cold-start SFT.
- [ ] **Step 9**: §9 GO/NO-GO gate (200 PIE val problems, **≥20% with ≥8/16 strict-gate passing**, post-SFT, training temperature) → `GATE_REPORT.md`. NO-GO → §9.1 ladder (`hq→full`, `full+format-OK → Qwen3-4B-Base`, Qwen3-only); never autonomous.
- [ ] **Step 10**: One RL smoke step — wiring + in-process reward + non-degenerate reward histogram + `compute_score` dict shape + `reward_kwargs` resolve + **insitu_eval non-zero reward on known-good C++ that the §4 core passes (silent-zero hard stop)** + one save→carve-out→resume cycle + tiny post-hoc eval-job dry-run (1 checkpoint, reduced n acceptable here only) exercising the idempotent manifest + best@K/pass@K at K∈{1,4,16,64,256}.
- [ ] **Step 11**: Hand back for the human-launched full multi-day training decision.

---

## Current chunk: §14 steps 1-2 → GATE A

### Files written this chunk

- New branch `pie-efficiency` from `deltaAI` (no commits; spec hard rule)
- `experiment11_code_optim/`
  - `README.md` — navigation only
  - `train.py` — 2-line wrapper per spec §2 (`from experiment4_llm.src.train import main`)
  - `reward/`, `measurement/`, `precompute/`, `sft/`, `guards/`, `docs/` — empty subdirs reserved for later chunks
  - `scripts/qwen3_1.7b_base_pie_maxrl_g16.sh` — GOS arm (`maxrl_unnorm`)
  - `scripts/qwen3_1.7b_base_pie_grpo_g16.sh` — GRPO arm (verl-native `grpo`)
  - `scripts/qwen3_1.7b_base_pie_rloo_g16.sh` — RLOO arm (CUSTOM_ESTIMATORS `rloo`)
  - `PARITY_CONFIG.md` — verified parity, 7 drift items, epoch formula, Tajwar rationale, decoupling caveat, redirect-mechanism placeholder for §14 step 2b
  - `SECURITY_NOTE.md` — W&B key leak in `experiment4_llm/scripts/*.sh:48`, human-only remediation
- `docs/superpowers/plans/2026-05-19-pie-efficiency.md` — this file

### Why no TDD scaffolding in this chunk

Spec §14 step 2 is parity transcription: copying validated hyperparameters from the live DeepCoder scripts into an independent PIE skeleton. The artifacts are SLURM bash scripts + markdown notes; there is no executable unit to test. **Future chunks WILL use TDD strictly**:

- §14 step 3 (C++ `measure()` core): unit tests on known-good / known-fail / known-timeout / known-segfault programs before wiring into the reward fn.
- §14 step 6 prerequisites (midrank percentile, best@K, shared parser anchored on `### Optimized Version:`): unit tests before implementation, including the n_refs=1 degenerate-{0,1} case and the bisect_left bug-avoidance check.
- §14 step 7 (reward fn integration): integration tests verifying `compute_score` returns the `{"score": float, "acc": float}` dict, `compute_scores_batch` accepts the verified signature with `**kwargs`, and isolation-guard assertions fire on the right violations.

### Why no commits this chunk

Spec §0 hard rule: "No `git commit`/`push`." All artifacts live on the working tree of branch `pie-efficiency`, awaiting human approval at GATE A before any commit decision is made.

### GATE A hand-back triggers

The agent stops after writing the files above and presents to the human:

1. The verified parity (all shared hparams confirmed from script reads — see `experiment11_code_optim/PARITY_CONFIG.md`).
2. The seven drift items D1–D7 requiring confirmation (account, epoch arithmetic formula + 10-epoch=Tajwar rationale, checkpoint policy, W&B key handling, in-training eval data source, `max_response_length` deferred decision, RLOO arm added).
3. Open questions: PIE RL training data source (recommend `train.jsonl`), missing test-case archive at `/work/nvme/bgmo/.../pie_data/raw/` (to be re-checked at §14 step 4), SFT defaults flagged MUST-RETUNE.
4. No proceed to §14 step 2b without explicit human approval.

---

## Verification commands (for the human at GATE A)

```bash
# Branch + files exist
git branch --show-current               # → pie-efficiency
git status -s | grep experiment11_code_optim

# PIE scripts differ from DeepCoder ONLY in expected lines (estimator + names + paths)
diff -u \
  experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_maxrl_g16.sh \
  experiment11_code_optim/scripts/qwen3_1.7b_base_pie_maxrl_g16.sh

# The 3 PIE scripts differ from each other only in expected lines
diff -u \
  experiment11_code_optim/scripts/qwen3_1.7b_base_pie_maxrl_g16.sh \
  experiment11_code_optim/scripts/qwen3_1.7b_base_pie_grpo_g16.sh

# train.py is the 2-line wrapper
cat experiment11_code_optim/train.py

# No W&B key leak in PIE scripts
grep -n 'WANDB_API_KEY' experiment11_code_optim/scripts/*.sh
# Expected: only ${WANDB_API_KEY:?...} guards and `export WANDB_ENTITY=...`, no literal keys.

# Hard-fail behavior if epoch arithmetic missing
bash -n experiment11_code_optim/scripts/qwen3_1.7b_base_pie_maxrl_g16.sh
# Syntactic check only; do NOT submit yet.
```
