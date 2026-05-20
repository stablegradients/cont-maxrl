# Deferred Fixes — Tracking Doc

Each item below was deferred during the two-level-scheduler review (commits
`938f872` + `0e04f08`). Every entry pins the **target step** at which the
fix lands and a **verification criterion** that must be checked off before
the gate for that step is signed off. Inline `TODO(step N)` comments in the
relevant files point back to this file by its `F#` tag.

When a step closes, its `F#` items move from `OPEN` to `CLOSED` with the
commit hash that landed the fix. Do not close the gate for a step until
every `F#` tagged to that step is `CLOSED`.

---

## F1 — `instr_count` → `elapsed_ns` rename
**Status:** CLOSED (landed in the F1 commit, ahead of the precompute commit;
the reference-table precompute writes `"elapsed_ns_values"` from the start)
**Target step:** §14 step 6 (reference-table precompute)
**Severity:** Medium — semantic mismatch; not a bug because both sides of the
percentile comparison use the same units, but the name is misleading.

**What to do:**
1. Rename dict field `instr_count` → `elapsed_ns` across:
   - `experiment11_code_optim/measurement/core.py` (`measure()` return dict)
   - `experiment11_code_optim/reward/pie_reward_fn.py`
     (`compute_scores_batch` aggregation; `_percentile_from_reference_table`)
   - `experiment11_code_optim/precompute/probe.py` (any reads)
   - `experiment11_code_optim/tests/test_measurement.py` (assertion field
     references — search for `instr_count`)
   - `experiment11_code_optim/tests/test_pie_reward_fn.py` (likewise)
2. Rename reference-table JSON key `"instr_counts"` → `"elapsed_ns_values"`.
3. Remove `instr_count` TODOs once renamed.

**Verification criteria (GATE C cannot close without all green):**
- `grep -r "instr_count" experiment11_code_optim/` returns ONLY references in
  PROBE_REPORT.md or other historical docs (no code, no JSON, no tests).
- The precompute writes `"elapsed_ns_values"`, the reader reads
  `"elapsed_ns_values"`; no fallback to the old key.
- Existing `test_pie_reward_fn.py::test_compute_scores_batch_known_good_program_passes_via_dedup_path`
  still passes against the new schema.

---

## F5 — `global_timeout` is accepted but never enforced
**Status:** CLOSED (commit landing this with the SFT/step-7 work)
**Target step:** §14 step 7 (verl integration / isolation guard)
**Severity:** Medium — current code accepts `global_timeout=60` and silently
ignores it. A pathological program at 9.9s/test × K=5 = 49.5s would block
one test-run worker (only one, not 63 — the two-level scheduler limits
the blast radius compared to the previous dispatch). Still: the parameter
is a lie until enforced.

**What to do:**
Wrap the phase-3 (`_run_one`) pool in `concurrent.futures.wait(timeout=global_timeout)`
or equivalent; on timeout cancel pending futures and mark un-completed
grades' results as `all_pass=False, error="batch timeout"`.

**Verification criteria (cannot close §14 step 7 without):**
- New test in `test_pie_reward_fn.py` builds a batch with one program that
  sleeps `> global_timeout` seconds; the call returns within
  `global_timeout * 1.5` and that program's score is `0.0`.
- Other programs in the same batch return their normal scores (single bad
  rollout does not poison the whole batch).

---

## F6 — `insitu_eval` redirect (PIE_MODE shim) NOT IMPLEMENTED
**Status:** CLOSED (commit landing this with the SFT/step-7 work)
**Target step:** §14 step 7
**Severity:** High — this is the silent-failure trap spec §8 warns about.

**Current state:** The module docstring of `pie_reward_fn.py` describes a
`sys.modules` shim that redirects `experiment4_llm.src.reward_function` to
the PIE module when `PIE_MODE=1`. **The shim does not exist.** Until it
lands, `insitu_eval` will import DeepCoder's reward function, grade C++
programs as Python, and every rollout will score 0 with no error.

**What to do:**
1. Add the shim at the entrypoint of `experiment11_code_optim/train.py`
   (BEFORE any other imports that pull in `experiment4_llm.src`):
   ```python
   import os, sys
   if os.environ.get("PIE_MODE") == "1":
       from experiment11_code_optim.reward import pie_reward_fn
       sys.modules["experiment4_llm.src.reward_function"] = pie_reward_fn
   ```
2. Set `export PIE_MODE=1` in all 3 PIE SLURM scripts.
3. Add a smoke test in `test_pie_reward_fn.py` that constructs the shim,
   imports `experiment4_llm.src.reward_function`, calls
   `compute_score(...)` with the known-good `p02587` `tgt_code`, and asserts
   `result["acc"] == 1.0`.

**Verification criteria (cannot close §14 step 7 / cannot launch §11 full
training without):**
- The smoke test above passes.
- Step 10 RL smoke step shows non-zero `insitu_eval` scores on a known-good
  C++ program. If all insitu_eval scores are 0, the shim is not wired in.

---

## F7 — `compute_score` `**kwargs` is load-bearing (informational, do NOT remove)
**Status:** NOTE-ONLY (no change required)
**Severity:** Low — already correct.

`pie_reward_fn.compute_score` has `**kwargs` to absorb `test_case_dir` and
`reference_table_dir` from verl's `reward_kwargs` injection. DeepCoder's
sibling does not, because it does not take those args. If someone "cleans
up" the `**kwargs` later, verl will pass the unknown kwarg, the call will
raise `TypeError`, verl's try/except will swallow it, and every rollout
will silently score 0.

**Verification criteria (just don't break it):**
- `git grep "def compute_score(" experiment11_code_optim/` shows `**kwargs`.
- If removing in a refactor: replace with explicit kwargs the SLURM script
  actually passes, so the failure is loud (TypeError unwrapped).

---

## F8 — `extract_code` fallback returns whole text when no fence found
**Status:** LEAVE AS-IS (intentional)
**Severity:** Low cosmetic.

When the model emits a response without a ```cpp fence, `extract_code`
falls through to returning the full cleaned text. That text then goes to
g++ → compile fails → score 0 — which is the correct outcome. Cost is one
unnecessary `_compile` call per unique unparseable code (~0.35s with PCH);
dedup amortizes identical unparseables.

**Not worth fixing** unless a future profile shows it dominates.

---

## How to use this file

When picking up §14 step 6 or step 7:

1. Search this file for `Target step: §14 step N`.
2. For each OPEN item with that step, read its **What to do** and
   **Verification criteria** sections.
3. Implement.
4. Run the verification commands listed.
5. Update the item to `CLOSED` with the commit hash.
6. ONLY then sign off on the step's gate.
