# Qwen3 LCB Evaluation Bug Report

**Date**: 2026-03-20
**Affected**: All LCB evaluation scripts and training insitu_eval configs

## Bugs Found and Fixed (in source code)

### Bug 1: Missing `class Solution` Support (CRITICAL)
- **File**: `experiment4_llm/src/sandbox.py`
- **Impact**: 381/876 functional (LeetCode-style) problems scored ZERO
- **Root cause**: Sandbox only looked up standalone functions via `_ns.get(fn_name)`. Official LCB also checks `Solution().method_name`.
- **Fix**: Added fallback: `if _fn is None and 'Solution' in _ns: _sol = _ns['Solution'](); _fn = getattr(_sol, fn_name, None)`

### Bug 2: Wrong Code Extraction (CRITICAL)
- **File**: `experiment4_llm/src/reward_function.py` → `extract_code()`
- **Impact**: Extracted wrong code block when model refines its answer
- **Root cause**: Took the FIRST code block between backticks. Official LCB takes the LAST.
- **Fix**: Changed to `backtick_indices[-2]` / `backtick_indices[-1]`

### Bug 3: Wrong Prompt Template (HIGH)
- **File**: `experiment4_llm/src/data_prep.py` → `format_prompt()`
- **Impact**: Model never saw starter code for functional problems, wrong format instructions
- **Root cause**: Custom prompt format instead of official LCB template
- **Fix**: Rewrote to match official LCB: `### Question:\n{content}\n\n### Format: ...starter code...\n### Answer:`

### Bug 4: Restrictive System Prompt (HIGH)
- **File**: `experiment4_llm/src/data_prep.py`, multiple scripts
- **Impact**: Constrained model output, especially thinking models
- **Old**: "You are an expert Python programmer. Output ONLY the code..."
- **Fix**: Official LCB: `"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."`

### Bug 5: Missing Import Prepend (MEDIUM)
- **File**: `experiment4_llm/src/sandbox.py`
- **Impact**: Call-based code that relied on common imports (collections, heapq, typing, etc.) failed
- **Root cause**: Official LCB prepends a massive import block before executing call-based code
- **Fix**: Added `_IMPORT_STRING` constant matching official `testing_util.py`, prepended via `exec(repr(_IMPORT_STRING) + "\n\n" + code)`

### Bug 6: Missing stdin Patching (MEDIUM)
- **File**: `experiment4_llm/src/sandbox.py`
- **Impact**: stdin problems that use `open()` or `sys.stdin.readline()` failed
- **Root cause**: Only patched `sys.stdin`, not `builtins.open`, `sys.stdin.readline`, `sys.stdin.readlines`, `sys.stdin.read`
- **Fix**: Added AST-based `if __name__` stripping, `builtins.open` mock, full stdin patching

### Bug 7: Lenient Output Matching (MEDIUM)
- **File**: `experiment4_llm/src/reward_function.py` → `_outputs_match()`
- **Impact**: Inflated pass rates by accepting near-matches
- **Old behavior**: Float tolerance (`rel_tol=1e-4`), TACO list unwrapping, `_tuples_to_lists()` deep coercion
- **Fix**: Strict matching per official LCB:
  - Call-based: `json.loads()` then `==`, only top-level tuple→list coercion
  - Stdin: line-by-line stripped comparison, exact `Decimal` equality (no epsilon)

## Bugs Found but NOT Yet Fixed (in scripts)

### Bug 8: Wrong LCB Dataset in Eval Scripts (HIGH)
- **Impact**: Evaluating on release_v5 (876 problems) instead of v5-only (167 problems). Paper numbers are NOT comparable.
- **Root cause**: `version_tag="release_v5"` gives cumulative 880 problems. Qwen3 paper evaluates on `version_tag="v5"` (167 problems, 2024-09-22 to 2025-01-04).
- **Affected scripts** (eval):
  - `qwen3_1.7b_base_lcb.sh` → uses `livecodebench_test.parquet` (876)
  - `qwen3_1.7b_thinking_budget_sweep.sh` → uses `livecodebench_test.parquet` (876)
  - `qwen3_1.7b_thinking_budget_32k.sh` → uses `livecodebench_test.parquet` (876)
  - `qwen3_1.7b_thinking_lcbv5.sh` → uses `livecodebench_test.parquet` (876)
  - `qwen3_4b_thinking_lcbv5.sh` → uses `livecodebench_test.parquet` (876)
  - `qwen3_4b_deepcoder_eval.sh` → uses `livecodebench_test.parquet` (876)
- **Affected scripts** (training insitu_eval):
  - `qwen3_1.7b_base_deepcoder_grpo.sh`
  - `qwen3_1.7b_base_deepcoder_maxrl.sh`
  - `qwen3_1.7b_instruct_deepcoder_grpo.sh`
  - `qwen3_1.7b_instruct_deepcoder_maxrl.sh`
  - `qwen3_4b_deepcoder_grpo.sh`
  - `qwen3_4b_deepcoder_maxrl.sh`
- **Workaround**: Current results on 876 problems CAN be filtered to v5-only (indices 713-875) for paper comparison. See "Quick Filter" section below.
- **Proper fix**: Change `DATA_FILE` to `livecodebench_v5_test.parquet` in all scripts. Note: our v5 parquet has 163 problems vs official 167 (4 lost to decompression/filtering).

### Bug 9: Timeout 5s Instead of 6s (LOW)
- **Impact**: Marginal — a few edge-case solutions may timeout at 5s but pass at 6s
- **Affected**: ALL LCB eval scripts use `--timeout_per_test 5`
- **Fix**: Change to `--timeout_per_test 6`

### Bug 10: Missing System Prompt in Some Scripts (LOW)
- **Impact**: Falls back to system prompt baked into parquet (which is now correct), so NOT a real issue
- **Affected**: `qwen3_1.7b_base_lcb.sh`, `qwen3_1.7b_thinking_budget_32k.sh`, `qwen3_4b_deepcoder_eval.sh` — no `--system_prompt` flag
- **Note**: Since the parquet data now has the correct system prompt, these scripts work correctly. The `--system_prompt` CLI flag is an OVERRIDE, not required.

## Dataset Reference

| Version tag | Problems | Date range | Use case |
|-------------|----------|------------|----------|
| `v5` | 167 | 2024-09-22 to 2025-01-04 | Paper comparison (Qwen3 tech report) |
| `release_v5` | 880 | 2023-05 to 2025-01 | Full cumulative benchmark |
| Our `livecodebench_v5_test.parquet` | 163 | v5-only | 4 problems lost to filtering |
| Our `livecodebench_test.parquet` | 876 | release_v5 | 4 problems lost to filtering |

## Quick Filter: V5 Scores from 876-Problem Results

V5 problems are at indices 713-875 in the 876-problem dataset. To extract v5-only pass@1:

```python
import json
results = json.load(open("eval_results.json"))
v5 = [p for p in results['per_problem'] if p['problem_idx'] >= 713]
p1 = [p['pass_at_k']['1'] for p in v5 if p['pass_at_k']['1'] is not None]
print(f"V5 pass@1: {sum(p1)/len(p1)*100:.1f}% ({len(v5)} problems)")
```

## Current Results (with fixes applied to source, 876-problem dataset)

| Model | Config | Full 876 p@1 | V5-only p@1 | Paper (v5) |
|-------|--------|-------------|-------------|------------|
| 1.7B Base | greedy | 15.5% | 7.4% | — |
| 1.7B Base | sampling | 12.7% | 5.4% | — |
| 1.7B Think | 4k budget | 23.9% | 7.2% | — |
| 1.7B Think | 8k budget | 40.6% | 16.6% | — |
| 1.7B Think | 32k budget | pending | pending | **33.2%** |

## Sampling Configuration (matches Qwen3 paper, page 13)

- Temperature: 0.6, top_p: 0.95, top_k: 20
- n_completions: 10 (paper uses 8 independent reasoning attempts for CodeForces, methodology varies)
- max_output_length: 32,768 tokens
- presence_penalty: 0.0

## Impact on Training

Training scripts are NOT affected by bugs 1-7 because:
- Training reward uses the same `reward_function.py` and `sandbox.py` (now fixed)
- Training data parquets were regenerated with the correct prompt template and system prompt
- The insitu_eval dataset issue (876 vs 167) only affects monitoring — training reward is computed on the training dataset (DeepCoder), not LCB

However, insitu_eval pass@1 numbers logged during training are inflated (876 easier problems). For paper-comparable insitu_eval, switch to `livecodebench_v5_test.parquet`.
