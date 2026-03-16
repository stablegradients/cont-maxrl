"""
Continuous reward function for code generation: fraction of test cases passed.

Training reward:  r = (# test cases passed) / (# total test cases) in [0, 1]
This is NOT the standard binary reward. A solution passing 7/10 tests gets r=0.7.

verl reward function interface:
    compute_score(solution_str, ground_truth, ...) -> float
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from experiment4_llm.src.sandbox import execute_batch_in_sandbox, execute_in_sandbox


def extract_code(solution_str: str) -> str:
    """
    Extract Python code from a model completion.
    Handles markdown code blocks, raw code, and thinking model output.
    Works with skip_special_tokens=False (preserves <think></think> but
    may include other special tokens like <|im_end|>, <|endoftext|>).
    """
    solution_str = solution_str.strip()

    # Strip special tokens (from skip_special_tokens=False decode)
    solution_str = re.sub(r"<\|[^>]+\|>", "", solution_str).strip()

    # Strip thinking content for reasoning models (e.g., Qwen3, DeepSeek-R1)
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>", 1)[1].strip()

    # Try to extract from markdown code block
    solution_lower = solution_str.lower()
    if "```python" in solution_lower:
        # Find the position case-insensitively, then extract from original
        idx = solution_lower.index("```python")
        after = solution_str[idx + len("```python"):]
        # Skip optional "3" after "python"
        if after.startswith("3") and (len(after) == 1 or after[1] in "\n\r \t"):
            after = after[1:]
        code = after.split("```")[0]
        return code.strip()

    if "```" in solution_str:
        parts = solution_str.split("```")
        if len(parts) >= 3:
            code = parts[1]
            # Remove optional language identifier on first line
            lines = code.split("\n")
            if lines and lines[0].strip().lower() in ("python", "python3", "py", ""):
                lines = lines[1:]
            return "\n".join(lines).strip()

    # No code block markers -- treat entire string as code
    return solution_str


def compute_score(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 5,
    memory_limit_mb: int = 512,
    data_source: str | None = None,
    extra_info: dict | None = None,
) -> dict:
    """
    Compute continuous reward: fraction of test cases passed.

    Args:
        solution_str: the model's generated code (may include markdown)
        ground_truth: JSON string with test cases:
            {"inputs": [...], "outputs": [...], "fn_name": optional}
        timeout_per_test: seconds per test case
        memory_limit_mb: memory limit per test case

    Returns:
        dict with:
            "score": float in [0, 1] (fraction of test cases passed)
            "acc": 1.0 if all tests passed, 0.0 otherwise
    """
    # Parse ground truth
    try:
        test_cases = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "acc": 0.0}

    inputs = test_cases.get("inputs", [])
    expected_outputs = test_cases.get("outputs", [])
    fn_name = test_cases.get("fn_name", None)

    if not inputs or not expected_outputs:
        return {"score": 0.0, "acc": 0.0}

    num_total = min(len(inputs), len(expected_outputs))
    if num_total == 0:
        return {"score": 0.0, "acc": 0.0}

    # Extract code from model output
    code = extract_code(solution_str)
    if not code:
        return {"score": 0.0, "acc": 0.0}

    num_passed = 0

    for i in range(num_total):
        inp = str(inputs[i]) if not isinstance(inputs[i], str) else inputs[i]
        expected = str(expected_outputs[i]) if not isinstance(expected_outputs[i], str) else expected_outputs[i]

        try:
            actual = execute_in_sandbox(
                code=code,
                input_data=inp,
                fn_name=fn_name,
                timeout_seconds=timeout_per_test,
                memory_limit_mb=memory_limit_mb,
            )
            if _outputs_match(actual.strip(), expected.strip(), fn_name):
                num_passed += 1
        except (TimeoutError, MemoryError, RuntimeError, SyntaxError, Exception):
            pass  # Failed test case -- counts as not passed

    score = num_passed / num_total
    acc = 1.0 if num_passed == num_total else 0.0
    return {"score": score, "acc": acc}


def _outputs_match(actual: str, expected: str, fn_name: str | None) -> bool:
    """Compare actual and expected outputs.

    For stdin/stdout problems: exact string match, with numeric fallback.
    For fn_name problems: JSON-based semantic comparison (handles list
    wrapping differences in TACO/APPS datasets).
    """
    # Exact string match (works for both stdin/stdout and fn_name)
    if actual == expected:
        return True

    # Numeric fallback: handle floating-point formatting differences
    if _numeric_match(actual, expected):
        return True

    # For fn_name problems: compare parsed JSON values
    if fn_name is not None:
        try:
            actual_val = json.loads(actual)
        except (json.JSONDecodeError, TypeError):
            actual_val = actual
        try:
            expected_val = json.loads(expected)
        except (json.JSONDecodeError, TypeError):
            expected_val = expected
        if actual_val == expected_val:
            return True
        # Handle TACO convention: expected may wrap return value in [result]
        if isinstance(expected_val, list) and len(expected_val) == 1:
            if actual_val == expected_val[0]:
                return True

    return False


def _numeric_match(a: str, b: str, rel_tol: float = 1e-4) -> bool:
    """Check if two output strings represent the same number(s).

    Handles single numbers and multi-line numeric outputs.
    """
    a_lines = a.strip().splitlines()
    b_lines = b.strip().splitlines()
    if len(a_lines) != len(b_lines):
        return False
    for al, bl in zip(a_lines, b_lines):
        al, bl = al.strip(), bl.strip()
        if al == bl:
            continue
        try:
            af, bf = float(al), float(bl)
            if bf == 0.0:
                if af != 0.0:
                    return False
            elif abs(af - bf) / max(abs(bf), 1e-15) > rel_tol:
                return False
        except ValueError:
            return False
    return True


def compute_score_binary(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 5,
    memory_limit_mb: int = 512,
) -> float:
    """
    Binary reward: 1.0 if ALL test cases pass, 0.0 otherwise.
    Used for evaluation (pass@k), not training.
    """
    result = compute_score(
        solution_str, ground_truth, timeout_per_test, memory_limit_mb
    )
    return 1.0 if result["score"] == 1.0 else 0.0


def compute_score_binary_train(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 5,
    memory_limit_mb: int = 512,
    data_source: str | None = None,
    extra_info: dict | None = None,
) -> dict:
    """
    Binary training reward: 1.0 if ALL test cases pass, 0.0 otherwise.

    Same interface as compute_score (dict return with data_source/extra_info params)
    so it plugs directly into verl's reward manager.
    """
    result = compute_score(
        solution_str, ground_truth, timeout_per_test, memory_limit_mb,
        data_source=data_source, extra_info=extra_info,
    )
    binary = 1.0 if result["acc"] == 1.0 else 0.0
    return {"score": binary, "acc": binary}


def compute_scores_batch_binary_train(
    reward_inputs: list[dict],
    timeout_per_test: int = 5,
    memory_limit_mb: int = 512,
    num_workers: int | None = None,
    global_timeout: int = 60,
) -> list[dict]:
    """Batch binary rewards: binarizes the output of compute_scores_batch."""
    results = compute_scores_batch(
        reward_inputs, timeout_per_test, memory_limit_mb, num_workers, global_timeout,
    )
    for r in results:
        binary = 1.0 if r["acc"] == 1.0 else 0.0
        r["score"] = binary
    return results


# ---------------------------------------------------------------------------
# Batch scoring: syntax filter → dedup → parallel batched subprocesses
# ---------------------------------------------------------------------------

_ZERO = {"score": 0.0, "acc": 0.0}


def _run_completion(task: tuple) -> tuple:
    """Run all tests for one unique (code, ground_truth) pair in one subprocess."""
    dedup_key, code, inputs, expected_outputs, fn_name, timeout, mem, global_timeout = task
    n = len(inputs)
    try:
        test_outputs = execute_batch_in_sandbox(
            code=code,
            inputs=inputs,
            fn_name=fn_name,
            timeout_per_test=timeout,
            memory_limit_mb=mem,
            global_timeout=global_timeout,
        )
        passed = 0
        for actual, expected in zip(test_outputs, expected_outputs):
            if actual is not None:
                exp = str(expected) if not isinstance(expected, str) else expected
                if _outputs_match(actual.strip(), exp.strip(), fn_name):
                    passed += 1
        score = passed / n
        acc = 1.0 if passed == n else 0.0
        return (dedup_key, {"score": score, "acc": acc})
    except Exception:
        return (dedup_key, dict(_ZERO))


def compute_scores_batch(
    reward_inputs: list[dict],
    timeout_per_test: int = 5,
    memory_limit_mb: int = 512,
    num_workers: int | None = None,
    global_timeout: int = 60,
) -> list[dict]:
    """Compute rewards for all completions with three compounding optimizations.

    1. Syntax pre-check: compile() rejects ~40% of completions at zero cost.
    2. Code dedup: hash extracted code — identical completions scored once.
    3. Batched subprocess: one fork per unique completion, all tests inside
       with signal.alarm per-test timeout + parent global_timeout cap.
       No early-stop: all tests always run to preserve continuous reward signal.

    Parallelism is completion-level via ThreadPoolExecutor, but each completion
    is bounded to global_timeout seconds so the tail is capped.

    Args:
        reward_inputs: list of dicts with "solution_str" and "ground_truth"
        timeout_per_test: per-test alarm timeout (seconds)
        memory_limit_mb: memory limit per subprocess
        num_workers: thread pool size (default from VERL_REWARD_WORKERS)
        global_timeout: hard wall-clock cap per subprocess (seconds)

    Returns:
        list of dicts with "score" and "acc", one per input completion
    """
    if num_workers is None:
        num_workers = int(os.environ.get("VERL_REWARD_WORKERS", 256))

    results = [dict(_ZERO) for _ in range(len(reward_inputs))]

    # ── Phase 1: Parse, extract, syntax-check, dedup ──
    # dedup_key = hash(code + ground_truth) → only run tests once per unique pair
    dedup_tasks: dict[str, tuple] = {}  # key → task tuple
    completion_to_key: dict[int, str] = {}  # completion idx → dedup key

    for i, item in enumerate(reward_inputs):
        ground_truth = item["ground_truth"]
        try:
            test_cases = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            continue

        inputs = test_cases.get("inputs", [])
        expected_outputs = test_cases.get("outputs", [])
        fn_name = test_cases.get("fn_name", None)

        if not inputs or not expected_outputs:
            continue

        num_total = min(len(inputs), len(expected_outputs))
        code = extract_code(item["solution_str"])
        if not code:
            continue

        # Syntax pre-check — zero cost, eliminates broken code
        try:
            compile(code, "<test>", "exec")
        except SyntaxError:
            continue

        # Dedup by (code, ground_truth) — same code + same tests = same score
        key = hashlib.md5((code + ground_truth).encode()).hexdigest()
        completion_to_key[i] = key

        if key not in dedup_tasks:
            inp_strs = [
                str(inputs[j]) if not isinstance(inputs[j], str) else inputs[j]
                for j in range(num_total)
            ]
            exp_strs = [
                str(expected_outputs[j])
                if not isinstance(expected_outputs[j], str)
                else expected_outputs[j]
                for j in range(num_total)
            ]
            dedup_tasks[key] = (
                key, code, inp_strs, exp_strs, fn_name,
                timeout_per_test, memory_limit_mb, global_timeout,
            )

    # ── Phase 2: Execute unique completions in parallel ──
    dedup_results: dict[str, dict] = {}

    if dedup_tasks:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for key, score_dict in pool.map(_run_completion, dedup_tasks.values()):
                dedup_results[key] = score_dict

    # ── Phase 3: Scatter scores back to all completions (including duplicates) ──
    for i, key in completion_to_key.items():
        results[i] = dedup_results.get(key, dict(_ZERO))

    return results
