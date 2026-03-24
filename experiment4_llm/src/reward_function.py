"""
Continuous reward function for code generation: fraction of test cases passed.

Training reward:  r = (# test cases passed) / (# total test cases) in [0, 1]
This is NOT the standard binary reward. A solution passing 7/10 tests gets r=0.7.

verl reward function interface:
    compute_score(solution_str, ground_truth, ...) -> float
"""
from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing
import os
import re

from experiment4_llm.src.sandbox import execute_batch_in_sandbox, execute_in_sandbox, pool_execute_completion


def extract_code(solution_str: str) -> str:
    """
    Extract Python code from a model completion.

    Follows the official LiveCodeBench extraction convention: when multiple
    code blocks are present, use the **last** complete block (the model may
    refine its answer).

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

    # Find all ``` delimiters and extract the LAST complete code block
    # (matches official LiveCodeBench extraction)
    lines = solution_str.split("\n")
    backtick_indices = [i for i, line in enumerate(lines) if "```" in line]

    if len(backtick_indices) >= 2:
        # Take the last pair of backtick lines
        start_idx = backtick_indices[-2]
        end_idx = backtick_indices[-1]
        code_lines = lines[start_idx + 1 : end_idx]
        # Remove language identifier on the opening ``` line if present
        # (the start_idx line itself may have ```python — skip it via +1)
        return "\n".join(code_lines).strip()

    # No complete code block — treat entire string as code
    return solution_str


def compute_score(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 6,
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

    Matches the official LiveCodeBench evaluation logic:
    - Call-based (fn_name): direct Python == on parsed values, tuple→list only.
    - Stdin/stdout: line-by-line stripped comparison, Decimal fallback (exact,
      no epsilon tolerance).
    """
    # Exact string match (works for both)
    if actual == expected:
        return True

    if fn_name is not None:
        # ── Call-based: parse JSON, compare Python objects ──
        # Official LCB: prediction == gt_out, with tuple→list coercion
        try:
            actual_val = json.loads(actual)
        except (json.JSONDecodeError, TypeError):
            return False
        try:
            expected_val = json.loads(expected)
        except (json.JSONDecodeError, TypeError):
            return False
        # Coerce tuples to lists (official LCB does this)
        if isinstance(actual_val, tuple):
            actual_val = list(actual_val)
        return actual_val == expected_val
    else:
        # ── Stdin/stdout: line-by-line stripped, Decimal fallback ──
        # Official LCB: exact string per line, then exact Decimal comparison
        return _stdio_match(actual, expected)


def _stdio_match(actual: str, expected: str) -> bool:
    """Line-by-line comparison matching official LCB grade_stdio logic.

    Uses exact Decimal equality (no epsilon tolerance) as fallback.
    """
    from decimal import Decimal, InvalidOperation

    a_lines = [line.strip() for line in actual.strip().split("\n")]
    b_lines = [line.strip() for line in expected.strip().split("\n")]

    if len(a_lines) != len(b_lines):
        return False

    for al, bl in zip(a_lines, b_lines):
        if al == bl:
            continue
        # Exact Decimal comparison (no tolerance, matches official LCB)
        try:
            da = [Decimal(x) for x in al.split()]
            db = [Decimal(x) for x in bl.split()]
        except (InvalidOperation, ValueError):
            return False
        if da != db:
            return False

    return True


def compute_score_binary(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 6,
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
    timeout_per_test: int = 6,
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
    timeout_per_test: int = 6,
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


_forkserver_pool = None
_forkserver_pool_size = 0


def _get_forkserver_pool(num_workers):
    """Get or create a forkserver-based multiprocessing pool.

    Workers fork from a lightweight server process (~10 MB VSZ), not the
    main training process (~400 GB VSZ), avoiding virtual memory overcommit
    failures on large-memory nodes (Committed_AS >> CommitLimit).
    """
    global _forkserver_pool, _forkserver_pool_size
    if _forkserver_pool is None or _forkserver_pool_size != num_workers:
        if _forkserver_pool is not None:
            _forkserver_pool.terminate()
            _forkserver_pool.join()
        ctx = multiprocessing.get_context("forkserver")
        _forkserver_pool = ctx.Pool(processes=num_workers, maxtasksperchild=50)
        _forkserver_pool_size = num_workers
    return _forkserver_pool


def compute_scores_batch(
    reward_inputs: list[dict],
    timeout_per_test: int = 6,
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

    Parallelism is completion-level via a pre-forked multiprocessing.Pool
    (avoids fork()-ing the large training process). Each completion is bounded
    to global_timeout seconds so the tail is capped.

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
        except (SyntaxError, ValueError):
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

    # ── Phase 2: Execute unique completions in forkserver pool ──
    # Workers fork from a lightweight forkserver process (not the 400GB+
    # training process), avoiding virtual memory overcommit failures.
    dedup_results: dict[str, dict] = {}

    if dedup_tasks:
        pool = _get_forkserver_pool(num_workers)
        # Build pool tasks (exclude expected_outputs; matching done in parent)
        pool_tasks = []
        task_meta = {}  # key → (expected_outputs, fn_name)
        for key, task in dedup_tasks.items():
            _, code, inp_strs, exp_strs, fn_name, timeout, mem, gtimeout = task
            pool_tasks.append((key, code, inp_strs, fn_name, timeout, mem, gtimeout))
            task_meta[key] = (exp_strs, fn_name)

        for dedup_key, outputs in pool.map(pool_execute_completion, pool_tasks):
            exp_strs, fn_name = task_meta[dedup_key]
            n = len(exp_strs)
            passed = 0
            for actual, expected in zip(outputs, exp_strs):
                if actual is not None:
                    exp = str(expected) if not isinstance(expected, str) else expected
                    if _outputs_match(actual.strip(), exp.strip(), fn_name):
                        passed += 1
            score = passed / n
            acc = 1.0 if passed == n else 0.0
            dedup_results[dedup_key] = {"score": score, "acc": acc}

    gc.collect()

    # ── Phase 3: Scatter scores back to all completions (including duplicates) ──
    for i, key in completion_to_key.items():
        results[i] = dedup_results.get(key, dict(_ZERO))

    return results
