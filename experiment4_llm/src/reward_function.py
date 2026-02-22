"""
Continuous reward function for code generation: fraction of test cases passed.

Training reward:  r = (# test cases passed) / (# total test cases) in [0, 1]
This is NOT the standard binary reward. A solution passing 7/10 tests gets r=0.7.

verl reward function interface:
    compute_score(solution_str, ground_truth, ...) -> float
"""
from __future__ import annotations

import json

from experiment4_llm.src.sandbox import execute_in_sandbox


def extract_code(solution_str: str) -> str:
    """
    Extract Python code from a model completion.
    Handles markdown code blocks and raw code.
    """
    solution_str = solution_str.strip()

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
    timeout_per_test: int = 1,
    memory_limit_mb: int = 512,
    data_source: str | None = None,
    extra_info: dict | None = None,
) -> float:
    """
    Compute continuous reward: fraction of test cases passed.

    Args:
        solution_str: the model's generated code (may include markdown)
        ground_truth: JSON string with test cases:
            {"inputs": [...], "outputs": [...], "fn_name": optional}
        timeout_per_test: seconds per test case
        memory_limit_mb: memory limit per test case

    Returns:
        float in [0, 1]: fraction of test cases passed
    """
    # Parse ground truth
    try:
        test_cases = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    inputs = test_cases.get("inputs", [])
    expected_outputs = test_cases.get("outputs", [])
    fn_name = test_cases.get("fn_name", None)

    if not inputs or not expected_outputs:
        return 0.0

    num_total = min(len(inputs), len(expected_outputs))
    if num_total == 0:
        return 0.0

    # Extract code from model output
    code = extract_code(solution_str)
    if not code:
        return 0.0

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

    return num_passed / num_total


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
    timeout_per_test: int = 1,
    memory_limit_mb: int = 512,
) -> float:
    """
    Binary reward: 1.0 if ALL test cases pass, 0.0 otherwise.
    Used for evaluation (pass@k), not training.
    """
    score = compute_score(
        solution_str, ground_truth, timeout_per_test, memory_limit_mb
    )
    return 1.0 if score == 1.0 else 0.0
