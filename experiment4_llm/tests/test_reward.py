"""Tests for the continuous reward function."""
import json
import pytest

from experiment4_llm.src.reward_function import (
    compute_score,
    compute_score_binary,
    extract_code,
)


# ── extract_code ──


def test_extract_code_markdown_block():
    text = "Here is my solution:\n```python\ndef foo():\n    return 42\n```\nDone."
    assert extract_code(text) == "def foo():\n    return 42"


def test_extract_code_bare_block():
    text = "```\ndef bar():\n    pass\n```"
    assert extract_code(text) == "def bar():\n    pass"


def test_extract_code_no_block():
    text = "def baz():\n    return 1"
    assert extract_code(text) == "def baz():\n    return 1"


def test_extract_code_empty():
    assert extract_code("") == ""


# ── compute_score (continuous reward) ──


def _make_ground_truth(inputs, outputs, fn_name=None):
    d = {"inputs": inputs, "outputs": outputs}
    if fn_name:
        d["fn_name"] = fn_name
    return json.dumps(d)


def test_score_all_pass_stdin():
    """Code that produces correct output for all test cases -> score 1.0."""
    code = "x = int(input())\nprint(x * 2)"
    gt = _make_ground_truth(["3\n", "5\n"], ["6\n", "10\n"])
    score = compute_score(code, gt)
    assert score == 1.0


def test_score_partial_pass_stdin():
    """Code that passes 1 of 2 test cases -> score 0.5."""
    # This code only works for input 3
    code = "print(6)"
    gt = _make_ground_truth(["3\n", "5\n"], ["6\n", "10\n"])
    score = compute_score(code, gt)
    assert score == 0.5


def test_score_all_fail():
    """Code that produces wrong output -> score 0.0."""
    code = "print('wrong')"
    gt = _make_ground_truth(["3\n"], ["6\n"])
    score = compute_score(code, gt)
    assert score == 0.0


def test_score_syntax_error():
    """Code with syntax error -> score 0.0."""
    code = "def ("  # syntax error
    gt = _make_ground_truth(["3\n"], ["6\n"])
    score = compute_score(code, gt)
    assert score == 0.0


def test_score_timeout():
    """Code that infinite loops -> score 0.0."""
    code = "while True: pass"
    gt = _make_ground_truth(["1\n"], ["1\n"])
    score = compute_score(code, gt, timeout_per_test=2)
    assert score == 0.0


def test_score_function_call():
    """Function-call mode: code defines a function, we call it."""
    code = "def add(a, b):\n    return a + b"
    gt = _make_ground_truth(["[1, 2]", "[3, 4]"], ["3", "7"], fn_name="add")
    score = compute_score(code, gt)
    assert score == 1.0


def test_score_markdown_extraction():
    """Score should extract code from markdown blocks."""
    text = "Here's my solution:\n```python\nx = int(input())\nprint(x * 2)\n```"
    gt = _make_ground_truth(["3\n"], ["6\n"])
    score = compute_score(text, gt)
    assert score == 1.0


def test_score_empty_code():
    """Empty code -> score 0.0."""
    score = compute_score("", _make_ground_truth(["1\n"], ["1\n"]))
    assert score == 0.0


def test_score_invalid_ground_truth():
    """Invalid JSON ground truth -> score 0.0."""
    score = compute_score("print(1)", "not json")
    assert score == 0.0


# ── compute_score_binary ──


def test_binary_all_pass():
    code = "x = int(input())\nprint(x * 2)"
    gt = _make_ground_truth(["3\n", "5\n"], ["6\n", "10\n"])
    assert compute_score_binary(code, gt) == 1.0


def test_binary_partial_pass():
    """Partial pass -> binary 0.0."""
    code = "print(6)"
    gt = _make_ground_truth(["3\n", "5\n"], ["6\n", "10\n"])
    assert compute_score_binary(code, gt) == 0.0


# ── extract_code: case-insensitive & python3 variants (BUG R1-3) ──


def test_extract_code_python3_tag():
    """```python3 should be recognized as a Python code block."""
    text = "```python3\ndef foo():\n    return 42\n```"
    assert extract_code(text) == "def foo():\n    return 42"


def test_extract_code_uppercase_python():
    """```Python (capitalized) should be recognized."""
    text = "```Python\ndef foo():\n    return 42\n```"
    assert extract_code(text) == "def foo():\n    return 42"


def test_extract_code_all_caps_python():
    """```PYTHON should be recognized."""
    text = "```PYTHON\ndef foo():\n    return 42\n```"
    assert extract_code(text) == "def foo():\n    return 42"


def test_extract_code_python3_uppercase():
    """```Python3 should be recognized."""
    text = "```Python3\ndef foo():\n    return 42\n```"
    assert extract_code(text) == "def foo():\n    return 42"


# ── Function call with nested list args (BUG R1-2: str() vs json.dumps()) ──


def test_function_call_with_nested_list_arg():
    """Round-trip: function with a list-of-lists arg serialized via json.dumps.

    Bug R1-2: data_prep used str() instead of json.dumps() for non-string I/O,
    producing Python repr that json.loads() in sandbox would fail on.

    fn_name convention: input is a JSON array of args.
    Single-arg fn taking a matrix needs [matrix] wrapping.
    """
    code = "def sum_nested(matrix):\n    return sum(sum(row) for row in matrix)"
    # One arg: the matrix [[1,2,3],[4,5,6]] wrapped in args list
    test_input = json.dumps([[[1, 2, 3], [4, 5, 6]]])
    test_output = json.dumps(21)
    gt = _make_ground_truth([test_input], [test_output], fn_name="sum_nested")
    score = compute_score(code, gt)
    assert score == 1.0


def test_function_call_with_dict_arg():
    """Function with a dict argument should round-trip via json.dumps."""
    code = "def get_key(d, k):\n    return d[k]"
    test_input = json.dumps([{"a": 1, "b": 2}, "b"])
    test_output = json.dumps(2)
    gt = _make_ground_truth([test_input], [test_output], fn_name="get_key")
    score = compute_score(code, gt)
    assert score == 1.0


def test_function_call_with_bool_return():
    """Boolean returns must be JSON booleans (true/false), not Python (True/False)."""
    code = "def is_even(n):\n    return n % 2 == 0"
    test_input = json.dumps([4])
    test_output = json.dumps(True)  # "true", not "True"
    gt = _make_ground_truth([test_input], [test_output], fn_name="is_even")
    score = compute_score(code, gt)
    assert score == 1.0


# ── Float tolerance (BUG R1-7) ──


def test_score_float_tolerance():
    """Outputs differing only by float precision should match."""
    code = "print(f'{1/3:.6f}')"  # prints "0.333333"
    gt = _make_ground_truth([""], ["0.3333333333"])
    score = compute_score(code, gt)
    assert score == 1.0


def test_numeric_match_multiline():
    """Multi-line numeric output with slight precision differences."""
    from experiment4_llm.src.reward_function import _numeric_match
    assert _numeric_match("3.14159\n2.71828", "3.14159265\n2.71828182")
    assert not _numeric_match("3.14\n2.71", "3.14\n9.99")


def test_numeric_match_zero():
    """0.0 vs 0 should match; 0 vs nonzero should not."""
    from experiment4_llm.src.reward_function import _numeric_match
    assert _numeric_match("0", "0.0")
    assert _numeric_match("0.0", "0.000000")
    assert not _numeric_match("0.0", "0.001")


def test_numeric_match_line_count_mismatch():
    """Different number of lines should not match."""
    from experiment4_llm.src.reward_function import _numeric_match
    assert not _numeric_match("1.0\n2.0", "1.0")
    assert not _numeric_match("1.0", "1.0\n2.0")


# ── _outputs_match: independent JSON parsing & TACO unwrapping (BUG R2-1) ──


def test_outputs_match_one_side_json_other_not():
    """When one side is valid JSON and the other is not, should not crash.

    Bug R2-1: originally both json.loads() calls were in a single try-block,
    so if the first failed, the second was never attempted.
    """
    from experiment4_llm.src.reward_function import _outputs_match

    # actual is valid JSON, expected is not
    assert _outputs_match("[1, 2, 3]", "not json", fn_name="solve") is False
    # actual is not valid JSON, expected is valid JSON
    assert _outputs_match("not json", "[1, 2, 3]", fn_name="solve") is False
    # Both valid JSON, same value
    assert _outputs_match("[1, 2, 3]", "[1, 2, 3]", fn_name="solve") is True


def test_outputs_match_taco_list_wrapping():
    """TACO wraps single return values in [result]. Should unwrap and compare."""
    from experiment4_llm.src.reward_function import _outputs_match

    # Expected is [42], actual is 42
    assert _outputs_match("42", "[42]", fn_name="solve") is True
    # Expected is ["hello"], actual is "hello"
    assert _outputs_match('"hello"', '["hello"]', fn_name="solve") is True
    # Expected is [[1,2]], actual is [1,2]
    assert _outputs_match("[1, 2]", "[[1, 2]]", fn_name="solve") is True
    # Should NOT unwrap multi-element lists
    assert _outputs_match("1", "[1, 2]", fn_name="solve") is False


def test_outputs_match_exact_string_no_fn_name():
    """Without fn_name, only exact string or numeric match is used."""
    from experiment4_llm.src.reward_function import _outputs_match

    assert _outputs_match("hello", "hello", fn_name=None) is True
    assert _outputs_match("hello", "world", fn_name=None) is False
    # Numeric match still works without fn_name
    assert _outputs_match("3.14159", "3.14159265", fn_name=None) is True


def test_outputs_match_json_semantic_equality():
    """JSON semantic comparison: different formatting, same value."""
    from experiment4_llm.src.reward_function import _outputs_match

    # Whitespace differences in JSON
    assert _outputs_match("[1,2,3]", "[1, 2, 3]", fn_name="solve") is True
    # Key order in dicts
    assert _outputs_match('{"a": 1, "b": 2}', '{"b": 2, "a": 1}', fn_name="solve") is True


# ── Function returning None: reward pipeline (BUG R5-1) ──


def test_score_function_returning_none():
    """Function returning None should match expected 'null' output.

    Bug R5-1: sandbox previously suppressed None output, so any function
    returning None silently scored 0. With the fix, json.dumps(None) = "null"
    is printed and matched against expected "null".
    """
    code = "def get_none():\n    return None"
    gt = _make_ground_truth(["[]"], ["null"], fn_name="get_none")
    score = compute_score(code, gt)
    assert score == 1.0


def test_score_function_returning_none_wrong_expected():
    """Function returning None should NOT match non-null expected output."""
    code = "def get_none():\n    return None"
    gt = _make_ground_truth(["[]"], ["42"], fn_name="get_none")
    score = compute_score(code, gt)
    assert score == 0.0


# ── extract_code: python3 boundary check (BUG R5-2) ──


def test_extract_code_python3_no_false_strip():
    """```python3xyz should NOT strip the '3' — 'python3' must end at word boundary.

    Bug R5-2: extract_code blindly stripped '3' after 'python', so
    '```python3xyz' would become 'xyz...' instead of '3xyz...'.
    """
    # python3some_code should NOT be treated as python3 + some_code
    text = "```python3some_code_here\ndef foo():\n    return 42\n```"
    result = extract_code(text)
    # Should NOT strip "3" because next char is 's', not whitespace/newline
    # The "3some_code_here" stays as part of the first line of the block
    assert "3some_code_here" in result or "def foo():" in result


def test_extract_code_python3_with_newline():
    """```python3\\n should strip '3' — newline is a valid boundary."""
    text = "```python3\ndef foo():\n    return 42\n```"
    assert extract_code(text) == "def foo():\n    return 42"


def test_extract_code_python3_with_space():
    """```python3 \\n should strip '3' — space is a valid boundary."""
    text = "```python3 \ndef foo():\n    return 42\n```"
    assert extract_code(text) == "def foo():\n    return 42"
