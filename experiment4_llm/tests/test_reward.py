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
