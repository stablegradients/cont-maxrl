"""
Regression tests for the sandbox stdout-silencing fix.

The bug: module-level print() in user code corrupts the JSON result channel
(subprocess stdout) for call-based execution, causing json.loads() to fail
and every test case to score 0.

These tests verify that:
- Trailing prints in call-based code do not break scoring
- Functions with internal prints still return correctly
- Clean (no-print) code still works (no regression)
- Wrong code still fails correctly (no false positives)
- LCB stdin/stdout path is unaffected
- MBPP+ assert pattern works
"""
from __future__ import annotations

import json

import pytest

from experiment4_llm.src.reward_function import compute_score
from experiment4_llm.src.sandbox import execute_batch_in_sandbox, execute_in_sandbox


# ── Ground truth fixtures ──

IS_HAPPY_GROUND_TRUTH = json.dumps({
    "inputs": [
        json.dumps(["a"]),
        json.dumps(["aa"]),
        json.dumps(["abcd"]),
        json.dumps(["aabb"]),
        json.dumps(["adb"]),
        json.dumps(["xyy"]),
        json.dumps(["iopaxpoi"]),
    ],
    "outputs": ["false", "false", "true", "false", "true", "false", "true"],
    "fn_name": "is_happy",
})

ADD_GROUND_TRUTH = json.dumps({
    "inputs": [json.dumps([2, 3]), json.dumps([0, 0]), json.dumps([-1, 1])],
    "outputs": ["5", "0", "0"],
    "fn_name": "add",
})


# ── Completions ──

IS_HAPPY_CORRECT_WITH_PRINTS = '''\
def is_happy(s):
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i] == s[i+2] or s[i+1] == s[i+2]:
            return False
    return True

print("DEBUG: module loaded")
print("this should not break anything")
'''

IS_HAPPY_CORRECT_CLEAN = '''\
def is_happy(s):
    if len(s) < 3:
        return False
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i] == s[i+2] or s[i+1] == s[i+2]:
            return False
    return True
'''

IS_HAPPY_WRONG = '''\
def is_happy(s):
    return True
'''

IS_HAPPY_WRONG_WITH_PRINTS = '''\
def is_happy(s):
    return True

print("wrong answer incoming")
'''

ADD_CORRECT_WITH_PRINTS = '''\
def add(a, b):
    return a + b

print("loaded add module")
'''

ADD_WITH_INTERNAL_PRINT = '''\
def add(a, b):
    print(f"adding {a} + {b}")
    return a + b
'''


# ── Test 1: trailing prints do not break call-based scoring ──

def test_trailing_prints_do_not_break_call_based_scoring():
    """is_happy with trailing prints must score 1.0"""
    result = compute_score(IS_HAPPY_CORRECT_WITH_PRINTS, IS_HAPPY_GROUND_TRUTH)
    assert result["score"] == 1.0, f"Expected score 1.0, got {result}"
    assert result["acc"] == 1.0


# ── Test 2: batch sandbox ignores trailing prints ──

def test_batch_sandbox_ignores_trailing_prints():
    """Direct execute_batch_in_sandbox test with trailing prints"""
    gt = json.loads(ADD_GROUND_TRUTH)
    outputs = execute_batch_in_sandbox(
        code=ADD_CORRECT_WITH_PRINTS,
        inputs=gt["inputs"],
        fn_name=gt["fn_name"],
    )
    assert len(outputs) == 3
    for out in outputs:
        assert out is not None, f"Test case returned None (JSON parse failure?): {outputs}"
    assert json.loads(outputs[0]) == 5
    assert json.loads(outputs[1]) == 0
    assert json.loads(outputs[2]) == 0


# ── Test 3: single sandbox ignores trailing prints ──

def test_single_sandbox_ignores_trailing_prints():
    """Direct execute_in_sandbox test with trailing prints"""
    result = execute_in_sandbox(
        code=ADD_CORRECT_WITH_PRINTS,
        input_data=json.dumps([2, 3]),
        fn_name="add",
    )
    assert json.loads(result.strip()) == 5


# ── Test 4: function with internal print still returns correctly ──

def test_function_that_prints_internally_still_passes():
    """Function with print() inside body returns correctly"""
    gt = json.loads(ADD_GROUND_TRUTH)
    outputs = execute_batch_in_sandbox(
        code=ADD_WITH_INTERNAL_PRINT,
        inputs=gt["inputs"],
        fn_name=gt["fn_name"],
    )
    assert len(outputs) == 3
    for out in outputs:
        assert out is not None, f"Internal-print function returned None: {outputs}"
    assert json.loads(outputs[0]) == 5
    assert json.loads(outputs[1]) == 0
    assert json.loads(outputs[2]) == 0


# ── Test 5: clean call-based completion still passes (regression) ──

def test_clean_call_based_completion_still_passes():
    """Happy path: no prints, correct code, score 1.0"""
    result = compute_score(IS_HAPPY_CORRECT_CLEAN, IS_HAPPY_GROUND_TRUTH)
    assert result["score"] == 1.0, f"Expected score 1.0, got {result}"
    assert result["acc"] == 1.0


# ── Test 6: wrong call-based completion fails correctly ──

def test_wrong_call_based_completion_fails_correctly():
    """Wrong function (always True) must NOT score 1.0"""
    result = compute_score(IS_HAPPY_WRONG, IS_HAPPY_GROUND_TRUTH)
    assert result["score"] < 1.0, f"Wrong code should not score 1.0, got {result}"


# ── Test 7: wrong + prints still fails correctly ──

def test_wrong_call_based_with_trailing_prints_fails_correctly():
    """Wrong function with trailing prints must NOT score 1.0"""
    result = compute_score(IS_HAPPY_WRONG_WITH_PRINTS, IS_HAPPY_GROUND_TRUTH)
    assert result["score"] < 1.0, f"Wrong+prints code should not score 1.0, got {result}"


# ── Test 8: LCB stdin/stdout clean solution unchanged ──

def test_lcb_stdin_stdout_clean_solution_unchanged():
    """LCB stdin/stdout path still works (no fn_name)"""
    code = "n = int(input())\nprint(n * 2)"
    gt = json.dumps({
        "inputs": ["3\n", "5\n"],
        "outputs": ["6\n", "10\n"],
    })
    result = compute_score(code, gt)
    assert result["score"] == 1.0, f"LCB stdin/stdout should score 1.0, got {result}"


# ── Test 9: LCB stdin/stdout wrong solution still fails ──

def test_lcb_stdin_stdout_wrong_solution_still_fails():
    """Wrong LCB solution still fails"""
    code = "n = int(input())\nprint(n + 1)"
    gt = json.dumps({
        "inputs": ["3\n", "5\n"],
        "outputs": ["6\n", "10\n"],
    })
    result = compute_score(code, gt)
    assert result["score"] < 1.0, f"Wrong LCB code should not score 1.0, got {result}"


# ── Test 10: MBPP+ style trailing assert passes ──

def test_mbpp_style_trailing_assert_passes():
    """MBPP+ pattern: function def + trailing assert (no fn_name)"""
    code = "def add(a, b):\n    return a + b\nassert add(2, 3) == 5"
    gt = json.dumps({
        "inputs": [""],
        "outputs": [""],
    })
    # stdin/stdout path: code runs with no input, produces no output
    # The assert passes silently, so output is empty => matches expected ""
    result = compute_score(code, gt)
    assert result["score"] == 1.0, f"MBPP assert pattern should score 1.0, got {result}"
