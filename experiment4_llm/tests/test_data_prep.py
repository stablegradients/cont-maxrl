"""Tests for data preparation: serialization round-trips and input parsing.

Covers bugs:
  R1-2: str() vs json.dumps() for non-string I/O serialization
  R3-1: LiveCodeBench functional input parsing (_parse_lcb_functional_input)
"""
import json
import pytest

from experiment4_llm.src.data_prep import parse_test_cases


# ── parse_test_cases: non-string I/O serialization (BUG R1-2) ──


def test_parse_test_cases_nonstring_inputs_are_valid_json():
    """Non-string inputs must be serialized with json.dumps, not str().

    str([[1,2]]) produces "[[1, 2]]" which happens to be valid JSON for simple
    lists, but str(True) produces "True" (not "true") and str({'a':1}) produces
    single quotes — both invalid JSON.
    """
    raw = json.dumps({
        "inputs": [[[1, 2, 3]], True, {"key": "val"}],
        "outputs": [[6], True, "result"],
        "fn_name": "solve",
    })
    result = parse_test_cases(raw)
    assert result is not None

    # Each non-string input must be valid JSON
    for i, inp in enumerate(result["inputs"]):
        try:
            json.loads(inp)
        except json.JSONDecodeError:
            pytest.fail(f"Input {i} is not valid JSON: {inp!r}")

    # Specifically, True must become "true" (JSON) not "True" (Python str)
    assert result["inputs"][1] == "true"
    # Dict must use double quotes
    assert '"key"' in result["inputs"][2]


def test_parse_test_cases_nonstring_outputs_are_valid_json():
    """Non-string outputs must also be valid JSON."""
    raw = json.dumps({
        "inputs": ["1", "2"],
        "outputs": [[1, 2, 3], False],
    })
    result = parse_test_cases(raw)
    assert result is not None

    for i, out in enumerate(result["outputs"]):
        try:
            json.loads(out)
        except json.JSONDecodeError:
            pytest.fail(f"Output {i} is not valid JSON: {out!r}")

    assert result["outputs"][1] == "false"


def test_parse_test_cases_string_inputs_unchanged():
    """String inputs should be left as-is (not double-encoded)."""
    raw = json.dumps({
        "inputs": ["hello world\n", "42\n"],
        "outputs": ["HELLO WORLD\n", "42\n"],
    })
    result = parse_test_cases(raw)
    assert result is not None
    assert result["inputs"][0] == "hello world\n"
    assert result["inputs"][1] == "42\n"


def test_parse_test_cases_fn_name_preserved():
    """fn_name should be preserved in output."""
    raw = json.dumps({
        "inputs": ["[1]"],
        "outputs": ["1"],
        "fn_name": "my_func",
    })
    result = parse_test_cases(raw)
    assert result is not None
    assert result["fn_name"] == "my_func"


def test_parse_test_cases_no_fn_name():
    """Without fn_name, result should not contain fn_name key."""
    raw = json.dumps({
        "inputs": ["1\n"],
        "outputs": ["2\n"],
    })
    result = parse_test_cases(raw)
    assert result is not None
    assert "fn_name" not in result


def test_parse_test_cases_invalid_json():
    """Invalid JSON should return None, not crash."""
    assert parse_test_cases("not json") is None
    assert parse_test_cases("") is None
    assert parse_test_cases(None) is None


def test_parse_test_cases_empty_inputs():
    """Empty inputs or outputs should return None."""
    assert parse_test_cases(json.dumps({"inputs": [], "outputs": []})) is None
    assert parse_test_cases(json.dumps({"inputs": ["1"], "outputs": []})) is None


def test_parse_test_cases_mismatched_lengths():
    """Mismatched input/output lengths should return None."""
    raw = json.dumps({"inputs": ["1", "2"], "outputs": ["3"]})
    assert parse_test_cases(raw) is None


# ── _parse_lcb_functional_input (BUG R3-1) ──


def test_lcb_input_string_and_list():
    """Two-arg functional input: string + list, newline-separated."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '"leetscode"\n["leet", "code"]'
    result = _parse_lcb_functional_input(raw)
    assert result == ["leetscode", ["leet", "code"]]


def test_lcb_input_list_and_int():
    """Two-arg: list + integer."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '[12, 9]\n1'
    result = _parse_lcb_functional_input(raw)
    assert result == [[12, 9], 1]


def test_lcb_input_nested_list():
    """Single-arg: nested list."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '[[7, 2, 1], [6, 4, 2]]'
    result = _parse_lcb_functional_input(raw)
    assert result == [[[7, 2, 1], [6, 4, 2]]]


def test_lcb_input_single_string():
    """Single-arg: just a JSON string."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '"hello world"'
    result = _parse_lcb_functional_input(raw)
    assert result == ["hello world"]


def test_lcb_input_empty():
    """Empty string should return empty list."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    assert _parse_lcb_functional_input("") == []
    assert _parse_lcb_functional_input("  \n  ") == []


def test_lcb_input_python_literal_fallback():
    """Python True (not valid JSON) should be parsed via ast.literal_eval."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    # JSON "true" → Python True
    assert _parse_lcb_functional_input("true") == [True]
    # Python "True" (not valid JSON) → fallback to ast.literal_eval
    assert _parse_lcb_functional_input("True") == [True]


def test_lcb_input_three_args():
    """Three arguments: string, int, list."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '"abc"\n42\n[1, 2, 3]'
    result = _parse_lcb_functional_input(raw)
    assert result == ["abc", 42, [1, 2, 3]]


# ── _json_encode_value ──


def test_json_encode_value_already_json_string():
    """String that is already valid JSON should be returned as-is."""
    from experiment4_llm.src.data_prep import _json_encode_value
    assert _json_encode_value('"hello"') == '"hello"'
    assert _json_encode_value("[1, 2, 3]") == "[1, 2, 3]"


def test_json_encode_value_bare_string():
    """Bare string (not JSON-encoded) should be JSON-encoded."""
    from experiment4_llm.src.data_prep import _json_encode_value
    result = _json_encode_value("hello")
    assert result == '"hello"'


def test_json_encode_value_non_string():
    """Non-string values should be json.dumps'd."""
    from experiment4_llm.src.data_prep import _json_encode_value
    assert _json_encode_value(42) == "42"
    assert _json_encode_value(True) == "true"
    assert _json_encode_value([1, 2]) == "[1, 2]"
    assert _json_encode_value({"a": 1}) == '{"a": 1}'


# ── End-to-end: data_prep serialization -> reward function ──


def test_end_to_end_fn_name_round_trip():
    """Full pipeline: parse_test_cases output -> compute_score -> correct result.

    This is the integration test that would have caught BUG R1-2.
    Uses bool return values -- with str(), True becomes "True" (wrong),
    while json.dumps(True) produces "true" (correct JSON).
    """
    from experiment4_llm.src.reward_function import compute_score

    # Simulate raw APPS/TACO format data with non-string I/O.
    # inputs[i] = list of args for test case i, outputs[i] = expected return value
    raw_io = json.dumps({
        "inputs": [[4], [3]],     # two test cases, each with one int arg
        "outputs": [True, False],  # bool return values (critical for str vs json.dumps)
        "fn_name": "is_even",
    })
    parsed = parse_test_cases(raw_io)
    assert parsed is not None

    # Verify serialization is correct: json.dumps(True)="true", NOT str(True)="True"
    assert parsed["outputs"][0] == "true"
    assert parsed["outputs"][1] == "false"

    ground_truth = json.dumps(parsed)
    code = "def is_even(n):\n    return n % 2 == 0"
    score = compute_score(code, ground_truth)
    assert score == 1.0, f"End-to-end round trip failed: score={score}"


def test_end_to_end_nested_list_round_trip():
    """Pipeline with nested list args and list return value."""
    from experiment4_llm.src.reward_function import compute_score

    # fn takes two list args: inputs[i] = [arg1, arg2]
    raw_io = json.dumps({
        "inputs": [[[1, 2], [3, 4]]],    # one test case: add_lists([1,2], [3,4])
        "outputs": [[4, 6]],              # return value (non-string list)
        "fn_name": "add_lists",
    })
    parsed = parse_test_cases(raw_io)
    assert parsed is not None

    ground_truth = json.dumps(parsed)
    code = "def add_lists(a, b):\n    return [x+y for x,y in zip(a, b)]"
    score = compute_score(code, ground_truth)
    assert score == 1.0, f"Nested list round trip failed: score={score}"
