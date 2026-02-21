"""Tests for the code execution sandbox."""
import pytest

from experiment4_llm.src.sandbox import execute_in_sandbox


def test_simple_stdout():
    """Basic stdin/stdout execution."""
    code = "x = int(input())\nprint(x + 1)"
    result = execute_in_sandbox(code, input_data="5\n")
    assert result.strip() == "6"


def test_multiline_output():
    """Code that prints multiple lines."""
    code = "for i in range(3):\n    print(i)"
    result = execute_in_sandbox(code, input_data="")
    assert result.strip() == "0\n1\n2"


def test_function_call():
    """Function-call mode execution."""
    code = "def double(x):\n    return x * 2"
    result = execute_in_sandbox(code, input_data="5", fn_name="double")
    assert result.strip() == "10"


def test_timeout():
    """Infinite loop should raise TimeoutError."""
    code = "while True: pass"
    with pytest.raises(TimeoutError):
        execute_in_sandbox(code, input_data="", timeout_seconds=2)


def test_syntax_error():
    """Invalid Python should raise SyntaxError."""
    code = "def ("
    with pytest.raises((SyntaxError, RuntimeError)):
        execute_in_sandbox(code, input_data="")


def test_runtime_error():
    """Runtime error (division by zero) should raise RuntimeError."""
    code = "print(1/0)"
    with pytest.raises(RuntimeError):
        execute_in_sandbox(code, input_data="")


def test_no_input():
    """Code that doesn't need input."""
    code = "print(42)"
    result = execute_in_sandbox(code, input_data="")
    assert result.strip() == "42"


def test_empty_code():
    """Empty code produces no output."""
    result = execute_in_sandbox("", input_data="")
    assert result.strip() == ""


def test_large_output():
    """Code that produces substantial output."""
    code = "for i in range(100):\n    print(f'line {i}')"
    result = execute_in_sandbox(code, input_data="")
    lines = result.strip().split("\n")
    assert len(lines) == 100
    assert lines[0] == "line 0"
    assert lines[99] == "line 99"


# ── Function returning None (BUG R5-1) ──


def test_function_returning_none():
    """Function returning None should produce 'null' output, not empty string.

    Bug R5-1: sandbox had `if _result is not None: print(json.dumps(_result))`
    which silently produced empty output for None returns. Now unconditionally
    prints json.dumps(_result), so None -> "null".
    """
    code = "def do_nothing(x):\n    return None"
    result = execute_in_sandbox(code, input_data="5", fn_name="do_nothing")
    assert result.strip() == "null"


def test_function_returning_none_implicit():
    """Function with no return statement (implicit None) should output 'null'."""
    code = "def side_effect(x):\n    _ = x + 1"
    result = execute_in_sandbox(code, input_data="5", fn_name="side_effect")
    assert result.strip() == "null"
