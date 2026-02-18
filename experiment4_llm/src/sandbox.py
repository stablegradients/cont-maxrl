from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: int = 5
    memory_limit_mb: int = 256


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    memory_exceeded: bool


def _limit_resources(memory_limit_mb: int) -> None:
    """Apply hard memory/file/core limits in child process (Linux/macOS)."""
    try:
        import resource

        limit_bytes = max(int(memory_limit_mb), 1) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Prevent large accidental file outputs.
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    except Exception:
        # If limits cannot be set on a platform, continue best-effort.
        return


def _coerce_stdin(input_data: Any) -> str:
    if input_data is None:
        return ""
    if isinstance(input_data, str):
        return input_data
    return str(input_data)


def _build_function_wrapper(code: str, fn_name: str, input_data: Any) -> str:
    fn_literal = repr(fn_name)
    input_literal = repr(input_data)

    return f"""\
{code}

import ast
import json


def _parse_payload(raw):
    if isinstance(raw, (list, tuple, dict, int, float, bool)) or raw is None:
        return raw
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return stripped
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except Exception:
            pass
    return raw


def _call_target(fn_name, payload):
    target = globals()[fn_name]
    if isinstance(payload, dict):
        return target(**payload)
    if isinstance(payload, tuple):
        return target(*payload)
    if isinstance(payload, list):
        return target(*payload)
    if payload is None:
        return target()
    return target(payload)


_payload = _parse_payload({input_literal})
_result = _call_target({fn_literal}, _payload)
if isinstance(_result, (dict, list)):
    print(json.dumps(_result, ensure_ascii=False))
elif isinstance(_result, tuple):
    print(json.dumps(list(_result), ensure_ascii=False))
else:
    print(_result)
"""


def _detect_memory_exceeded(stderr: str, returncode: int, timed_out: bool) -> bool:
    if timed_out:
        return False
    lowered = stderr.lower()
    if "memoryerror" in lowered:
        return True
    if returncode in (-signal.SIGKILL, -signal.SIGSEGV):
        return True
    return False


def execute_python_code(
    source_code: str,
    stdin_data: str = "",
    limits: SandboxLimits | None = None,
) -> ExecutionResult:
    """
    Execute Python source in a constrained subprocess.

    This is a best-effort local sandbox. For production isolation, use Docker/VM
    sandboxing (e.g., sandbox-fusion) and route execution through that service.
    """
    if limits is None:
        limits = SandboxLimits()

    with tempfile.TemporaryDirectory(prefix="exp4_sandbox_") as tmpdir:
        script_path = Path(tmpdir) / "solution.py"
        script_path.write_text(source_code, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        proc = subprocess.Popen(
            [sys.executable, "-I", str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmpdir,
            env=env,
            preexec_fn=lambda: _limit_resources(limits.memory_limit_mb),
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(
                input=stdin_data,
                timeout=max(float(limits.timeout_seconds), 0.1),
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()

        returncode = proc.returncode if proc.returncode is not None else -1
        memory_exceeded = _detect_memory_exceeded(
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
        )
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=timed_out,
            memory_exceeded=memory_exceeded,
        )


def execute_in_sandbox(
    code: str,
    input_data: Any,
    fn_name: str | None,
    timeout_seconds: int = 5,
    memory_limit_mb: int = 256,
) -> ExecutionResult:
    """
    Execute generated code for one APPS test case.

    - fn_name is set: function-call mode
    - fn_name is None: stdin/stdout mode
    """
    limits = SandboxLimits(
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )

    if fn_name:
        wrapped_source = _build_function_wrapper(code=code, fn_name=fn_name, input_data=input_data)
        return execute_python_code(source_code=wrapped_source, stdin_data="", limits=limits)

    stdin_text = _coerce_stdin(input_data)
    return execute_python_code(source_code=code, stdin_data=stdin_text, limits=limits)


def normalize_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, (list, dict)):
        text = json.dumps(output, ensure_ascii=False)
    elif isinstance(output, tuple):
        text = json.dumps(list(output), ensure_ascii=False)
    else:
        text = str(output)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.strip().split("\n")]
    return "\n".join(lines).strip()


def try_parse_scalar(text: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def outputs_match(actual: Any, expected: Any, float_tol: float = 1e-6) -> bool:
    """Robust but conservative output matching for APPS-style answers."""
    if isinstance(expected, (list, tuple)):
        return any(outputs_match(actual, candidate, float_tol=float_tol) for candidate in expected)

    actual_norm = normalize_output(actual)
    expected_norm = normalize_output(expected)

    if actual_norm == expected_norm:
        return True

    if actual_norm.split() == expected_norm.split():
        return True

    actual_num = try_parse_scalar(actual_norm)
    expected_num = try_parse_scalar(expected_norm)
    if actual_num is not None and expected_num is not None:
        return abs(actual_num - expected_num) <= float_tol

    # If both are parseable literals (lists/dicts/tuples), compare structure.
    try:
        actual_obj = ast.literal_eval(actual_norm)
        expected_obj = ast.literal_eval(expected_norm)
        return actual_obj == expected_obj
    except Exception:
        return False
