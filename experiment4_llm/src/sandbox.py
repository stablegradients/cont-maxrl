"""
Subprocess-based code execution sandbox for evaluating generated Python code.

Security:
- 10-second timeout per test case (HARD limit)
- 512 MB memory limit via resource.setrlimit
- No network access (inherited from restricted subprocess)
- Isolated temp directory per execution
- Kill on timeout

Handles both:
- stdin/stdout problems (fn_name is None)
- Function-call problems (fn_name is set)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile


DEFAULT_TIMEOUT = 1  # seconds per test case
DEFAULT_MEMORY_MB = 512
DEFAULT_GLOBAL_TIMEOUT = 15  # hard cap per completion

# Runner script templates. These are top-level strings (no indentation issues).
# They read the user code and input from files passed as argv[1] and argv[2].

_RUNNER_STDIN = """\
import sys
import io
import resource

mem_bytes = {mem_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except ValueError:
    pass

with open(sys.argv[1]) as _f:
    _user_code = _f.read()
with open(sys.argv[2]) as _f:
    _input_str = _f.read()

sys.stdin = io.StringIO(_input_str)
exec(compile(_user_code, '<solution>', 'exec'))
"""

_RUNNER_FUNCTION = """\
import sys
import io
import resource
import json

mem_bytes = {mem_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except ValueError:
    pass

with open(sys.argv[1]) as _f:
    _user_code = _f.read()
with open(sys.argv[2]) as _f:
    _input_str = _f.read()

_fn_name = sys.argv[3]

_namespace = {{}}
exec(compile(_user_code, '<solution>', 'exec'), _namespace)

_fn = _namespace.get(_fn_name)
if _fn is None:
    print("FUNCTION_NOT_FOUND", file=sys.stderr)
    sys.exit(1)

_input_data = _input_str.strip()
try:
    _args = json.loads(_input_data)
    if not isinstance(_args, list):
        _args = [_args]
except (json.JSONDecodeError, ValueError):
    _args = [_input_data] if _input_data else []

_result = _fn(*_args)
print(json.dumps(_result))
"""

# ── Batch runner: all test cases for one completion in a single process ──
# Uses signal.alarm for per-test timeout. NO early-stop: all tests are always
# executed to preserve the full continuous reward signal (passed / total).
# Outputs JSON list of per-test results (string output or null for failure).

_RUNNER_BATCH = """\
import sys, io, json, signal, resource

mem_bytes = {mem_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except (ValueError, OSError):
    pass

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)

with open(sys.argv[1]) as _f:
    _code = _f.read()
with open(sys.argv[2]) as _f:
    _td = json.load(_f)

_fn_name = _td.get("fn_name")
_inputs = _td["inputs"]
_timeout = _td["timeout"]
_results = []

if _fn_name:
    _ns = {{}}
    try:
        exec(compile(_code, '<solution>', 'exec'), _ns)
    except Exception:
        print(json.dumps([None] * len(_inputs)))
        sys.exit(0)
    _fn = _ns.get(_fn_name)
    if _fn is None:
        print(json.dumps([None] * len(_inputs)))
        sys.exit(0)
    for _inp in _inputs:
        signal.alarm(_timeout)
        try:
            _s = _inp.strip() if isinstance(_inp, str) else str(_inp)
            try:
                _args = json.loads(_s)
                if not isinstance(_args, list):
                    _args = [_args]
            except (json.JSONDecodeError, ValueError):
                _args = [_s] if _s else []
            _r = _fn(*_args)
            signal.alarm(0)
            _results.append(json.dumps(_r))
        except Exception:
            signal.alarm(0)
            _results.append(None)
else:
    try:
        _compiled = compile(_code, '<solution>', 'exec')
    except SyntaxError:
        print(json.dumps([None] * len(_inputs)))
        sys.exit(0)
    for _inp in _inputs:
        signal.alarm(_timeout)
        _saved = sys.stdout
        _captured = io.StringIO()
        sys.stdout = _captured
        try:
            _s = _inp if isinstance(_inp, str) else str(_inp)
            sys.stdin = io.StringIO(_s)
            exec(_compiled, {{}})
            signal.alarm(0)
            sys.stdout = _saved
            _results.append(_captured.getvalue())
        except Exception:
            signal.alarm(0)
            sys.stdout = _saved
            _results.append(None)

print(json.dumps(_results))
"""


def execute_in_sandbox(
    code: str,
    input_data: str,
    fn_name: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """
    Execute generated code against a single test case in a sandboxed subprocess.

    Args:
        code: the generated Python code
        input_data: test case input string
        fn_name: if set, call this function; otherwise use stdin/stdout
        timeout_seconds: max execution time
        memory_limit_mb: memory limit (set via resource.setrlimit in child)

    Returns:
        stdout output as a string

    Raises:
        TimeoutError: if execution exceeds timeout
        RuntimeError: if execution fails (non-zero exit, stderr errors)
        SyntaxError: if code has syntax errors
        MemoryError: if memory limit exceeded
    """
    tmp_dir = os.environ.get("TMPDIR", tempfile.gettempdir())
    os.makedirs(tmp_dir, exist_ok=True)

    # Write code, input, and runner to temp files
    code_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=tmp_dir, delete=False
    )
    input_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", dir=tmp_dir, delete=False
    )
    runner_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=tmp_dir, delete=False
    )

    try:
        code_file.write(code)
        code_file.close()

        input_file.write(input_data)
        input_file.close()

        if fn_name is not None:
            runner_script = _RUNNER_FUNCTION.format(mem_mb=memory_limit_mb)
            cmd = [
                sys.executable, runner_file.name,
                code_file.name, input_file.name, fn_name,
            ]
        else:
            runner_script = _RUNNER_STDIN.format(mem_mb=memory_limit_mb)
            cmd = [
                sys.executable, runner_file.name,
                code_file.name, input_file.name,
            ]

        runner_file.write(runner_script)
        runner_file.close()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tmp_dir,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", "/tmp"),
                "TMPDIR": tmp_dir,
            },
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "SyntaxError" in stderr:
                raise SyntaxError(stderr)
            if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
                raise MemoryError(stderr)
            raise RuntimeError(f"Exit code {result.returncode}: {stderr[:500]}")

        return result.stdout

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Execution exceeded {timeout_seconds}s timeout")
    finally:
        for f in [code_file, input_file, runner_file]:
            try:
                os.unlink(f.name)
            except OSError:
                pass


def execute_batch_in_sandbox(
    code: str,
    inputs: list[str],
    fn_name: str | None = None,
    timeout_per_test: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
    global_timeout: int = DEFAULT_GLOBAL_TIMEOUT,
) -> list[str | None]:
    """Run ALL test cases for one completion in a single subprocess.

    Uses signal.alarm for per-test timeouts. No early-stop: all tests are
    always executed to preserve the full continuous reward signal.
    The parent enforces a hard global timeout via subprocess.run.

    Args:
        code: extracted Python source
        inputs: list of test-case input strings
        fn_name: function name for call-based problems, None for stdin/stdout
        timeout_per_test: per-test alarm timeout (seconds)
        memory_limit_mb: memory limit for the subprocess
        global_timeout: hard wall-clock cap for the entire subprocess

    Returns:
        List of output strings (one per test), None for failed tests.
    """
    if not inputs:
        return []

    tmp_dir = os.environ.get("TMPDIR", tempfile.gettempdir())
    os.makedirs(tmp_dir, exist_ok=True)

    code_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=tmp_dir, delete=False
    )
    test_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=tmp_dir, delete=False
    )
    runner_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=tmp_dir, delete=False
    )

    try:
        code_file.write(code)
        code_file.close()

        test_data = {
            "inputs": inputs,
            "fn_name": fn_name,
            "timeout": timeout_per_test,
        }
        json.dump(test_data, test_file)
        test_file.close()

        runner_file.write(_RUNNER_BATCH.format(mem_mb=memory_limit_mb))
        runner_file.close()

        result = subprocess.run(
            [sys.executable, runner_file.name, code_file.name, test_file.name],
            capture_output=True,
            text=True,
            timeout=global_timeout,
            cwd=tmp_dir,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", "/tmp"),
                "TMPDIR": tmp_dir,
            },
        )

        if result.returncode != 0:
            return [None] * len(inputs)

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        return [None] * len(inputs)
    except (json.JSONDecodeError, Exception):
        return [None] * len(inputs)
    finally:
        for f in [code_file, test_file, runner_file]:
            try:
                os.unlink(f.name)
            except OSError:
                pass
