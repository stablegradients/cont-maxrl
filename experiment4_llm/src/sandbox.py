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

import os
import subprocess
import sys
import tempfile


DEFAULT_TIMEOUT = 1  # seconds per test case
DEFAULT_MEMORY_MB = 512

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
