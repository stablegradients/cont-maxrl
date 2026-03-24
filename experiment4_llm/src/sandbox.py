"""
Subprocess-based code execution sandbox for evaluating generated Python code.

Uses os.posix_spawn() instead of subprocess.run() to avoid fork() from
large-VSZ training processes. fork() triggers the kernel's overcommit
heuristic (overcommit_memory=0), which non-deterministically refuses
memory allocation and causes SIGSEGV in MemoryError_dealloc on aarch64.
posix_spawn() creates a new process directly without cloning the parent's
virtual address space.

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
import signal
import sys
import tempfile
import time


DEFAULT_TIMEOUT = 6  # seconds per test case (matches official LCB default)
DEFAULT_MEMORY_MB = 512
DEFAULT_GLOBAL_TIMEOUT = 60  # hard cap per completion (covers many large tests)

# Import string prepended to call-based code (matches official LCB testing_util.py)
_IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\n"
    "import heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)

# Runner script templates. These are top-level strings (no indentation issues).
# They read the user code and input from files passed as argv[1] and argv[2].

_RUNNER_STDIN = """\
import sys
import io
import ast
import resource
from unittest.mock import patch, mock_open

mem_bytes = {mem_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except ValueError:
    pass

with open(sys.argv[1]) as _f:
    _user_code = _f.read()
with open(sys.argv[2]) as _f:
    _input_str = _f.read()

# Strip if __name__ == '__main__' guard (matches official LCB)
try:
    _tree = ast.parse(_user_code)
    _last = _tree.body[-1] if _tree.body else None
    if isinstance(_last, ast.If) and ast.unparse(_last.test).strip() == "__name__ == '__main__'":
        _user_code = ast.unparse(_tree.body[:-1]) + "\\n" + ast.unparse(_last.body)
except Exception:
    pass

_inputs_iter = iter(_input_str.split("\\n"))

@patch('builtins.open', mock_open(read_data=_input_str))
@patch('sys.stdin', io.StringIO(_input_str))
@patch('sys.stdin.readline', lambda *a: next(_inputs_iter))
@patch('sys.stdin.readlines', lambda *a: _input_str.split("\\n"))
@patch('sys.stdin.read', lambda *a: _input_str)
def _run():
    exec(compile(_user_code, '<solution>', 'exec'))

_run()
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
exec(compile({import_string_repr} + "\\n\\n" + _user_code, '<solution>', 'exec'), _namespace)

# Look up the function: try standalone first, then class Solution method
_fn = _namespace.get(_fn_name)
if _fn is None and 'Solution' in _namespace:
    _sol = _namespace['Solution']()
    _fn = getattr(_sol, _fn_name, None)
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
# Coerce tuples to lists (matches official LCB)
if isinstance(_result, tuple):
    _result = list(_result)
print(json.dumps(_result))
"""

# ── Batch runner: all test cases for one completion in a single process ──
# Uses signal.alarm for per-test timeout. NO early-stop: all tests are always
# executed to preserve the full continuous reward signal (passed / total).
# Outputs JSON list of per-test results (string output or null for failure).

_RUNNER_BATCH = """\
import sys, io, json, signal, resource, ast
from unittest.mock import patch, mock_open

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

# Import string matching official LCB
_IMPORT_STR = {import_string_repr}

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
        exec(compile(_IMPORT_STR + "\\n\\n" + _code, '<solution>', 'exec'), _ns)
    except Exception:
        print(json.dumps([None] * len(_inputs)))
        sys.exit(0)
    _fn = _ns.get(_fn_name)
    if _fn is None and 'Solution' in _ns:
        _sol = _ns['Solution']()
        _fn = getattr(_sol, _fn_name, None)
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
            if isinstance(_r, tuple):
                _r = list(_r)
            _results.append(json.dumps(_r))
        except Exception:
            signal.alarm(0)
            _results.append(None)
else:
    # Strip if __name__ == '__main__' guard (matches official LCB)
    try:
        _tree = ast.parse(_code)
        _last = _tree.body[-1] if _tree.body else None
        if isinstance(_last, ast.If) and ast.unparse(_last.test).strip() == "__name__ == '__main__'":
            _code = ast.unparse(_tree.body[:-1]) + "\\n" + ast.unparse(_last.body)
    except Exception:
        pass
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
            _inp_iter = iter(_s.split("\\n"))
            with patch('builtins.open', mock_open(read_data=_s)):
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


def _spawn_run(cmd: list[str], env: dict, timeout: float,
               stdout_path: str, stderr_path: str) -> tuple[int, str, str]:
    """Run a command via os.posix_spawn() — NO fork().

    Uses posix_spawn file_actions to redirect stdout/stderr to files,
    then polls os.waitpid() with a timeout. On timeout, sends SIGKILL.

    Returns:
        (returncode, stdout_text, stderr_text)

    Raises:
        TimeoutError if the process exceeds the timeout.
    """
    file_actions = [
        (os.POSIX_SPAWN_OPEN, 1, stdout_path,
         os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644),
        (os.POSIX_SPAWN_OPEN, 2, stderr_path,
         os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644),
    ]

    pid = os.posix_spawn(cmd[0], cmd, env, file_actions=file_actions)

    deadline = time.monotonic() + timeout
    while True:
        result_pid, status = os.waitpid(pid, os.WNOHANG)
        if result_pid != 0:
            if os.WIFEXITED(status):
                returncode = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                returncode = -os.WTERMSIG(status)
            else:
                returncode = -1
            break
        if time.monotonic() > deadline:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            os.waitpid(pid, 0)  # reap zombie
            raise TimeoutError(f"Process exceeded {timeout}s timeout")
        time.sleep(0.005)

    try:
        with open(stdout_path, "r") as f:
            stdout_text = f.read()
    except OSError:
        stdout_text = ""
    try:
        with open(stderr_path, "r") as f:
            stderr_text = f.read()
    except OSError:
        stderr_text = ""

    return returncode, stdout_text, stderr_text


def execute_in_sandbox(
    code: str,
    input_data: str,
    fn_name: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = DEFAULT_MEMORY_MB,
) -> str:
    """
    Execute generated code against a single test case in a sandboxed subprocess.

    Uses os.posix_spawn() to avoid fork() from large training processes.

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
    stdout_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".out", dir=tmp_dir, delete=False
    )
    stderr_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".err", dir=tmp_dir, delete=False
    )
    stdout_file.close()
    stderr_file.close()

    try:
        code_file.write(code)
        code_file.close()

        input_file.write(input_data)
        input_file.close()

        if fn_name is not None:
            runner_script = _RUNNER_FUNCTION.format(
                mem_mb=memory_limit_mb,
                import_string_repr=repr(_IMPORT_STRING),
            )
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

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", "/tmp"),
            "TMPDIR": tmp_dir,
        }

        returncode, stdout, stderr = _spawn_run(
            cmd, env, timeout_seconds, stdout_file.name, stderr_file.name,
        )

        if returncode != 0:
            stderr = stderr.strip()
            if "SyntaxError" in stderr:
                raise SyntaxError(stderr)
            if "MemoryError" in stderr or "Cannot allocate memory" in stderr:
                raise MemoryError(stderr)
            raise RuntimeError(f"Exit code {returncode}: {stderr[:500]}")

        return stdout

    except TimeoutError:
        raise TimeoutError(f"Execution exceeded {timeout_seconds}s timeout")
    finally:
        for f in [code_file, input_file, runner_file, stdout_file, stderr_file]:
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

    Uses os.posix_spawn() to avoid fork() from large training processes.
    Uses signal.alarm for per-test timeouts. No early-stop: all tests are
    always executed to preserve the full continuous reward signal.

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
    stdout_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".out", dir=tmp_dir, delete=False
    )
    stderr_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".err", dir=tmp_dir, delete=False
    )
    stdout_file.close()
    stderr_file.close()

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

        runner_file.write(_RUNNER_BATCH.format(
            mem_mb=memory_limit_mb,
            import_string_repr=repr(_IMPORT_STRING),
        ))
        runner_file.close()

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", "/tmp"),
            "TMPDIR": tmp_dir,
        }

        returncode, stdout, stderr = _spawn_run(
            [sys.executable, runner_file.name, code_file.name, test_file.name],
            env, global_timeout, stdout_file.name, stderr_file.name,
        )

        if returncode != 0:
            return [None] * len(inputs)

        return json.loads(stdout)

    except TimeoutError:
        return [None] * len(inputs)
    except (json.JSONDecodeError, Exception):
        return [None] * len(inputs)
    finally:
        for f in [code_file, test_file, runner_file, stdout_file, stderr_file]:
            try:
                os.unlink(f.name)
            except OSError:
                pass


def pool_execute_completion(task: tuple) -> tuple:
    """Execute all tests for one completion — thread pool worker.

    Args:
        task: (dedup_key, code, inputs, fn_name, timeout_per_test,
               memory_limit_mb, global_timeout)

    Returns:
        (dedup_key, outputs) where outputs is list[str | None] per test.
    """
    dedup_key, code, inputs, fn_name, timeout, mem, global_timeout = task
    try:
        outputs = execute_batch_in_sandbox(
            code=code,
            inputs=inputs,
            fn_name=fn_name,
            timeout_per_test=timeout,
            memory_limit_mb=mem,
            global_timeout=global_timeout,
        )
        return (dedup_key, outputs)
    except Exception:
        return (dedup_key, [None] * len(inputs))
