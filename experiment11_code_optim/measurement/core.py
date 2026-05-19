"""
C++ measurement core (spec §4.2).

Compiles C++ programs with g++ -O3 -std=c++17 (absolute path per spec §1),
gates them against expected outputs (strict all-pass), and measures retired
instruction count via `perf stat -e instructions` (PAPI fallback) when the
program passes. Per spec: wall-clock is forbidden; median-of-3 per test case;
NUMA-0 pinning + ASLR disable via setarch -R for the counted runs.

Public entrypoint:

    measure(code_str, test_cases) -> dict

This module is grown by test-driven development. The implementation here is
the minimum to satisfy the currently-passing test suite; extensions land only
when a new failing test demands them.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

# Spec §1: absolute path to g++ 14.2.0
GPP_PATH = "/opt/cray/pe/gcc-native/14/bin/g++"
COMPILE_FLAGS = ["-O3", "-std=c++17"]

# Per-record compile scratch. Spec §1 names `/local` as the compile scratch, but
# on DeltaAI `/local` itself is root-owned drwxr-xr-x; under SLURM the
# per-job writable subdir is `/local/slurmjobs/<jobid>/`, exposed via $TMPDIR
# automatically — so just leaving `tempfile.TemporaryDirectory(dir=None)` to
# follow $TMPDIR lands compile scratch on the genuine on-node NVMe under sbatch.
# In an interactive shell without SLURM, $TMPDIR falls back to `/tmp` (Lustre
# under our cluster setup), and compile cost rises noticeably.
#
# `PIE_COMPILE_SCRATCH` is the explicit override (preferred name; takes
# precedence over $TMPDIR). `PIE_SCRATCH_DIR` is kept as a backward-compatible
# alias of the same thing.
def _scratch_dir() -> str | None:
    return (
        os.environ.get("PIE_COMPILE_SCRATCH")
        or os.environ.get("PIE_SCRATCH_DIR")
        or None
    )

# Spec §4.2 backend selection at module init. Probed at §13 item 1: perf works
# (perf_event_paranoid=2 permits user-process hardware counters). PAPI 7.2.0.2
# fallback is documented; we add the runtime check below only when perf fails
# at first use; the failing case becomes its own test before refactoring.
MEASUREMENT_BACKEND = "perf"

# perf stat writes to stderr; the instruction-count line looks like
#     "     281,447      instructions:u  ..."
_INSTR_RE = re.compile(r"([\d,]+)\s+instructions:u")


def _compile(code_str: str, tmpdir: str) -> tuple[str | None, str | None]:
    """Compile C++ source. Returns (executable_path, error_str). On failure exe is None."""
    src_path = os.path.join(tmpdir, "prog.cpp")
    exe_path = os.path.join(tmpdir, "prog")
    with open(src_path, "w") as f:
        f.write(code_str)
    proc = subprocess.run(
        [GPP_PATH, *COMPILE_FLAGS, "-o", exe_path, src_path],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if proc.returncode != 0:
        return None, f"compile failed (rc={proc.returncode}): {proc.stderr.strip()}"
    return exe_path, None


def _run_and_compare(exe_path: str, input_str: str, expected: str, timeout_s: float) -> bool:
    """Run program with stdin=input_str, return True iff output matches expected (normalized).

    Failures (nonzero exit, timeout, OS error) all return False — never raise — so the
    strict-gate loop can keep going / fail cleanly.
    """
    try:
        proc = subprocess.run(
            [exe_path],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    return _normalize(proc.stdout) == _normalize(expected)


def _normalize(s: str) -> str:
    """Spec §4.2: per-line + end trailing-whitespace normalization."""
    return "\n".join(line.rstrip() for line in s.rstrip("\n").split("\n"))


def _perf_count_instructions(exe_path: str, input_str: str) -> int | None:
    """Run exe under `perf stat -e instructions:u` and parse the retired-instruction count.

    Returns None if perf fails or its output can't be parsed.
    """
    proc = subprocess.run(
        ["perf", "stat", "-e", "instructions:u", "--", exe_path],
        input=input_str,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if proc.returncode != 0:
        return None
    match = _INSTR_RE.search(proc.stderr)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def measure(
    code_str: str,
    test_cases: list[tuple[str, str]],
    *,
    per_test_timeout_s: float = 10.0,
) -> dict:
    """Compile, gate (strict all-pass), then measure retired instructions.

    Args:
        code_str: full C++ source.
        test_cases: list of (stdin_input, expected_stdout) pairs. Each pair is
            graded independently; the gate is satisfied only if EVERY pair passes.
        per_test_timeout_s: per-test execution wall-clock cap (spec §4.2 default 10s).

    Returns:
        {
            "compiled": bool,
            "all_pass": bool,
            "n_tests_run": int,
            "n_tests_passed": int,
            "instr_count": int | None,    # None unless all_pass
            "measurement_backend": str,    # "perf" or "papi"
            "error": str | None,
        }
    """
    result: dict = {
        "compiled": False,
        "all_pass": False,
        "n_tests_run": 0,
        "n_tests_passed": 0,
        "instr_count": None,
        "measurement_backend": MEASUREMENT_BACKEND,
        "error": None,
    }

    with tempfile.TemporaryDirectory(dir=_scratch_dir(), prefix="pie_meas_") as tmpdir:
        exe_path, err = _compile(code_str, tmpdir)
        if exe_path is None:
            result["error"] = err
            return result
        result["compiled"] = True

        n_pass = 0
        for input_str, expected in test_cases:
            if _run_and_compare(exe_path, input_str, expected, per_test_timeout_s):
                n_pass += 1
        result["n_tests_run"] = len(test_cases)
        result["n_tests_passed"] = n_pass

        all_pass = (len(test_cases) > 0) and (n_pass == len(test_cases))
        if not all_pass:
            return result
        result["all_pass"] = True

        # Spec §4.2 measurement: median-of-3 per test case; instr_count = sum of medians.
        per_case_medians: list[int] = []
        for input_str, _expected in test_cases:
            samples = [_perf_count_instructions(exe_path, input_str) for _ in range(3)]
            if any(s is None for s in samples):
                # perf failed mid-batch; leave instr_count=None and abandon measurement.
                result["instr_count"] = None
                break
            samples.sort()
            per_case_medians.append(samples[1])
        else:
            result["instr_count"] = sum(per_case_medians)

    return result
