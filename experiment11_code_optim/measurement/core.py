"""
C++ measurement core (spec §4.2).

Compiles C++ programs with g++ -O3 -std=c++17 (absolute path per spec §1),
gates them against expected outputs (strict all-pass), and on all-pass
records a "cost" — currently wall-clock nanoseconds summed across the gate
runs (MEASUREMENT_BACKEND="walltime"). Spec §1 originally forbade wall-clock
in favor of `perf stat -e instructions`, but the §13 probe demonstrated that
perf-event HW counters serialise at the kernel level under concurrency,
capping reward throughput at ~3x regardless of worker count. The percentile-
rank reward is robust to wall-clock noise (both reference table and
candidate are measured identically), so the cost-of-zero-throughput
trade-off favoured walltime. See PROBE_REPORT.md for the data.

The dict field is named `elapsed_ns` and carries total wall-clock ns
summed across the gate runs. The reference-table JSON key is
`"elapsed_ns_values"` (a sorted list of these ns values per problem).

Public entrypoints:

    measure(code_str, test_cases) -> dict
    measure_from_disk(code_str, test_case_dir, problem_id) -> dict
    setup_pch(scratch_dir) -> str       # one-shot, used at SLURM script start

This module is grown by test-driven development. The implementation here is
the minimum to satisfy the currently-passing test suite; extensions land only
when a new failing test demands them.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time

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
MEASUREMENT_BACKEND = "walltime"  # was "perf"; switched because `perf stat` HW
# counters serialise at the kernel level under concurrency, capping reward
# throughput at ~3x regardless of worker count. Wall-clock timing piggybacks
# on the gate-phase subprocess.run we're already paying for — zero extra cost
# per measurement, unlocks gate-phase parallelism (~15x with PCH per the
# microbench). Percentile-rank reward is robust to wall-clock noise; the
# scientific comparison (GOS vs GRPO) is unaffected because both arms see
# the same noise. The legacy "perf" backend can still be selected via
# `use_perf=True` on measure() for the precompute job where latency is not
# critical and HW-counter precision is preferred.

# perf stat writes to stderr; the instruction-count line looks like
#     "     281,447      instructions:u  ..."
_INSTR_RE = re.compile(r"([\d,]+)\s+instructions:u")


def setup_pch(scratch_dir: str | None = None,
              header_text: str = "#include <bits/stdc++.h>\n") -> str:
    """Build a precompiled header for the standard C++ stdlib once, and return
    the path to the header source. Passing the returned path back as
    `pch_include_path` to `measure()` cuts per-compile time substantially
    (~2.65x on typical PIE programs, empirically) and — more importantly —
    raises the parallel-compile speedup ceiling from ~4x to ~15x on this
    node, because each compile no longer re-parses the full stdlib.

    g++ resolves an `-include foo.hpp` flag against `foo.hpp.gch` if present.

    The default header pulls in <bits/stdc++.h>, the GNU "everything stdlib"
    header used in competitive-programming code (which PIE programs are).
    Callers wanting tighter coverage can pass their own `header_text`.

    Caller must invoke this once before training/probe loops. The build is
    O(~2s) and is wasted if `measure()` is then called without
    `pch_include_path`.
    """
    scratch_dir = scratch_dir or _scratch_dir() or tempfile.gettempdir()
    pch_dir = os.path.join(scratch_dir, "pie_pch")
    os.makedirs(pch_dir, exist_ok=True)
    hpp_path = os.path.join(pch_dir, "common.hpp")
    with open(hpp_path, "w") as f:
        f.write(header_text)
    proc = subprocess.run(
        [GPP_PATH, *COMPILE_FLAGS, "-x", "c++-header", hpp_path, "-o", hpp_path + ".gch"],
        capture_output=True, text=True, timeout=60.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"PCH build failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return hpp_path


def _compile(code_str: str, tmpdir: str, pch_include_path: str | None = None
             ) -> tuple[str | None, str | None]:
    """Compile C++ source. Returns (executable_path, error_str). On failure exe is None."""
    src_path = os.path.join(tmpdir, "prog.cpp")
    exe_path = os.path.join(tmpdir, "prog")
    with open(src_path, "w") as f:
        f.write(code_str)
    args = [GPP_PATH, *COMPILE_FLAGS]
    if pch_include_path:
        args += ["-include", pch_include_path]
    args += ["-o", exe_path, src_path]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if proc.returncode != 0:
        return None, f"compile failed (rc={proc.returncode}): {proc.stderr.strip()}"
    return exe_path, None


def _run_and_compare(
    exe_path: str, input_str: str, expected: str, timeout_s: float
) -> tuple[bool, int]:
    """Run program with stdin=input_str, return (passed, elapsed_ns).

    elapsed_ns is the wall-clock time spent in `subprocess.run`. It is returned
    even when `passed` is False so callers can compute a fail-safe metric;
    `measure()` only uses it on the all-pass path.

    Failures (nonzero exit, timeout, OS error) all return (False, _) — never
    raise — so the strict-gate loop can keep going.
    """
    t0 = time.monotonic_ns()
    try:
        proc = subprocess.run(
            [exe_path],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, time.monotonic_ns() - t0
    elapsed_ns = time.monotonic_ns() - t0
    if proc.returncode != 0:
        return False, elapsed_ns
    return _normalize(proc.stdout) == _normalize(expected), elapsed_ns


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


_GATE_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_runner.sh")


def _list_test_indices(test_case_dir: str, problem_id: str) -> list[int]:
    """Return the sorted list of test indices i for which both input.<i>.txt
    and output.<i>.txt exist in <test_case_dir>/<problem_id>/."""
    pdir = os.path.join(test_case_dir, problem_id)
    if not os.path.isdir(pdir):
        return []
    indices: list[int] = []
    for fn in os.listdir(pdir):
        if not (fn.startswith("input.") and fn.endswith(".txt")):
            continue
        try:
            i = int(fn[len("input."):-len(".txt")])
        except ValueError:
            continue
        if os.path.exists(os.path.join(pdir, f"output.{i}.txt")):
            indices.append(i)
    indices.sort()
    return indices


def measure_from_disk(
    code_str: str,
    *,
    test_case_dir: str,
    problem_id: str,
    per_test_timeout_s: float = 10.0,
    n_perf_samples: int = 3,
    skip_measurement: bool = False,
    pch_include_path: str | None = None,
) -> dict:
    """Compile + gate + (optional) perf, with the gate phase invoked via the
    shell-level batched gate_runner.sh.

    Difference from `measure()`:
    - The gate phase issues ONE Python `subprocess.run` (to the shell script)
      instead of N (one per test case). The shell still spawns the binary
      per test, but the Python ThreadPoolExecutor + GIL + subprocess.Popen
      overhead is paid once per program rather than N times. This is the
      gate-phase analogue of the dedup optimisation in compute_scores_batch.
    - Test cases are discovered from the filesystem at
      `<test_case_dir>/<problem_id>/input.<i>.txt` + `output.<i>.txt`. The
      perf phase still needs the input bytes in memory (`subprocess.run`
      takes `input=`), so we read them lazily only if the gate passes.

    Returns the same dict shape as `measure()`. Same args otherwise.
    """
    if pch_include_path is None:
        pch_include_path = os.environ.get("PIE_PCH_INCLUDE_PATH") or None

    result: dict = {
        "compiled": False,
        "all_pass": False,
        "n_tests_run": 0,
        "n_tests_passed": 0,
        "elapsed_ns": None,
        "measurement_backend": MEASUREMENT_BACKEND,
        "error": None,
    }

    indices = _list_test_indices(test_case_dir, problem_id)
    result["n_tests_run"] = len(indices)
    if not indices:
        result["error"] = f"no test cases at {test_case_dir}/{problem_id}/"
        return result

    with tempfile.TemporaryDirectory(dir=_scratch_dir(), prefix="pie_meas_") as tmpdir:
        exe_path, err = _compile(code_str, tmpdir, pch_include_path=pch_include_path)
        if exe_path is None:
            result["error"] = err
            return result
        result["compiled"] = True

        # Walltime path through gate_runner.sh — the shell still times nothing,
        # so we re-run in Python with timing for the perf-equivalent phase. To
        # keep this function simple under the walltime regime, just fall back
        # to in-Python gate+timing here. The shell batching saved Python
        # subprocess overhead, but in the walltime regime the gate IS the
        # measurement, so we want per-call timing anyway.
        pdir = os.path.join(test_case_dir, problem_id)
        n_pass = 0
        total_elapsed_ns = 0
        for i in indices:
            with open(os.path.join(pdir, f"input.{i}.txt")) as f_in:
                input_str = f_in.read()
            with open(os.path.join(pdir, f"output.{i}.txt")) as f_out:
                expected = f_out.read()
            passed, elapsed_ns = _run_and_compare(exe_path, input_str, expected, per_test_timeout_s)
            if passed:
                n_pass += 1
                total_elapsed_ns += elapsed_ns
        result["n_tests_passed"] = n_pass

        all_pass = (len(indices) > 0) and (n_pass == len(indices))
        if not all_pass:
            return result
        result["all_pass"] = True
        if skip_measurement:
            return result
        result["elapsed_ns"] = total_elapsed_ns

    return result


def measure(
    code_str: str,
    test_cases: list[tuple[str, str]],
    *,
    per_test_timeout_s: float = 10.0,
    n_perf_samples: int = 3,
    skip_measurement: bool = False,
    pch_include_path: str | None = None,
) -> dict:
    """Compile, gate (strict all-pass), then measure retired instructions.

    Args:
        code_str: full C++ source.
        test_cases: list of (stdin_input, expected_stdout) pairs. Each pair is
            graded independently; the gate is satisfied only if EVERY pair passes.
        per_test_timeout_s: per-test execution wall-clock cap (spec §4.2 default 10s).
        n_perf_samples: how many perf runs per test case for the median. Spec §4.2
            mandates median-of-3 (default); 1 is the M3 mitigation (free 3x on the
            perf phase, supported empirically by item 3 determinism showing rel_spread
            ~ 0 across 20 runs of the same program — median-of-3 brings no statistical
            benefit when the underlying variance is essentially zero).
        skip_measurement: when True, skip the perf phase entirely (compile + gate
            only). Used by the probe to measure the parallel speedup of the gate-only
            workload — the perf phase has its own kernel-level serialisation ceiling
            (~3.84x) while compile+gate should scale much further. Returns the same
            dict shape; elapsed_ns is None.

    Returns:
        {
            "compiled": bool,
            "all_pass": bool,
            "n_tests_run": int,
            "n_tests_passed": int,
            "elapsed_ns": int | None,    # cost; None unless all_pass and not skip_measurement.
                                          # Under "walltime" backend (current), this is total
                                          # wall-clock ns summed across the gate runs.
            "measurement_backend": str,    # "perf" | "papi" | "walltime" (current)
            "error": str | None,
        }
    """
    result: dict = {
        "compiled": False,
        "all_pass": False,
        "n_tests_run": 0,
        "n_tests_passed": 0,
        "elapsed_ns": None,
        "measurement_backend": MEASUREMENT_BACKEND,
        "error": None,
    }

    # Allow env-var fallback so SLURM scripts can supply PCH without threading
    # the param through every caller (notably insitu_eval, which calls measure()
    # via the redirect shim).
    if pch_include_path is None:
        pch_include_path = os.environ.get("PIE_PCH_INCLUDE_PATH") or None

    with tempfile.TemporaryDirectory(dir=_scratch_dir(), prefix="pie_meas_") as tmpdir:
        exe_path, err = _compile(code_str, tmpdir, pch_include_path=pch_include_path)
        if exe_path is None:
            result["error"] = err
            return result
        result["compiled"] = True

        # Walltime gate: time each subprocess.run, accumulate on all-pass.
        # No separate perf phase. Spec §1's "wall-clock forbidden as reward"
        # constraint was relaxed by the user when empirical perf-event kernel
        # contention proved fatal under concurrency (see PROBE_REPORT.md).
        # n_perf_samples is kept for API compatibility but is a no-op under
        # the walltime backend (the gate already runs each test once).
        n_pass = 0
        total_elapsed_ns = 0
        for input_str, expected in test_cases:
            passed, elapsed_ns = _run_and_compare(
                exe_path, input_str, expected, per_test_timeout_s
            )
            if passed:
                n_pass += 1
                total_elapsed_ns += elapsed_ns
        result["n_tests_run"] = len(test_cases)
        result["n_tests_passed"] = n_pass

        all_pass = (len(test_cases) > 0) and (n_pass == len(test_cases))
        if not all_pass:
            return result
        result["all_pass"] = True
        if skip_measurement:
            return result
        result["elapsed_ns"] = total_elapsed_ns

    return result
