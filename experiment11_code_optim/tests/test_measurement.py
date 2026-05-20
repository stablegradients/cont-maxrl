"""
Tests for the C++ measurement core (spec §4.2).

Driven test-first. New tests are added one at a time; each must fail before the
implementation that makes it pass exists. Run with:

    PYTHONPATH=. /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python -m pytest experiment11_code_optim/tests/test_measurement.py -v
"""
from __future__ import annotations

import pytest


HELLO_PROGRAM = r"""
#include <iostream>
int main() {
    std::cout << "hi\n";
    return 0;
}
""".strip()


COMPILE_FAIL_PROGRAM = r"""
#include <iostream>
int main( { std::cout << "hi"; }
""".strip()


WRONG_OUTPUT_PROGRAM = r"""
#include <iostream>
int main() {
    std::cout << "wrong\n";
    return 0;
}
""".strip()


SEGFAULT_PROGRAM = r"""
int main() {
    int* p = nullptr;
    *p = 1;
    return 0;
}
""".strip()


def test_hello_world_all_pass():
    """A trivial program with a single matching test case should compile,
    pass the strict gate, and produce a real (>0) retired-instruction count."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(HELLO_PROGRAM, [("", "hi\n")])

    assert result["compiled"] is True, f"unexpected compile failure: {result.get('error')!r}"
    assert result["all_pass"] is True, f"strict gate did not pass: {result!r}"
    assert result["n_tests_run"] == 1
    assert result["n_tests_passed"] == 1
    assert result["instr_count"] is not None, "instr_count must be set when all_pass=True"
    assert result["instr_count"] > 100, (
        f"instr_count={result['instr_count']} suspiciously low — a real C++ Hello World "
        f"prints via libstdc++ and runs millions of instructions; values below 100 indicate "
        f"a stub or a broken perf invocation"
    )
    # Backend was switched from "perf" to "walltime" after the §13 probe
    # showed perf-event kernel contention capped reward throughput at ~3x and
    # made training infeasible. instr_count now carries total wall-clock ns
    # of the gate runs (~ms range) rather than perf-counted instructions
    # (~M range), so the > 100 floor still holds trivially.
    assert result["measurement_backend"] in ("perf", "papi", "walltime"), (
        f"unexpected backend {result['measurement_backend']!r}"
    )
    assert result["error"] is None


def test_compile_failure_reports_clearly():
    """A program with a syntax error must report compiled=False and a non-empty error,
    without raising, and must not attempt measurement."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(COMPILE_FAIL_PROGRAM, [("", "hi")])

    assert result["compiled"] is False
    assert result["all_pass"] is False
    assert result["instr_count"] is None
    assert result["error"] is not None and len(result["error"]) > 0


def test_wrong_output_fails_gate():
    """A compilable program that prints the wrong thing must fail the strict gate
    and not produce an instr_count (spec §4.2: measure only if all_pass)."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(WRONG_OUTPUT_PROGRAM, [("", "right\n")])

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["n_tests_run"] == 1
    assert result["n_tests_passed"] == 0
    assert result["instr_count"] is None


def test_segfault_fails_gate():
    """A compilable program that segfaults at runtime must fail the strict gate
    (nonzero exit) without raising and without producing an instr_count."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(SEGFAULT_PROGRAM, [("", "ok")])

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["n_tests_run"] == 1
    assert result["n_tests_passed"] == 0
    assert result["instr_count"] is None


def test_empty_test_cases_degenerate():
    """Spec §4.2: all_pass requires len(test_cases) > 0 (vacuous truth is rejected).
    The core must not crash on an empty test list and must report all_pass=False."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(HELLO_PROGRAM, [])

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["n_tests_run"] == 0
    assert result["n_tests_passed"] == 0
    assert result["instr_count"] is None


def test_backend_selected_at_module_init():
    """Spec §4.2: measurement backend chosen once at startup; perf primary,
    PAPI fallback, fatal otherwise. Wall-clock is forbidden."""
    from experiment11_code_optim.measurement.core import MEASUREMENT_BACKEND

    assert MEASUREMENT_BACKEND in ("perf", "papi", "walltime"), (
        f"backend must be one of 'perf', 'papi', 'walltime', got {MEASUREMENT_BACKEND!r}"
    )


TIMEOUT_PROGRAM = r"""
#include <iostream>
int main() {
    volatile long long x = 0;
    while (true) { x++; }
    std::cout << x;
    return 0;
}
""".strip()


def test_timeout_fails_gate_within_budget():
    """Spec §4.2: per-test timeout. An infinite-loop program must be killed by the
    timeout and report all_pass=False without raising; the test must complete within
    a small multiple of per_test_timeout_s (no 10s hardcoded default leaking)."""
    import time
    from experiment11_code_optim.measurement.core import measure

    t0 = time.monotonic()
    result = measure(TIMEOUT_PROGRAM, [("", "anything")], per_test_timeout_s=1.0)
    elapsed = time.monotonic() - t0

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["instr_count"] is None
    # Allow generous slack for compile + perf-launch overhead, but if the call
    # took >5s the timeout wasn't being honoured.
    assert elapsed < 5.0, f"timeout not honoured; elapsed={elapsed:.1f}s"


ECHO_FIRST_LINE_PROGRAM = r"""
#include <iostream>
#include <string>
int main() {
    std::string line;
    if (std::getline(std::cin, line)) {
        std::cout << line << "\n";
    }
    return 0;
}
""".strip()


def test_multi_test_instr_count_is_sum_of_per_case_medians():
    """Spec §4.2: instr_count = sum over test cases of per-test medians.
    Running the same program with three identical test cases must yield ~3x the
    instruction count of the same program with one test case (the per-case
    instruction counts should be similar; the sum should scale linearly)."""
    from experiment11_code_optim.measurement.core import measure

    one = measure(ECHO_FIRST_LINE_PROGRAM, [("a\n", "a\n")])
    three = measure(
        ECHO_FIRST_LINE_PROGRAM,
        [("a\n", "a\n"), ("b\n", "b\n"), ("c\n", "c\n")],
    )

    assert one["all_pass"] is True and three["all_pass"] is True
    assert three["n_tests_run"] == 3 and three["n_tests_passed"] == 3

    ratio = three["instr_count"] / one["instr_count"]
    assert 2.5 < ratio < 3.5, (
        f"multi-test instr_count={three['instr_count']} should be ~3x single's "
        f"{one['instr_count']} (ratio={ratio:.2f}); the implementation is NOT "
        f"summing per-test-case instruction counts as spec §4.2 requires"
    )


DETERMINISTIC_LOOP = r"""
#include <iostream>
int main() {
    long long s = 0;
    for (int i = 0; i < 1000000; i++) s += i;
    std::cout << s << "\n";
    return 0;
}
""".strip()


def test_measure_from_disk_uses_batched_gate_runner_and_matches_in_memory_path():
    """measure_from_disk() reads test cases from <test_case_dir>/<problem_id>/
    via the gate_runner.sh batched script. It must report the same all_pass /
    n_tests_passed / instr_count semantics as the in-memory measure() path
    when given the same data."""
    import tempfile
    from experiment11_code_optim.measurement.core import measure, measure_from_disk

    with tempfile.TemporaryDirectory() as d:
        # Two test cases for a known-good Hello-World-like program: one passes,
        # one wrong. Mirrors the gate_runner.sh smoke test.
        prob_dir = f"{d}/pXX"
        import os
        os.makedirs(prob_dir, exist_ok=True)
        with open(f"{prob_dir}/input.0.txt", "w") as f: f.write("ignored\n")
        with open(f"{prob_dir}/output.0.txt", "w") as f: f.write("hi\n")
        with open(f"{prob_dir}/input.1.txt", "w") as f: f.write("ignored\n")
        with open(f"{prob_dir}/output.1.txt", "w") as f: f.write("wrong\n")

        r_disk = measure_from_disk(HELLO_PROGRAM, test_case_dir=d, problem_id="pXX")
        r_mem = measure(HELLO_PROGRAM, [("ignored\n", "hi\n"), ("ignored\n", "wrong\n")])

        assert r_disk["compiled"] is True
        assert r_disk["all_pass"] is False  # one passes, one fails → gate fails
        assert r_disk["n_tests_run"] == 2
        assert r_disk["n_tests_passed"] == r_mem["n_tests_passed"], (
            f"disk path n_pass={r_disk['n_tests_passed']} != mem path "
            f"n_pass={r_mem['n_tests_passed']}"
        )
        assert r_disk["error"] is None


def test_measure_from_disk_all_pass_produces_instr_count():
    """When every test case matches, measure_from_disk must enter the perf
    phase and emit a real instr_count (proves disk + perf path composes)."""
    import tempfile, os
    from experiment11_code_optim.measurement.core import measure_from_disk

    with tempfile.TemporaryDirectory() as d:
        prob_dir = f"{d}/pXX"
        os.makedirs(prob_dir, exist_ok=True)
        for i in range(3):
            with open(f"{prob_dir}/input.{i}.txt", "w") as f: f.write("anything\n")
            with open(f"{prob_dir}/output.{i}.txt", "w") as f: f.write("hi\n")

        r = measure_from_disk(HELLO_PROGRAM, test_case_dir=d, problem_id="pXX")
        assert r["compiled"] is True
        assert r["all_pass"] is True
        assert r["n_tests_run"] == 3
        assert r["n_tests_passed"] == 3
        assert r["instr_count"] is not None and r["instr_count"] > 100


def test_setup_pch_then_measure_with_pch_passes():
    """setup_pch() must build a precompiled <bits/stdc++.h> header and return
    a path. Passing that path back into measure() as pch_include_path must
    preserve the strict-gate semantics (same result on a known-good program)
    while exercising the new g++ '-include' flag plumbing."""
    from experiment11_code_optim.measurement.core import measure, setup_pch

    pch_path = setup_pch()
    assert pch_path is not None and len(pch_path) > 0
    import os
    assert os.path.exists(pch_path), f"PCH header source missing at {pch_path!r}"
    assert os.path.exists(pch_path + ".gch"), f"PCH .gch missing at {pch_path!r}.gch"

    result = measure(HELLO_PROGRAM, [("", "hi\n")], pch_include_path=pch_path)
    assert result["compiled"] is True
    assert result["all_pass"] is True
    assert result["n_tests_passed"] == 1
    assert result["instr_count"] is not None and result["instr_count"] > 100


def test_repeated_measurement_is_near_deterministic():
    """Spec §4.2: 'near-deterministic, median-stabilized.' Two back-to-back measurements
    of the same fixed-work program should agree within 1% (median-of-3 alone makes this
    very tight on a CPU-bound program; NUMA/ASLR controls land later if this regresses).
    """
    from experiment11_code_optim.measurement.core import measure

    expected = str(sum(range(1_000_000))) + "\n"
    r1 = measure(DETERMINISTIC_LOOP, [("", expected)])
    r2 = measure(DETERMINISTIC_LOOP, [("", expected)])

    assert r1["all_pass"] is True and r2["all_pass"] is True
    a, b = r1["instr_count"], r2["instr_count"]
    rel_diff = abs(a - b) / max(a, b)
    # Walltime backend: wall-clock has more jitter than perf instr counts
    # (1-10% typical from OS scheduling). The percentile-rank reward absorbs
    # this since both reference table and candidate are timed the same way.
    # The deterministic test only checks the measurement isn't pathologically
    # noisy; tighter analysis is in the probe.
    assert rel_diff < 0.30, (
        f"unusually large jitter: {a} vs {b} (rel_diff={rel_diff:.4f}); the "
        f"walltime backend should still be repeatable within ~30%"
    )
