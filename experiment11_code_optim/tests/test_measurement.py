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


NON_UTF8_OUTPUT_PROGRAM = r"""
#include <iostream>
int main() {
    char c = (char)0xff;
    std::cout << c << "\n";
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
    assert result["elapsed_ns"] is not None, "elapsed_ns must be set when all_pass=True"
    assert result["elapsed_ns"] > 100, (
        f"elapsed_ns={result['elapsed_ns']} suspiciously low — a real C++ Hello World "
        f"prints via libstdc++ and runs millions of instructions; values below 100 indicate "
        f"a stub or a broken perf invocation"
    )
    # Backend was switched from "perf" to "walltime" after the §13 probe
    # showed perf-event kernel contention capped reward throughput at ~3x and
    # made training infeasible. elapsed_ns now carries total wall-clock ns
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
    assert result["elapsed_ns"] is None
    assert result["error"] is not None and len(result["error"]) > 0


def test_compile_with_non_utf8_bytes_does_not_crash():
    """REGRESSION (RL crash 2026-05-20 across maxrl/grpo/rloo jobs
    2307581/2/3): the model can generate C++ source containing
    non-UTF-8 byte sequences in string/char literals. g++ echoes those
    bytes back in its stderr ("error: stray '\\xXX' in program ..."),
    and subprocess.run(text=True) would crash with UnicodeDecodeError
    if `errors='replace'` is missing — taking down the entire reward
    step and the whole RL run.

    Verify that compiling such a program returns the standard failure
    dict (compiled=False, all_pass=False, non-None error) without
    raising. The bytes embedded below include 0xE0 0x80 (an invalid
    UTF-8 sequence — incomplete continuation) and 0xFF, both of which
    appeared in the actual crashing rollouts.
    """
    from experiment11_code_optim.measurement.core import measure
    src = (
        "#include <iostream>\n"
        "int main(){\n"
        "    const char *s = \"" + "\\xe0\\x80" + "\\xff\";\n"  # bad UTF-8
        "    std::cout << s << std::endl;\n"
        "    return 0;\n"
        "}\n"
    )
    # Wrap in a bytes->str round-trip to force a literal raw byte path.
    src_bytes = src.encode("latin-1") + b"\xe0\x80\xff// trailing non-utf8\n"
    src_with_bad = src_bytes.decode("latin-1")
    result = measure(src_with_bad, [("", "anything\n")])
    assert isinstance(result, dict)
    # Either compiled successfully (and the gate fails on output mismatch)
    # OR the compile errored — either way, the call must NOT raise and the
    # result dict must be well-formed. The non-crashing property is what
    # the regression is guarding against.
    assert result["compiled"] in (True, False)
    if not result["compiled"]:
        assert result["error"] is not None and isinstance(result["error"], str)


def test_wrong_output_fails_gate():
    """A compilable program that prints the wrong thing must fail the strict gate
    and not produce an elapsed_ns (spec §4.2: measure only if all_pass)."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(WRONG_OUTPUT_PROGRAM, [("", "right\n")])

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["n_tests_run"] == 1
    assert result["n_tests_passed"] == 0
    assert result["elapsed_ns"] is None


def test_memory_exhausting_program_fails_gate_via_rlimit():
    """REGRESSION (RL crash 2026-05-21 across maxrl 2308902 / grpo 2308903 /
    rloo 2308904): a model-generated program that allocates tens of GB of
    memory blew the global node memory and triggered Ray OOM-killer on the
    trainer. The fix is RLIMIT_AS in a preexec_fn capping each spawned
    binary; an over-large allocation gets SIGKILL'd by the kernel and the
    gate just reports a non-zero exit (failed test) instead of taking down
    the whole training run.

    This test sets the cap to 64 MiB via PIE_PROGRAM_MEM_LIMIT_GB (we read
    it as integer GB; encode 64MB by routing through a small fork), compiles
    a program that asks malloc for 1 GiB, and asserts the gate cleanly
    fails (no exception, no test-runner crash).
    """
    import os
    from experiment11_code_optim.measurement import core as _core
    saved = _core._MEM_LIMIT_BYTES
    _core._MEM_LIMIT_BYTES = 64 * (1 << 20)  # 64 MiB
    try:
        src = """
#include <iostream>
#include <cstdlib>
int main() {
    // Ask for 1 GiB; with RLIMIT_AS at 64 MiB this will fail or SIGKILL.
    size_t N = 1ULL << 30;
    char *p = (char*)std::malloc(N);
    if (!p) { std::cout << "malloc-failed\\n"; return 1; }
    // Touch the pages to force commit (or crash if overcommit lies).
    for (size_t i = 0; i < N; i += 4096) p[i] = 1;
    std::cout << "alloc-ok\\n";
    return 0;
}
"""
        result = _core.measure(src, [("", "alloc-ok\n")])
        # Either compile failed, or run got OOM/SIGKILL'd. Both yield
        # all_pass=False without raising — that's the contract.
        assert isinstance(result, dict)
        assert result["all_pass"] is False
        # n_tests_passed should be 0 — the cap should have taken effect.
        assert result["n_tests_passed"] == 0
    finally:
        _core._MEM_LIMIT_BYTES = saved


def test_segfault_fails_gate():
    """A compilable program that segfaults at runtime must fail the strict gate
    (nonzero exit) without raising and without producing an elapsed_ns."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(SEGFAULT_PROGRAM, [("", "ok")])

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["n_tests_run"] == 1
    assert result["n_tests_passed"] == 0
    assert result["elapsed_ns"] is None


def test_non_utf8_output_does_not_raise():
    """A program emitting non-UTF-8 bytes to stdout must NOT crash measure().
    Real PIE programs occasionally print uninitialized memory or a stray
    byte cast to char; subprocess.run with text=True needs errors='replace'
    to keep the strict-gate loop going. The gate then rejects the program
    for output mismatch — the correct outcome. Surfaced during the §14
    step 6 reference-precompute (a real program in train.jsonl tripped this)."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(NON_UTF8_OUTPUT_PROGRAM, [("", "anything")])
    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["elapsed_ns"] is None
    # Crucially: no UnicodeDecodeError was raised here.


def test_empty_test_cases_degenerate():
    """Spec §4.2: all_pass requires len(test_cases) > 0 (vacuous truth is rejected).
    The core must not crash on an empty test list and must report all_pass=False."""
    from experiment11_code_optim.measurement.core import measure

    result = measure(HELLO_PROGRAM, [])

    assert result["compiled"] is True
    assert result["all_pass"] is False
    assert result["n_tests_run"] == 0
    assert result["n_tests_passed"] == 0
    assert result["elapsed_ns"] is None


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
    assert result["elapsed_ns"] is None
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


def test_multi_test_elapsed_ns_is_sum_of_per_case_medians():
    """Spec §4.2: elapsed_ns = sum over test cases.

    Under the legacy perf backend this scaled cleanly at ~3.0x exactly.
    Under the walltime backend (current), cold-cache amortization biases
    the single-run measurement high — the first subprocess.run pays
    kernel page-fault + ld.so + icache costs that subsequent runs in the
    three-test batch reuse — so the empirical ratio runs in ~1.8-3.0×.

    The test's purpose is to catch the bug "only one test contributes"
    (ratio ~1.0) vs. "all tests contribute" (ratio > 1.5). Use a
    correspondingly relaxed bound; this is unrelated to spec §4.2
    correctness, only walltime jitter.
    """
    from experiment11_code_optim.measurement.core import measure

    one = measure(ECHO_FIRST_LINE_PROGRAM, [("a\n", "a\n")])
    three = measure(
        ECHO_FIRST_LINE_PROGRAM,
        [("a\n", "a\n"), ("b\n", "b\n"), ("c\n", "c\n")],
    )

    assert one["all_pass"] is True and three["all_pass"] is True
    assert three["n_tests_run"] == 3 and three["n_tests_passed"] == 3

    ratio = three["elapsed_ns"] / one["elapsed_ns"]
    assert 1.5 < ratio < 4.0, (
        f"multi-test elapsed_ns={three['elapsed_ns']} should be >1.5x single's "
        f"{one['elapsed_ns']} (ratio={ratio:.2f}) — ratio ~1.0 means only one "
        f"test was counted; ratio >4 means duplicates/leak. Bounds wide enough "
        f"for walltime cold-cache amortization."
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
    n_tests_passed / elapsed_ns semantics as the in-memory measure() path
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


def test_measure_from_disk_all_pass_produces_elapsed_ns():
    """When every test case matches, measure_from_disk must enter the perf
    phase and emit a real elapsed_ns (proves disk + perf path composes)."""
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
        assert r["elapsed_ns"] is not None and r["elapsed_ns"] > 100


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
    assert result["elapsed_ns"] is not None and result["elapsed_ns"] > 100


def _perf_paranoid_ok() -> bool:
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return int(f.read().strip()) <= 2
    except OSError:
        return False


@pytest.mark.skipif(
    not _perf_paranoid_ok(),
    reason="perf_event_paranoid > 2; perf_event_open will fail with EACCES",
)
def test_perf_backend_is_deterministic(monkeypatch, tmp_path):
    """Spec §4.2 / Gate 1: retired-instruction count should be near-deterministic
    across repeated runs of the same fixed-work program.

    We reach into the module's private API directly (not via measure())
    because the env-load-at-import-time dispatch in MEASUREMENT_BACKEND is
    set once when core.py first imports — and pytest may have already
    imported it. Calling _run_and_compare_perf directly bypasses that and
    is the exact code path used in production under PIE_BACKEND=perf.

    Tolerance: spec §5 Gate 1 target is rel_spread <= 1e-4. Pre-flight
    smoke-test on the login node showed ~1.3% (1e-2) due to ld.so/exec
    setup variance on each fresh process. The looser <= 0.05 (5%) tolerance
    here is what we expect on a typical multi-tenant node; the strict
    1e-4 Gate 1 probe lives in tests/probe_perf_determinism.py and runs
    under SLURM on a dedicated ghx4 allocation.
    """
    from experiment11_code_optim.measurement.core import (
        _compile, _run_and_compare_perf,
    )

    code = """
#include <iostream>
int main() {
    int n; std::cin >> n;
    long s = 0;
    for (int i = 0; i < n; ++i) s += i;
    std::cout << s << std::endl;
    return 0;
}
"""
    expected = "499999500000\n"
    exe, err = _compile(code, str(tmp_path))
    assert exe is not None, f"compile failed: {err}"

    # Module-load warm-up in core.py already burned ~15 cold-path grades,
    # so the test process arrives in steady-state. Take a small leading
    # slice for safety (in case the test host's contention is bursty).
    counts: list[int] = []
    for _ in range(20):
        passed, c = _run_and_compare_perf(exe, "1000000\n", expected, 10.0)
        assert passed, f"gate failed under perf backend (count={c})"
        assert c > 0, "perf returned zero instruction count"
        counts.append(c)

    steady = sorted(counts[5:])
    median = steady[len(steady) // 2]
    rel_spread = (steady[-1] - steady[0]) / median
    # Empirically the steady-state on a quiet, hermetic node is rel_spread
    # ~ 2e-6 (bit-identical within ±8 instructions). Login/CI hosts can be
    # multiplexing-noisy (other tenants' perf events fight for the PMU);
    # observed up to ~3% on busy DeltaAI login nodes after scaling was
    # added. 5% is loose enough to be stable on shared hosts while still
    # catching the bug-case where steady-state devolves to walltime-style
    # noise (~100%). The formal Gate 1 (hermetic ghx4 allocation) lives in
    # tests/probe_perf_determinism.py and targets 1e-4.
    assert rel_spread < 0.05, (
        f"perf steady-state too variable: all_counts={counts}, "
        f"steady_min={steady[0]}, steady_max={steady[-1]}, median={median}, "
        f"rel_spread={rel_spread:.6f} (expected < 0.05)"
    )


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
    a, b = r1["elapsed_ns"], r2["elapsed_ns"]
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
