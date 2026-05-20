"""
Integration tests for the C++ measurement core against the staged PIE data.

These are SKIPPED if the unified test-case archive doesn't exist (so
pytest stays green in environments without /work/nvme staging).

Run with:

    PYTHONPATH=. /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        -m pytest experiment11_code_optim/tests/test_measurement_integration.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
TESTS_ROOT = PIE_DATA_ROOT / "tests"
VAL_JSONL = PIE_DATA_ROOT / "raw" / "val.jsonl"

_data_ready = TESTS_ROOT.is_dir() and (TESTS_ROOT / "MANIFEST.json").exists()
_skip_no_data = pytest.mark.skipif(
    not _data_ready,
    reason=(
        "PIE unified test-case archive not built. Run "
        "experiment11_code_optim/precompute/build_unified_test_cases.py first."
    ),
)


def _load_test_cases(problem_id: str) -> list[tuple[str, str]]:
    """Read all (input.<i>.txt, output.<i>.txt) pairs for a problem from the unified tree."""
    pdir = TESTS_ROOT / problem_id
    n = sum(1 for f in os.listdir(pdir) if f.startswith("input.") and f.endswith(".txt"))
    cases = []
    for i in range(n):
        with open(pdir / f"input.{i}.txt") as f_in, open(pdir / f"output.{i}.txt") as f_out:
            cases.append((f_in.read(), f_out.read()))
    return cases


def _first_val_record_with_problem(problem_id: str) -> dict:
    """Return the first val.jsonl record whose problem_id matches."""
    with open(VAL_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            if rec["problem_id"] == problem_id:
                return rec
    raise LookupError(f"problem_id={problem_id!r} not found in val.jsonl")


@_skip_no_data
def test_grade_known_good_solution_from_val_passes_strict_gate():
    """End-to-end: a known-good fast solution from val.jsonl, graded against the unified
    test-case archive for the same problem_id, must pass the strict all-pass gate and
    produce a real elapsed_ns > 0. Validates the whole measurement → data pipeline."""
    from experiment11_code_optim.measurement.core import measure

    # Use p02587 as the canonical example: it appears in val.jsonl AND in pie-perf Public,
    # so we get multi-case grading (5 inputs) rather than the CodeNet 1-test fallback.
    problem_id = "p02587"

    rec = _first_val_record_with_problem(problem_id)
    tgt_code: str = rec["tgt_code"]
    assert rec["tgt_status"] == "Accepted", "val.jsonl tgt_code is supposed to be Accepted"

    cases = _load_test_cases(problem_id)
    assert len(cases) >= 1, f"unified tree returned 0 cases for {problem_id}"

    result = measure(tgt_code, cases)

    # A historically-Accepted submission should compile cleanly under our g++ 14.2 -O3.
    assert result["compiled"] is True, (
        f"known-Accepted tgt_code failed to compile: {result.get('error')!r}"
    )
    # The strict all-pass gate is the contract; this is the most informative single signal.
    assert result["all_pass"] is True, (
        f"known-good tgt_code did NOT pass the strict gate. n_tests_run="
        f"{result['n_tests_run']}, n_tests_passed={result['n_tests_passed']}. "
        f"Likely cause: per-line trailing-whitespace normalisation mismatch, or the "
        f"Public test-case set encodes a problem-specific output convention (e.g. "
        f"multiple valid outputs, precision tolerance) that the strict gate rejects."
    )
    assert result["elapsed_ns"] is not None and result["elapsed_ns"] > 100
    assert result["measurement_backend"] in ("perf", "papi", "walltime")
