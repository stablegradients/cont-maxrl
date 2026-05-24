"""
Tests for the PIE reward function (spec §8.2) — dedup + batched gate.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
TESTS_ROOT = PIE_DATA_ROOT / "tests"
VAL_JSONL = PIE_DATA_ROOT / "raw" / "val.jsonl"

_data_ready = TESTS_ROOT.is_dir() and (TESTS_ROOT / "MANIFEST.json").exists()
_skip_no_data = pytest.mark.skipif(
    not _data_ready,
    reason="PIE unified test-case archive not built; see precompute/build_unified_test_cases.py",
)


def _gt(problem_id: str, split: str = "train", src_instr_count: int | None = None) -> str:
    """Build a ground_truth JSON string. Pass src_instr_count to exercise the
    log2(src/cand) reward; omit it to exercise the gate-pass-no-src-instr
    fallback path (returns score=0 acc=1 speedup=0 with a stderr warning)."""
    d: dict = {"problem_id": problem_id, "split": split}
    if src_instr_count is not None:
        d["src_instr_count"] = src_instr_count
    return json.dumps(d)


def _tgt_code_for(problem_id: str) -> str:
    """Look up val.jsonl's first record with the given problem_id and return tgt_code."""
    with open(VAL_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            if rec["problem_id"] == problem_id:
                return rec["tgt_code"]
    raise LookupError(problem_id)


# ---------------------------------------------------------------------------
# extract_code: parser behaviour
# ---------------------------------------------------------------------------

def test_extract_code_strips_special_tokens_and_takes_last_block():
    from experiment11_code_optim.reward.pie_reward_fn import extract_code
    out = extract_code("preamble <|im_start|>\nignored\n```cpp\nFIRST\n```\nmiddle\n```cpp\nLAST\n```<|im_end|>")
    assert out.strip() == "LAST"


def test_extract_code_anchors_on_pie_marker():
    from experiment11_code_optim.reward.pie_reward_fn import extract_code
    out = extract_code(
        "Some preamble that contains a misleading code block:\n"
        "```cpp\nIGNORED_BEFORE_ANCHOR\n```\n"
        "### Optimized Version:\n"
        "```cpp\nKEEP_THIS\n```\n"
    )
    assert out.strip() == "KEEP_THIS"


def test_extract_code_handles_thinking_models():
    from experiment11_code_optim.reward.pie_reward_fn import extract_code
    out = extract_code("<think>\ndeliberation\n```cpp\nIN_THOUGHT\n```\n</think>\n```cpp\nFINAL\n```")
    assert out.strip() == "FINAL"


# ---------------------------------------------------------------------------
# dedup_stats: pure-Python dedup math (no grading)
# ---------------------------------------------------------------------------

def test_dedup_stats_collapses_identical_inputs():
    from experiment11_code_optim.reward.pie_reward_fn import dedup_stats
    # 16 rollouts: 4 unique codes × 4 copies each, same problem_id
    inputs = []
    for k in range(4):
        for _ in range(4):
            inputs.append({
                "solution_str": f"```cpp\nint main() {{ return {k}; }}\n```",
                "ground_truth": _gt("pXX"),
            })
    s = dedup_stats(inputs)
    assert s["n_total"] == 16
    assert s["n_unparseable"] == 0
    assert s["n_unique_codes"] == 4
    assert s["dedup_ratio"] == 4.0


def test_dedup_stats_same_code_different_problem_ids_count_separately():
    """Spec'd dedup key includes problem_id; two identical codes for
    different problems must NOT collapse — they test against different inputs."""
    from experiment11_code_optim.reward.pie_reward_fn import dedup_stats
    same_code = "```cpp\nint main(){return 0;}\n```"
    inputs = [
        {"solution_str": same_code, "ground_truth": _gt("pA")},
        {"solution_str": same_code, "ground_truth": _gt("pB")},
        {"solution_str": same_code, "ground_truth": _gt("pA")},
    ]
    s = dedup_stats(inputs)
    assert s["n_total"] == 3
    assert s["n_unique_codes"] == 2  # (code,pA), (code,pB)


def test_dedup_stats_marks_unparseable_correctly():
    from experiment11_code_optim.reward.pie_reward_fn import dedup_stats
    inputs = [
        {"solution_str": "no fence anywhere", "ground_truth": _gt("pA")},
        {"solution_str": "```cpp\nint main(){return 0;}\n```", "ground_truth": _gt("pA")},
        {"solution_str": "```cpp\nint main(){return 0;}\n```", "ground_truth": "not json"},
    ]
    s = dedup_stats(inputs)
    assert s["n_total"] == 3
    assert s["n_unparseable"] == 1  # only "not json" GT is unparseable; "no fence" still extracts text


# ---------------------------------------------------------------------------
# compute_scores_batch: end-to-end with the unified test archive
# ---------------------------------------------------------------------------

@_skip_no_data
def test_compute_scores_batch_known_good_program_passes_via_dedup_path():
    """A known-Accepted val.jsonl tgt_code, submitted 4 times for the same
    problem_id, must come back as 4 identical non-zero results, AND dedup
    must collapse the 4 inputs to 1 compile/grade internally — proved by
    comparing wall-clock against the single-input baseline."""
    from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch

    pid = "p02587"
    code = _tgt_code_for(pid)
    fenced = f"```cpp\n{code}\n```"
    one = [{"solution_str": fenced, "ground_truth": _gt(pid)}]
    four = [{"solution_str": fenced, "ground_truth": _gt(pid)} for _ in range(4)]

    t0 = time.monotonic()
    r_single = compute_scores_batch(one, test_case_dir=str(TESTS_ROOT))
    t_single = time.monotonic() - t0
    t0 = time.monotonic()
    r_four = compute_scores_batch(four, test_case_dir=str(TESTS_ROOT))
    t_four = time.monotonic() - t0

    assert len(r_four) == 4
    for r in r_four:
        assert r["acc"] == 1.0, f"unexpected fail: {r}"
        # New reward: max(0, log2(src/cand)). Without src_instr_count in the
        # ground_truth, the fn returns score=0 acc=1 speedup=0 (warn-once);
        # with one, score can be any non-negative float. Lower-bound only.
        assert r["score"] >= 0.0, f"score must be non-negative; got {r['score']}"
        assert "speedup" in r, f"score dict must carry speedup field; got {r}"
    assert all(r == r_four[0] for r in r_four)
    # Without dedup, 4 identical inputs would take ~4× a single grade. With
    # dedup, the batch collapses to one grading-task, so wall-clock should be
    # ≤ ~2× the single-input wall-clock (slack for thread-pool overhead and
    # filesystem warmth differences).
    assert t_four < 2.0 * t_single, (
        f"dedup ineffective: t_single={t_single:.2f}s vs t_four={t_four:.2f}s "
        f"(ratio {t_four/t_single:.2f}× — expected ≤ 2.0× under dedup)"
    )


@_skip_no_data
def test_compute_scores_batch_unparseable_rollouts_score_zero():
    from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch
    results = compute_scores_batch(
        [{"solution_str": "no code here", "ground_truth": _gt("p02587")}],
        test_case_dir=str(TESTS_ROOT),
    )
    assert results == [{"score": 0.0, "acc": 0.0, "speedup": 0.0}]


@_skip_no_data
def test_compute_score_single_rollout_matches_batch():
    """compute_score is a thin wrapper over compute_scores_batch; its
    return value on a single rollout must equal the batch result for [item]."""
    from experiment11_code_optim.reward.pie_reward_fn import compute_score, compute_scores_batch

    pid = "p02587"
    code = _tgt_code_for(pid)
    item = {"solution_str": f"```cpp\n{code}\n```", "ground_truth": _gt(pid)}
    single = compute_score(item["solution_str"], item["ground_truth"],
                           test_case_dir=str(TESTS_ROOT))
    batched = compute_scores_batch([item], test_case_dir=str(TESTS_ROOT))[0]
    assert single == batched


# ---------------------------------------------------------------------------
# log2(src/cand) reward computation: regression test on the new reward formula.
# ---------------------------------------------------------------------------


@_skip_no_data
def test_compute_scores_batch_log2_speedup_reward_with_src_instr_count():
    """Reward = max(0, log2(src/cand)) on gate-pass. Construct a ground_truth
    with src_instr_count >> cand (the known-good code) and confirm we get a
    positive reward + speedup > 1, and that the score == log2(speedup)."""
    import math
    from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch

    pid = "p02587"
    code = _tgt_code_for(pid)
    # First grade to discover the actual cand_instr (whatever the backend
    # produces today). Then re-grade with src_instr = 10× that value, so we
    # expect reward = log2(10) ≈ 3.32.
    r0 = compute_scores_batch(
        [{"solution_str": f"```cpp\n{code}\n```",
          "ground_truth": _gt(pid)}],
        test_case_dir=str(TESTS_ROOT),
    )[0]
    assert r0["acc"] == 1.0, f"baseline grade must pass; got {r0}"

    # Pick a src_instr_count that gives a clean expected speedup. We don't
    # know exact cand_instr in advance, but a synthetic ratio works: set
    # src = 10 * a known small number, and trust the reward fn to use it
    # against whatever cand_instr comes back. Use a large dummy src so the
    # log2 ratio is unambiguously positive.
    dummy_src = 10_000_000_000  # 1e10; much larger than any candidate
    r1 = compute_scores_batch(
        [{"solution_str": f"```cpp\n{code}\n```",
          "ground_truth": _gt(pid, src_instr_count=dummy_src)}],
        test_case_dir=str(TESTS_ROOT),
    )[0]
    assert r1["acc"] == 1.0
    assert r1["speedup"] > 1.0, (
        f"speedup must be > 1 when src >> cand; got {r1}"
    )
    assert r1["score"] > 0.0, f"score must be positive when speedup > 1; got {r1}"
    # The score MUST equal log2(speedup) within float tolerance.
    expected = max(0.0, math.log2(r1["speedup"]))
    assert abs(r1["score"] - expected) < 1e-6, (
        f"score={r1['score']} != log2(speedup={r1['speedup']})={expected}"
    )


@_skip_no_data
def test_compute_scores_batch_log2_speedup_clamps_to_zero_when_cand_slower():
    """When cand is SLOWER than src (speedup < 1), score = max(0, log2(<1)) = 0,
    but acc remains 1.0 and speedup carries the < 1 ratio for diagnostics."""
    from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch

    pid = "p02587"
    code = _tgt_code_for(pid)
    # src_instr deliberately tiny → speedup < 1 → score clamped to 0.
    tiny_src = 1
    r = compute_scores_batch(
        [{"solution_str": f"```cpp\n{code}\n```",
          "ground_truth": _gt(pid, src_instr_count=tiny_src)}],
        test_case_dir=str(TESTS_ROOT),
    )[0]
    assert r["acc"] == 1.0, f"gate must still pass; got {r}"
    assert r["speedup"] < 1.0, f"synthetic tiny src must yield speedup < 1; got {r}"
    assert r["score"] == 0.0, f"score must clamp to 0 below speedup=1; got {r}"


def test_compute_scores_batch_unparseable_returns_full_zero_dict():
    """Unparseable rollout returns {"score": 0.0, "acc": 0.0, "speedup": 0.0}.
    Verl's NaiveRewardManager pivots every key into batch.non_tensor_batch,
    so the speedup field must always be present (otherwise the metrics hook
    crashes with a missing key)."""
    from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch
    r = compute_scores_batch(
        [{"solution_str": "no code here", "ground_truth": _gt("p02587")}],
        test_case_dir=str(TESTS_ROOT),
    )
    assert r == [{"score": 0.0, "acc": 0.0, "speedup": 0.0}]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# F6: PIE_MODE sys.modules shim — proves train.py's redirect works.
# Runs in a subprocess so the Python interpreter starts cold and sees only
# our shim (pytest's already-imported sys.modules don't pollute).
# ---------------------------------------------------------------------------

@_skip_no_data
def test_pie_mode_shim_grades_cpp_via_insitu_eval_import_path():
    """F6: train.py installs `sys.modules["experiment4_llm.src.reward_function"]
    = pie_reward_fn` when PIE_MODE=1. insitu_eval.py's hardcoded
        from experiment4_llm.src.reward_function import compute_scores_batch
    must therefore resolve to PIE's reward fn, so its C++ programs grade
    correctly. Without this, insitu_eval grades C++ as Python → every
    rollout scores 0 silently. This test runs as a subprocess so it
    exercises the exact import order train.py uses at launch."""
    import subprocess
    import sys as _sys
    import textwrap

    script = textwrap.dedent(
        '''
        import os, sys, json
        sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")

        assert os.environ.get("PIE_MODE") == "1", "PIE_MODE not set in env"

        # Mimic train.py's shim install (the actual under-test contract).
        from experiment11_code_optim.reward import pie_reward_fn
        sys.modules["experiment4_llm.src.reward_function"] = pie_reward_fn

        # Now mimic insitu_eval's hardcoded import.
        from experiment4_llm.src.reward_function import (
            compute_score, compute_scores_batch, extract_code,
        )

        # Verify provenance — must be PIE's, not DeepCoder's.
        for fn in (compute_score, compute_scores_batch, extract_code):
            assert fn.__module__.endswith("pie_reward_fn"), (
                f"shim NOT installed: {fn.__name__} is {fn.__module__}"
            )

        # End-to-end: grade a known-good C++ program through the shimmed import.
        pid = "p02587"
        code = None
        with open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/val.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec["problem_id"] == pid:
                    code = rec["tgt_code"]
                    break
        assert code, "could not load p02587 tgt_code"

        gt = json.dumps({"problem_id": pid, "split": "val"})
        result = compute_score(
            f"```cpp\\n{code}\\n```", gt,
            test_case_dir="/work/nvme/bgmo/sramasubramanian/pie_data/tests",
        )
        assert result["acc"] == 1.0, (
            f"shim wired but reward fn fails on known-good C++: {result}"
        )
        print("OK", result)
        '''
    )
    env = {**os.environ, "PIE_MODE": "1"}
    proc = subprocess.run(
        [_sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"F6 shim subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert proc.stdout.startswith("OK"), f"unexpected stdout: {proc.stdout!r}"


# ---------------------------------------------------------------------------
# F5: global_timeout enforcement
# ---------------------------------------------------------------------------

_SLEEP_PROGRAM = (
    "```cpp\n"
    "#include <chrono>\n"
    "#include <iostream>\n"
    "#include <thread>\n"
    "int main() {\n"
    "    std::this_thread::sleep_for(std::chrono::seconds(20));\n"
    "    std::cout << \"hi\\n\";\n"
    "    return 0;\n"
    "}\n"
    "```"
)


@_skip_no_data
def test_global_timeout_caps_batch_and_passes_other_grades():
    """F5: a pathological program (sleeps 20s) in the batch must not block
    the rest. With global_timeout=3 and per_test_timeout=30 (so per-test
    timeout does NOT save us), the batch must return within ~4s; the slow
    grade must return score=0 with error="batch timeout"; the good grade
    (known-Accepted p02587 code) must return its normal non-zero result."""
    import time
    from experiment11_code_optim.reward.pie_reward_fn import (
        compute_scores_batch, preload_test_cases,
    )

    # Warm the test-case cache (~22s cold). Without this the test mis-attributes
    # cold-cache preload time to the timeout path.
    preload_test_cases(str(TESTS_ROOT))

    pid = "p02587"
    good_code = _tgt_code_for(pid)
    reward_inputs = [
        {"solution_str": f"```cpp\n{good_code}\n```", "ground_truth": _gt(pid)},
        {"solution_str": _SLEEP_PROGRAM, "ground_truth": _gt(pid)},
    ]

    t0 = time.monotonic()
    results = compute_scores_batch(
        reward_inputs,
        test_case_dir=str(TESTS_ROOT),
        timeout_per_test=30,  # high, so per-test timeout cannot kick in first
        global_timeout=3.0,
        measurement_max_tests=5,  # bound case count regardless of env
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0, (
        f"global_timeout=3s not honoured; batch took {elapsed:.1f}s. "
        f"The sleep(20) grade must NOT hold the pool past the cap."
    )
    assert results[0]["acc"] == 1.0, f"good grade must still pass; got {results[0]}"
    assert results[1]["acc"] == 0.0, f"sleep grade must fail; got {results[1]}"


# NOTE: the legacy GATE C regression on the percentile-rank reference-table
# lookup has been removed. The new reward fn anchors against per-record
# src_instr_count (embedded in ground_truth by build_pie_parquet.py) and no
# longer reads a reference table. See
# test_compute_scores_batch_log2_speedup_reward_with_src_instr_count above
# for the replacement regression that exercises the new formula end-to-end.
