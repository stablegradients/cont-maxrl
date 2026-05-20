"""
PIE reward function (spec §8.2) with code deduplication and batched gate runner.

Public interface:

  - `compute_score(solution_str, ground_truth, ...)`         single-rollout
  - `compute_scores_batch(reward_inputs, ...)`               batch with dedup
  - `extract_code(solution_str)`                             shared parser

`compute_scores_batch` is the hot path: dedup-by-(code, problem_id), then grade
unique tasks via `measurement.core.measure_from_disk` (which uses gate_runner.sh
to run all test cases for a binary in ONE subprocess). Results are scattered
back to the original rollout indices.

This module is the redirect target for `experiment4_llm.src.reward_function`
when `PIE_MODE=1` (see PARITY_CONFIG.md §8.6 redirect mechanism). Therefore the
public surface (`compute_score`, `compute_scores_batch`, `extract_code`) MUST
mirror the existing DeepCoder reward_function so that `insitu_eval.py`'s
hardcoded imports work transparently under the sys.modules shim.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from experiment11_code_optim.measurement.core import measure, measure_from_disk


# ---------------------------------------------------------------------------
# In-memory test-case cache. 21,758 cases × ~1 KB each ≈ 22 MB total — cheap.
# On Lustre (/work/nvme) the per-rollout open()s of input.<i>.txt / output.<i>.txt
# serialise on Lustre metadata locks and cap concurrency at ~1×; preloading
# once at worker init and grading from RAM unlocks the full parallelism that
# `measure()`'s unit-test suite already validated.
# ---------------------------------------------------------------------------

_TEST_CASE_CACHE: dict[str, list[tuple[str, str]]] = {}
_TEST_CASE_CACHE_DIR: str | None = None


def preload_test_cases(test_case_dir: str) -> int:
    """Read every input.<i>.txt + output.<i>.txt pair under test_case_dir/<pid>/
    into memory. Returns the total number of (input, output) pairs cached.
    Idempotent: re-calls for the same dir are no-ops; calls for a different
    dir invalidate and re-fill the cache.
    """
    global _TEST_CASE_CACHE_DIR
    if _TEST_CASE_CACHE_DIR == test_case_dir and _TEST_CASE_CACHE:
        return sum(len(v) for v in _TEST_CASE_CACHE.values())
    _TEST_CASE_CACHE.clear()
    for pid in os.listdir(test_case_dir):
        pdir = os.path.join(test_case_dir, pid)
        if not os.path.isdir(pdir):
            continue
        cases: list[tuple[str, str]] = []
        indices: list[int] = []
        for fn in os.listdir(pdir):
            if not (fn.startswith("input.") and fn.endswith(".txt")):
                continue
            try:
                idx = int(fn[len("input."):-len(".txt")])
            except ValueError:
                continue
            if os.path.exists(os.path.join(pdir, f"output.{idx}.txt")):
                indices.append(idx)
        indices.sort()
        for i in indices:
            with open(os.path.join(pdir, f"input.{i}.txt")) as fi, \
                 open(os.path.join(pdir, f"output.{i}.txt")) as fo:
                cases.append((fi.read(), fo.read()))
        if cases:
            _TEST_CASE_CACHE[pid] = cases
    _TEST_CASE_CACHE_DIR = test_case_dir
    return sum(len(v) for v in _TEST_CASE_CACHE.values())


def _get_test_cases(test_case_dir: str, problem_id: str) -> list[tuple[str, str]]:
    if _TEST_CASE_CACHE_DIR != test_case_dir or not _TEST_CASE_CACHE:
        preload_test_cases(test_case_dir)
    return _TEST_CASE_CACHE.get(problem_id, [])


# ---------------------------------------------------------------------------
# Shared parser — mirrors experiment4_llm.src.reward_function.extract_code,
# with PIE's `### Optimized Version:` anchor handling (spec §7).
# ---------------------------------------------------------------------------

_SPECIAL_TOK_RE = re.compile(r"<\|[^>]+\|>")


def extract_code(solution_str: str) -> str:
    """Extract C++ code from a model completion.

    1. Strip `<|...|>` special tokens (from skip_special_tokens=False decode).
    2. Strip thinking content for reasoning models (split on `</think>`).
    3. If `### Optimized Version:` anchor is present (PIE prompt convention),
       focus on the post-anchor span.
    4. Take the LAST complete fenced code block (official LiveCodeBench
       extraction; matches `experiment4_llm.src.reward_function.extract_code`).
    5. Return "" if nothing extractable.
    """
    s = solution_str.strip()
    s = _SPECIAL_TOK_RE.sub("", s).strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if "### Optimized Version:" in s:
        s = s.split("### Optimized Version:", 1)[1].strip()

    lines = s.split("\n")
    bt_idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(bt_idx) >= 2:
        start_idx = bt_idx[-2]
        end_idx = bt_idx[-1]
        return "\n".join(lines[start_idx + 1:end_idx]).strip()
    # Fallback: no code fence — assume whole post-anchor body is code.
    return s


# ---------------------------------------------------------------------------
# Reference table lookup (percentile reward). For now this is a stub that
# returns 0.5 on all_pass; the real lookup lives at §14 step 6 when the
# reference-table precompute is wired in.
# ---------------------------------------------------------------------------

def _percentile_from_reference_table(
    instr_count: int,
    problem_id: str,
    split: str,
    reference_table_dir: str | None,
) -> float:
    """Midrank percentile of `instr_count` against the per-problem reference
    distribution. Spec §8.2:
        lo = bisect_left(refs, instr); hi = bisect_right(refs, instr)
        score = 1.0 - (lo + hi) / 2.0 / n_refs

    Until §14 step 6 produces real reference tables, return a stub 0.5 on
    all_pass so downstream callers see the right dict shape. The probe doesn't
    care about the numeric score; it cares about throughput.
    """
    if reference_table_dir is None:
        return 0.5
    ref_path = os.path.join(reference_table_dir, split, f"{problem_id}.json")
    try:
        with open(ref_path) as f:
            refs = json.load(f)["instr_counts"]
        if not refs:
            return 0.5
        lo = bisect.bisect_left(refs, instr_count)
        hi = bisect.bisect_right(refs, instr_count)
        return max(0.0, min(1.0, 1.0 - (lo + hi) / 2.0 / len(refs)))
    except (OSError, KeyError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Path resolution: kwargs (verl reward_kwargs path) → env (insitu_eval path).
# ---------------------------------------------------------------------------

def _resolve_paths(kwargs: dict[str, Any]) -> tuple[str, str | None]:
    test_case_dir = (
        kwargs.get("test_case_dir")
        or os.environ.get("PIE_TEST_CASE_DIR")
    )
    reference_table_dir = (
        kwargs.get("reference_table_dir")
        or os.environ.get("PIE_REFERENCE_TABLE_DIR")
    )
    if not test_case_dir:
        raise RuntimeError(
            "PIE reward fn requires test_case_dir via reward_kwargs or "
            "PIE_TEST_CASE_DIR env var (spec §8.6)"
        )
    return test_case_dir, reference_table_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ZERO = {"score": 0.0, "acc": 0.0}


def compute_score(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 6,
    memory_limit_mb: int = 512,
    data_source: str | None = None,
    extra_info: Any = None,
    **kwargs,
) -> dict:
    """Single-rollout reward; thin wrapper over compute_scores_batch."""
    return compute_scores_batch(
        [{"solution_str": solution_str, "ground_truth": ground_truth}],
        timeout_per_test=timeout_per_test,
        memory_limit_mb=memory_limit_mb,
        **kwargs,
    )[0]


def _subsample_cases(
    cases: list[tuple[str, str]], k: int, problem_id: str
) -> list[tuple[str, str]]:
    """Deterministically subsample to at most k test cases for a problem.
    Seed = hash(problem_id) so the precompute reference table and RL rollouts
    grade against IDENTICAL subsets — spec §13 GO-WITH-MITIGATION requires this.
    """
    if len(cases) <= k:
        return cases
    seed = int(hashlib.md5(problem_id.encode()).hexdigest()[:8], 16)
    return random.Random(seed).sample(cases, k)


def compute_scores_batch(
    reward_inputs: list[dict],
    timeout_per_test: int = 6,
    memory_limit_mb: int = 512,
    num_workers: int | None = None,
    global_timeout: int = 60,
    measurement_max_tests: int | None = None,
    **kwargs,
) -> list[dict]:
    """Batch reward with code-dedup. Returns one dict per input in the original
    order, each `{"score": float, "acc": float}`.

    Dedup key = `md5(code + "::" + problem_id)`. Per the speed_fix.md design
    note: the key MUST include problem_id because two identical C++ programs
    submitted for different problems must be graded against different test
    cases. Two rollouts with identical code for the SAME problem_id collapse
    to one compile/gate/measure cycle.
    """
    if not reward_inputs:
        return []

    test_case_dir, reference_table_dir = _resolve_paths(kwargs)
    pch_include = os.environ.get("PIE_PCH_INCLUDE_PATH") or None

    # ---- Phase 1: extract + dedup ----
    # completion_keys[i] = dedup key for rollout i (or None if unparseable)
    completion_keys: list[str | None] = []
    dedup_tasks: dict[str, dict] = {}  # key → {"code", "problem_id", "split"}

    for item in reward_inputs:
        code = extract_code(item["solution_str"])
        if not code:
            completion_keys.append(None)
            continue
        try:
            gt = json.loads(item["ground_truth"])
        except (json.JSONDecodeError, TypeError, KeyError):
            completion_keys.append(None)
            continue
        problem_id = gt.get("problem_id")
        if not problem_id:
            completion_keys.append(None)
            continue
        split = gt.get("split", "train")
        key = hashlib.md5(f"{code}::{problem_id}".encode()).hexdigest()
        completion_keys.append(key)
        if key not in dedup_tasks:
            dedup_tasks[key] = {"code": code, "problem_id": problem_id, "split": split}

    # ---- Phase 2: grade unique tasks in parallel ----
    n_workers = num_workers or int(os.environ.get("VERL_REWARD_WORKERS", "64"))

    n_perf_samples = int(os.environ.get("PIE_PERF_SAMPLES", "3"))
    if measurement_max_tests is None:
        env_cap = os.environ.get("PIE_MEASUREMENT_MAX_TESTS")
        if env_cap:
            measurement_max_tests = int(env_cap)

    def _grade(item: tuple[str, dict]) -> tuple[str, dict]:
        key, task = item
        cases = _get_test_cases(test_case_dir, task["problem_id"])
        if not cases:
            return key, {
                "compiled": False, "all_pass": False,
                "n_tests_run": 0, "n_tests_passed": 0,
                "instr_count": None, "measurement_backend": "perf",
                "error": f"no test cases for {task['problem_id']}",
            }
        if measurement_max_tests is not None and len(cases) > measurement_max_tests:
            cases = _subsample_cases(cases, measurement_max_tests, task["problem_id"])
        r = measure(
            task["code"],
            cases,
            per_test_timeout_s=float(timeout_per_test),
            n_perf_samples=n_perf_samples,
            pch_include_path=pch_include,
        )
        return key, r

    dedup_results: dict[str, dict] = {}
    if dedup_tasks:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(dedup_tasks))) as pool:
            for key, r in pool.map(_grade, dedup_tasks.items()):
                dedup_results[key] = r

    # ---- Phase 3: scatter, compute score from instr_count ----
    out: list[dict] = []
    for i, key in enumerate(completion_keys):
        if key is None:
            out.append(dict(_ZERO))
            continue
        r = dedup_results[key]
        if not r["all_pass"] or r["instr_count"] is None:
            out.append(dict(_ZERO))
            continue
        task = dedup_tasks[key]
        score = _percentile_from_reference_table(
            r["instr_count"], task["problem_id"], task["split"], reference_table_dir
        )
        out.append({"score": float(score), "acc": 1.0})

    return out


# ---------------------------------------------------------------------------
# Helper for the probe / debugging: report dedup stats on a batch.
# ---------------------------------------------------------------------------

def dedup_stats(reward_inputs: list[dict]) -> dict:
    """Without invoking grading: report how many unique (code, problem_id)
    pairs the batch dedups to. Used by the probe to size the speed-fix win."""
    keys: list[str | None] = []
    for item in reward_inputs:
        code = extract_code(item["solution_str"])
        if not code:
            keys.append(None); continue
        try:
            gt = json.loads(item["ground_truth"])
            pid = gt["problem_id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            keys.append(None); continue
        keys.append(hashlib.md5(f"{code}::{pid}".encode()).hexdigest())
    n_total = len(keys)
    n_unparseable = sum(1 for k in keys if k is None)
    n_unique = len({k for k in keys if k is not None})
    return {
        "n_total": n_total,
        "n_unparseable": n_unparseable,
        "n_parseable": n_total - n_unparseable,
        "n_unique_codes": n_unique,
        "dedup_ratio": (n_total - n_unparseable) / max(n_unique, 1),
    }
