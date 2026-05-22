"""
PIE reward function (spec §8.2) with code deduplication and batched gate runner.

Public interface:

  - `compute_score(solution_str, ground_truth, ...)`         single-rollout
  - `compute_scores_batch(reward_inputs, ...)`               batch with dedup
  - `extract_code(solution_str)`                             shared parser

`compute_scores_batch` is the hot path: dedup-by-(code, problem_id), then a
two-level scheduler (compile-in-parallel + per-test-case run-in-parallel)
calls `measurement.core._compile` and `_run_and_compare` directly. Results
are scattered back to the original rollout indices. `measurement.core.
measure_from_disk` and `gate_runner.sh` are used only by the offline
reference-table precompute (latency-insensitive), not the RL hot path.

This module is the redirect target for `experiment4_llm.src.reward_function`
when `PIE_MODE=1` (see PARITY_CONFIG.md §8.6 redirect mechanism). Therefore the
public surface (`compute_score`, `compute_scores_batch`, `extract_code`) MUST
mirror the existing DeepCoder reward_function so that `insitu_eval.py`'s
hardcoded imports work transparently under the sys.modules shim.

The shim itself is installed in `experiment11_code_optim/train.py` when
`PIE_MODE=1` (closed F6). A subprocess smoke test in
`tests/test_pie_reward_fn.py::test_pie_mode_shim_grades_cpp_via_insitu_eval_import_path`
verifies the redirect end-to-end on a known-good C++ program.
"""
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import os
import pickle
import random
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait
from typing import Any


# ---------------------------------------------------------------------------
# Stage 1 noise mitigation: per-thread CPU affinity pinning.
#
# When PIE_REWARD_CPUSET is set (e.g. "24-71,96-143,168-215,240-287" — the
# trailing 48 cores of each Grace socket, kept away from the first 24 cores
# where Ray/vLLM workers tend to land), the FIRST grading call from each
# ThreadPoolExecutor worker thread pins that thread (and its forked
# subprocesses) to a single dedicated core. Subprocesses inherit the
# parent thread's affinity at fork time, so the spawned C++ binary runs
# pinned to the same core too.
#
# Why pin to a SINGLE core (not a NUMA node):
#   - numactl --cpunodebind keeps the program on one socket but lets the
#     OS scheduler bounce it across 72 cores of that socket, destroying
#     L1/L2 cache locality on every reschedule
#   - One-core pinning eliminates the bouncing → consistent L1 hit-rate
#     once warm
#   - Cost: a small constant warm-up per program (cold L1 on first run)
#
# Why a Python-side set instead of `taskset` wrapper:
#   - `taskset` adds fork+exec overhead per subprocess.run — exactly the
#     thing we don't want for short programs
#   - sched_setaffinity inside the reward thread costs ~1 µs and only
#     happens once per thread lifetime
#
# The first-touch assignment is lock-protected to avoid two threads
# racing for the same core on initialization.
# ---------------------------------------------------------------------------

def _expand_cpuset(spec: str) -> list[int]:
    """Parse 'a-b,c,d-e' style cpuset specs into a sorted list of ints."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


_REWARD_CORES: list[int] = _expand_cpuset(os.environ.get("PIE_REWARD_CPUSET", ""))
_CORE_ASSIGN_LOCK = threading.Lock()
_CORE_ITER = itertools.cycle(_REWARD_CORES) if _REWARD_CORES else None
_THREAD_PINNED: set[int] = set()


def _ensure_thread_pinned() -> None:
    """Pin the calling thread (and any subprocess it later forks) to one
    dedicated core from PIE_REWARD_CPUSET. No-op if the env var is unset
    or if this thread has already been pinned."""
    if _CORE_ITER is None:
        return
    tid = threading.get_ident()
    if tid in _THREAD_PINNED:
        return
    with _CORE_ASSIGN_LOCK:
        if tid in _THREAD_PINNED:
            return
        core = next(_CORE_ITER)
        try:
            os.sched_setaffinity(0, {core})
        except OSError:
            # If the core isn't in our cgroup's allowed set, give up
            # silently. Worst case: this thread runs unpinned, which is
            # the same as the no-PIE_REWARD_CPUSET baseline.
            pass
        _THREAD_PINNED.add(tid)

from experiment11_code_optim.measurement.core import (
    measure, measure_from_disk,  # kept for re-export compatibility
    _compile, _run_and_compare, MEASUREMENT_BACKEND, _scratch_dir,
)


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
    """Populate the in-memory cache of test cases keyed by problem_id.
    Idempotent: re-calls for the same dir are no-ops.

    Fast path (preferred): a sibling `<test_case_dir>.pkl` file holds a
    pre-serialized {problem_id: [(input, output)]} dict. Loading it is
    one open(), one read(), <1s — works identically on Lustre/NVMe/tmpfs/NFS.

    Slow path (fallback): walk the directory tree and read 21k+ files. On
    Lustre this is metadata-server-bound and serialises under concurrent
    invocations (the trap we hit in §9 GO/NO-GO eval). Always prefer the
    pickle; the slow path stays for fresh environments where the pickle
    hasn't been built yet.

    Override the pickle path via PIE_TEST_CASE_PICKLE env var if you want
    a non-default location.
    """
    global _TEST_CASE_CACHE_DIR
    if _TEST_CASE_CACHE_DIR == test_case_dir and _TEST_CASE_CACHE:
        return sum(len(v) for v in _TEST_CASE_CACHE.values())

    pickle_path = (
        os.environ.get("PIE_TEST_CASE_PICKLE")
        or (test_case_dir.rstrip("/") + ".pkl")
    )
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            _TEST_CASE_CACHE.clear()
            _TEST_CASE_CACHE.update(pickle.load(f))
        _TEST_CASE_CACHE_DIR = test_case_dir
        return sum(len(v) for v in _TEST_CASE_CACHE.values())

    # Fallback: walk the tree (slow on cold Lustre under concurrency).
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

# Reference-table RAM cache: {(split, problem_id) -> sorted list of elapsed_ns}.
# Loaded once per process via preload_reference_table(); identical Lustre-
# avoidance design as _TEST_CASE_CACHE above.
_REF_TABLE_CACHE: dict[tuple[str, str], list[int]] = {}
_REF_TABLE_CACHE_DIR: str | None = None


def preload_reference_table(reference_table_dir: str) -> int:
    """Populate the in-memory cache {(split, pid): elapsed_ns_values} for
    percentile-rank scoring. Idempotent.

    Fast path: load `<reference_table_dir>.pkl` if it exists (one read).
    Slow path: walk the dir tree (~1592 JSON files). Override the pickle
    path via PIE_REFERENCE_TABLE_PICKLE env var.
    """
    global _REF_TABLE_CACHE_DIR
    if _REF_TABLE_CACHE_DIR == reference_table_dir and _REF_TABLE_CACHE:
        return len(_REF_TABLE_CACHE)

    pickle_path = (
        os.environ.get("PIE_REFERENCE_TABLE_PICKLE")
        or (reference_table_dir.rstrip("/") + ".pkl")
    )
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            _REF_TABLE_CACHE.clear()
            _REF_TABLE_CACHE.update(pickle.load(f))
        _REF_TABLE_CACHE_DIR = reference_table_dir
        return len(_REF_TABLE_CACHE)

    # Fallback: walk the tree.
    _REF_TABLE_CACHE.clear()
    for split in ("train", "val", "test"):
        sdir = os.path.join(reference_table_dir, split)
        if not os.path.isdir(sdir):
            continue
        for fn in os.listdir(sdir):
            if not fn.endswith(".json"):
                continue
            pid = fn[:-5]
            try:
                with open(os.path.join(sdir, fn)) as f:
                    d = json.load(f)
                _REF_TABLE_CACHE[(split, pid)] = d.get("elapsed_ns_values", [])
            except (OSError, json.JSONDecodeError):
                pass
    _REF_TABLE_CACHE_DIR = reference_table_dir
    return len(_REF_TABLE_CACHE)


def _percentile_from_reference_table(
    elapsed_ns: int,
    problem_id: str,
    split: str,
    reference_table_dir: str | None,
) -> float:
    """Midrank percentile of `elapsed_ns` against the per-problem reference
    distribution. Spec §8.2:
        lo = bisect_left(refs, instr); hi = bisect_right(refs, instr)
        score = 1.0 - (lo + hi) / 2.0 / n_refs

    Loads the reference table from the in-process cache (populated by
    `preload_reference_table` on first call, idempotent). Returns 0.5 stub
    when reference_table_dir is None or the (split, pid) entry has no
    references (e.g., problems excluded from the precompute).
    """
    if reference_table_dir is None:
        return 0.5
    if _REF_TABLE_CACHE_DIR != reference_table_dir or not _REF_TABLE_CACHE:
        preload_reference_table(reference_table_dir)
    refs = _REF_TABLE_CACHE.get((split, problem_id))
    if not refs:
        return 0.5
    lo = bisect.bisect_left(refs, elapsed_ns)
    hi = bisect.bisect_right(refs, elapsed_ns)
    return max(0.0, min(1.0, 1.0 - (lo + hi) / 2.0 / len(refs)))


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


def grade_dedup_batch(
    dedup_tasks: dict[str, dict],
    test_case_dir: str,
    *,
    timeout_per_test: float = 6.0,
    num_workers: int | None = None,
    measurement_max_tests: int | None = None,
    pch_include_path: str | None = None,
    global_timeout: float | None = None,
) -> dict[str, dict]:
    """Run the two-level scheduler on already-deduped (code, problem_id) tasks.

    This is the shared core between `compute_scores_batch` (RL hot path) and
    `precompute/build_reference_table.py` (offline reference-table build).
    Both call sites benefit identically from the speed-fix gains.

    Args:
        dedup_tasks: {key: {"code": str, "problem_id": str, "split": str?}}.
            `key` is the caller's stable identifier;
            `md5(code + "::" + pid)` is the convention but any unique string
            works.
        test_case_dir: root of the unified test-case archive (see
            build_unified_test_cases.py output).
        timeout_per_test: per-test wall-clock cap.
        num_workers: pool size; defaults to VERL_REWARD_WORKERS env (64).
        measurement_max_tests: K-cap on test cases per problem. None disables
            the cap; if None, the PIE_MEASUREMENT_MAX_TESTS env var is read.
        pch_include_path: forwarded to `_compile`; falls back to
            PIE_PCH_INCLUDE_PATH env.
        global_timeout: wall-clock cap (seconds) on the phase-3 test-run pool.
            None disables the cap (precompute uses None). Under the cap,
            test-run futures that don't complete in time are cancelled
            (if not yet started) or abandoned (if running); their grades
            are marked all_pass=False with error="batch timeout". One bad
            program in a batch therefore cannot hold the rest of the batch
            past the cap — closes deferred fix F5.

    Returns:
        {key: result_dict} where result_dict matches `measure()`'s shape:
            {"compiled": bool, "all_pass": bool, "n_tests_run": int,
             "n_tests_passed": int, "elapsed_ns": int | None,
             "measurement_backend": str, "error": str | None}.

    Implementation:
        2a. Pre-pass: collect cases per key; partition by has-data.
        2b. Parallel compile (one tmpdir per unique code; scheduling unit = code).
        2c. Build flat list of (key, test_idx) run jobs.
        2d. Parallel test execution (scheduling unit = ONE test run; this is
            what unlocks the 21x speedup vs. the old per-grade dispatch — see
            PROBE_REPORT.md for the straggler-ceiling diagnosis).
            global_timeout, if set, caps phase 2d wall-clock.
        2e. Aggregate per grade. Explicit tmpdir cleanup in finally.
    """
    if pch_include_path is None:
        pch_include_path = os.environ.get("PIE_PCH_INCLUDE_PATH") or None
    n_workers = num_workers or int(os.environ.get("VERL_REWARD_WORKERS", "64"))
    if measurement_max_tests is None:
        env_cap = os.environ.get("PIE_MEASUREMENT_MAX_TESTS")
        if env_cap:
            measurement_max_tests = int(env_cap)

    # ---- 2a. Pre-pass: collect cases per key; partition by has-data ----
    cases_per_key: dict[str, list[tuple[str, str]]] = {}
    keys_with_cases: list[str] = []
    keys_no_cases: list[str] = []
    for key, task in dedup_tasks.items():
        cases = _get_test_cases(test_case_dir, task["problem_id"])
        if not cases:
            keys_no_cases.append(key)
            continue
        if measurement_max_tests is not None and len(cases) > measurement_max_tests:
            cases = _subsample_cases(cases, measurement_max_tests, task["problem_id"])
        cases_per_key[key] = cases
        keys_with_cases.append(key)

    compile_dir = _scratch_dir() or tempfile.gettempdir()

    def _compile_one(item: tuple[str, dict]) -> tuple[str, str | None, str, str | None]:
        _ensure_thread_pinned()  # Stage 1 noise mitigation; no-op if PIE_REWARD_CPUSET unset
        key, task = item
        td = tempfile.mkdtemp(dir=compile_dir, prefix="pie_meas_")
        exe, err = _compile(task["code"], td, pch_include_path=pch_include_path)
        return key, exe, td, err

    def _run_one(job: tuple[str, int, str, str, str]) -> tuple[str, int, bool, int]:
        _ensure_thread_pinned()  # Stage 1 noise mitigation
        key, idx, exe_path, inp, exp = job
        passed, elapsed_ns = _run_and_compare(
            exe_path, inp, exp, float(timeout_per_test)
        )
        return key, idx, passed, elapsed_ns

    # key -> (exe_path or None, tmpdir, error or None)
    compiled: dict[str, tuple[str | None, str, str | None]] = {}
    dedup_results: dict[str, dict] = {}

    try:
        # ---- 2b. Parallel compile ----
        if keys_with_cases:
            sub_tasks = [(k, dedup_tasks[k]) for k in keys_with_cases]
            with ThreadPoolExecutor(max_workers=min(n_workers, len(sub_tasks))) as pool:
                for key, exe, td, err in pool.map(_compile_one, sub_tasks):
                    compiled[key] = (exe, td, err)

        # ---- 2c. Build per-test-case run jobs ----
        run_jobs: list[tuple[str, int, str, str, str]] = []
        for key in keys_with_cases:
            exe, _td, _err = compiled[key]
            if exe is None:
                continue
            for idx, (inp, exp) in enumerate(cases_per_key[key]):
                run_jobs.append((key, idx, exe, inp, exp))

        # ---- 2d. Parallel test execution (with optional global_timeout) ----
        run_results: list[tuple[str, int, bool, int]] = []
        timed_out_keys: set[str] = set()
        if run_jobs:
            pool = ThreadPoolExecutor(max_workers=min(n_workers, len(run_jobs)))
            try:
                fut_to_job = {pool.submit(_run_one, j): j for j in run_jobs}
                if global_timeout is None:
                    for f in fut_to_job:
                        run_results.append(f.result())
                else:
                    done, not_done = cf_wait(
                        fut_to_job.keys(), timeout=float(global_timeout)
                    )
                    for f in done:
                        run_results.append(f.result())
                    for f in not_done:
                        f.cancel()
                        # job tuple = (key, idx, exe_path, inp, exp); index 0 = key
                        timed_out_keys.add(fut_to_job[f][0])
            finally:
                # cancel_futures requires Python 3.9+; wait=False lets running
                # subprocess.runs finish in the background while we return.
                pool.shutdown(wait=False, cancel_futures=True)

        # ---- 2e. Aggregate per grade ----
        per_grade_pass: dict[str, dict[int, bool]] = {}
        per_grade_elapsed: dict[str, dict[int, int]] = {}
        for key, idx, passed, elapsed_ns in run_results:
            per_grade_pass.setdefault(key, {})[idx] = passed
            per_grade_elapsed.setdefault(key, {})[idx] = elapsed_ns

        for key in keys_no_cases:
            dedup_results[key] = {
                "compiled": False, "all_pass": False,
                "n_tests_run": 0, "n_tests_passed": 0,
                "elapsed_ns": None,
                "measurement_backend": MEASUREMENT_BACKEND,
                "error": f"no test cases for {dedup_tasks[key]['problem_id']}",
            }
        for key in keys_with_cases:
            exe, _td, err = compiled.get(key, (None, "", "compile not attempted"))
            if exe is None:
                dedup_results[key] = {
                    "compiled": False, "all_pass": False,
                    "n_tests_run": 0, "n_tests_passed": 0,
                    "elapsed_ns": None,
                    "measurement_backend": MEASUREMENT_BACKEND,
                    "error": err,
                }
                continue
            cases = cases_per_key[key]
            passes = per_grade_pass.get(key, {})
            elapsed = per_grade_elapsed.get(key, {})
            n_pass = sum(1 for p in passes.values() if p)
            all_pass = (len(cases) > 0) and (n_pass == len(cases))
            err: str | None = None
            if key in timed_out_keys and not all_pass:
                n_done = len(passes)
                err = f"batch timeout ({n_done}/{len(cases)} tests completed)"
            dedup_results[key] = {
                "compiled": True,
                "all_pass": all_pass,
                "n_tests_run": len(cases),
                "n_tests_passed": n_pass,
                "elapsed_ns": sum(elapsed.values()) if all_pass else None,
                "measurement_backend": MEASUREMENT_BACKEND,
                "error": err,
            }
    finally:
        # Explicit tmpdir cleanup (mkdtemp doesn't auto-clean).
        for _exe, td, _err in compiled.values():
            if td and os.path.isdir(td):
                shutil.rmtree(td, ignore_errors=True)

    return dedup_results


def compute_scores_batch(
    reward_inputs: list[dict],
    timeout_per_test: int = 6,
    memory_limit_mb: int = 512,
    num_workers: int | None = None,
    global_timeout: float = 60.0,  # wall-clock cap on the phase-3 test-run pool
                                   # (see grade_dedup_batch). One pathological program
                                   # cannot hold the batch past this; un-completed
                                   # grades return score 0 with error="batch timeout".
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

    The heavy lifting lives in `grade_dedup_batch`; this fn is the rollout-
    indexed wrapper that handles parse + dedup at the front and percentile-
    rank scoring at the back.
    """
    if not reward_inputs:
        return []

    test_case_dir, reference_table_dir = _resolve_paths(kwargs)

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

    # ---- Phase 2: grade (two-level scheduler) ----
    dedup_results = grade_dedup_batch(
        dedup_tasks,
        test_case_dir=test_case_dir,
        timeout_per_test=float(timeout_per_test),
        num_workers=num_workers,
        measurement_max_tests=measurement_max_tests,
        global_timeout=float(global_timeout) if global_timeout else None,
    )

    # ---- Phase 3: scatter, compute score from elapsed_ns ----
    out: list[dict] = []
    for i, key in enumerate(completion_keys):
        if key is None:
            out.append(dict(_ZERO))
            continue
        r = dedup_results[key]
        if not r["all_pass"] or r["elapsed_ns"] is None:
            out.append(dict(_ZERO))
            continue
        task = dedup_tasks[key]
        score = _percentile_from_reference_table(
            r["elapsed_ns"], task["problem_id"], task["split"], reference_table_dir
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
