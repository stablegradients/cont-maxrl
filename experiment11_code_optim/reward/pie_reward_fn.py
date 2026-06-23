"""
PIE reward function with code deduplication and batched gate runner.

Timing source (PIE_TIMING_SOURCE): "gem5" (DEFAULT) times gate-passing
programs with the validated x86 gem5 pipeline (measurement/gem5_backend.py),
speedup = src_gem5_ticks / cand_gem5_ticks against the precomputed reference
table (PIE_GEM5_REF_TABLE). "perf" is the legacy retired-instruction timing
(shown ~uncorrelated with gem5 speedup, r=0.07) kept for ablation only; there
reward = max(0, log2(src_instr_count / cand_instr_count)) on gate-pass, with
src_instr_count from ground_truth["src_instr_count"] (pre-computed by
experiment11_code_optim/precompute/build_src_instr_table.py and embedded by
build_pie_parquet.py).

Each rollout's score dict carries:
  {"score": float,    # the RL reward (mode- and timing-source-dependent)
   "acc":   float,    # 1.0 if all tests pass, else 0.0
   "speedup": float}  # src_time/cand_time on gate-pass (gem5 ticks or perf
                      # instructions per PIE_TIMING_SOURCE), else 0.0

verl's NaiveRewardManager pivots every key in this dict into
batch.non_tensor_batch[<key>], so the monkey-patched compute_data_metrics
hook in experiment11_code_optim/guards/pie_metrics.py can read acc/speedup
arrays per step to compute gate_pass_rate / speedup_rate / mean_speedup etc.

Public interface:
  - `compute_score(solution_str, ground_truth, ...)`         single-rollout
  - `compute_scores_batch(reward_inputs, ...)`               batch with dedup
  - `extract_code(solution_str)`                             shared parser

`compute_scores_batch` is the hot path: dedup-by-(code, problem_id), then a
two-level scheduler (compile-in-parallel + per-test-case run-in-parallel)
calls `measurement.core._compile` and `_run_and_compare` directly. Results
are scattered back to the original rollout indices. `measurement.core.
measure_from_disk` and `gate_runner.sh` are used only by the offline
build_src_instr_table.py / build_reference_table.py precompute (latency-
insensitive), not the RL hot path.

NOTE (2026-06-13): this is the LEGACY reward (perf / reference-table era). The
LIVE pie-gem5 runs use `reward/gem5_reward.py` instead — `PIE_REWARD_IMPL=gem5`
routes train.py's shim and insitu_eval/post_hoc_eval there. This module is the
DEFAULT redirect target for `experiment4_llm.src.reward_function` when
`PIE_MODE=1` and `PIE_REWARD_IMPL != gem5` (legacy reeval + the #46 overfit
ablation, which pins `PIE_TIMING_SOURCE=perf`). The public surface
(`compute_score`, `compute_scores_batch`, `extract_code`) MUST still mirror the
DeepCoder reward_function so `insitu_eval.py`'s hardcoded imports work under the
shim (see PARITY_CONFIG.md §8.6).

The shim itself is installed in `experiment11_code_optim/train.py` when
`PIE_MODE=1` (closed F6). A subprocess smoke test in
`tests/test_pie_reward_fn.py::test_pie_mode_shim_grades_cpp_via_insitu_eval_import_path`
verifies the redirect end-to-end on a known-good C++ program.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
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


# ---------------------------------------------------------------------------
# Stage 2 noise mitigation: median-of-K per-test.
#
# When PIE_MEASUREMENT_SAMPLES_PER_TEST >= 2, every test case is executed K
# times serially within the same grading job; the test is considered passed
# only if ALL K runs pass the gate, and the per-test elapsed_ns reported to
# the aggregator is the median of the K wall-clock samples. This targets the
# within-program upper-tail jitter (page faults, scheduler preemption, NUMA
# memory-bus contention) that the Stage 1 A/B probe (2026-05-22) showed
# Stage 1 affinity alone did not fix.
#
# IMPORTANT bias note: the reference table was built with K=1 (single
# sample per test). Setting K>=2 here makes candidate timings systematically
# smaller than the reference (median <= mean under right-skewed noise),
# inflating the candidate's percentile rank. The fair-comparison fix is
# Stage 3 (regrade the reference table with the same K) — without that,
# Stage 2 alone is acceptable as a noise probe (does the reward-stream CV
# drop?) but not as the long-term production setting.
# ---------------------------------------------------------------------------

_MEASUREMENT_SAMPLES_PER_TEST: int = max(
    1, int(os.environ.get("PIE_MEASUREMENT_SAMPLES_PER_TEST", "1"))
)


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
# Path resolution: kwargs (verl reward_kwargs path) → env (insitu_eval path).
# ---------------------------------------------------------------------------

def _resolve_paths(kwargs: dict[str, Any]) -> str:
    test_case_dir = (
        kwargs.get("test_case_dir")
        or os.environ.get("PIE_TEST_CASE_DIR")
    )
    if not test_case_dir:
        raise RuntimeError(
            "PIE reward fn requires test_case_dir via reward_kwargs or "
            "PIE_TEST_CASE_DIR env var"
        )
    return test_case_dir


_WARNED_MISSING_SRC_INSTR = False


def _warn_missing_src_instr_once() -> None:
    """Log once per process if ground_truth lacks src_instr_count. Indicates
    a parquet mismatch (built without build_src_instr_table.py being run
    first). Without src_instr_count the log2(src/cand) reward can't be
    computed; gate-pass rollouts will return score=0, speedup=0."""
    global _WARNED_MISSING_SRC_INSTR
    if _WARNED_MISSING_SRC_INSTR:
        return
    _WARNED_MISSING_SRC_INSTR = True
    print(
        "[pie_reward_fn] WARN: ground_truth lacks 'src_instr_count'. "
        "Rebuild parquets via build_src_instr_table.py + build_pie_parquet.py. "
        "All gate-pass rollouts will report score=0 until fixed.",
        file=sys.stderr, flush=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ZERO = {"score": 0.0, "acc": 0.0, "speedup": 0.0}

# Reward formula selector (env-gated; default = original PIE log2-speedup so
# existing runs are unaffected). "pass_plus_logspeedup": r = 1[gate-pass] +
# ln(1 + speedup) — rewards correctness itself, so even flat problems (speedup~1)
# get pass/fail reward variance and a non-degenerate gradient.
# gem5-era modes (spd = src_gem5_ticks / cand_gem5_ticks, natural log):
#   "spd":               r = 1[pass] * spd
#   "pass_plus_logspd":  r = 1[pass] * (1 + ln(spd))
#   "gated_logspd":      r = 1[pass & spd>1] * ln(spd)
#   "tgt_relative":      r = 1[pass] * tgt_gem5_ticks / cand_gem5_ticks
_REWARD_MODE = os.environ.get("PIE_REWARD_MODE", "log2_speedup") or "log2_speedup"
_KNOWN_REWARD_MODES = {
    "log2_speedup", "pass_plus_logspeedup", "gated_logspeedup",
    "spd", "pass_plus_logspd", "gated_logspd", "tgt_relative",
}
if _REWARD_MODE not in _KNOWN_REWARD_MODES:
    raise RuntimeError(
        f"PIE_REWARD_MODE={_REWARD_MODE!r} is not one of {sorted(_KNOWN_REWARD_MODES)}"
    )

# Under pass_plus_logspd a measured slow rollout scores 1+ln(spd) < 0, so a
# gem5 TIMEOUT must not score better (0) than honest slowness — that would
# reward producing code slow enough to time out. Timeout/budget-class
# failures of gate-passers get this floor instead (default 1+ln(0.01), i.e.
# "at least 100x slower than src"). Non-timeout infra failures still score 0.
_GEM5_TIMEOUT_FLOOR = float(os.environ.get(
    "PIE_GEM5_TIMEOUT_REWARD_FLOOR", str(1.0 + math.log(0.01))))

# Per-tc sim tick budget = factor x (src_gem5_ticks / k) from the ref table.
# Rollouts slower than ~factor x src stop simulating immediately (cause
# budget_exceeded -> censored to floor/0) instead of burning wall-clock to
# the 900s forkloop timeout. <=0 disables. Default 3.0.
_GEM5_TC_BUDGET_FACTOR = float(os.environ.get("PIE_GEM5_TC_BUDGET_FACTOR", "3.0"))

# ---------------------------------------------------------------------------
# Timing source selector. "gem5" (DEFAULT): the correctness gate still runs on
# the fast native path, but the timing that defines speedup comes from the
# validated x86 gem5 pipeline (measurement/gem5_backend.py) — the same metric
# as the PIE dataset's agg_runtime. The perf retired-instruction timing was
# shown ~uncorrelated with gem5 speedup (r=0.07, 2026-06); it stays available
# behind PIE_TIMING_SOURCE=perf for back-compat/ablation only.
#
# Under gem5: speedup = src_gem5_ticks / cand_gem5_ticks, where src (and tgt
# for the "tgt_relative" mode) come from a precomputed reference table
# (PIE_GEM5_REF_TABLE, default data_pie/gem5_ref_table.json) measured with the
# SAME harness and the SAME deterministic K-test-case subset (_subsample_cases
# seed) — so the ratio compares identical inputs end to end.
# ---------------------------------------------------------------------------

_TIMING_SOURCE = (os.environ.get("PIE_TIMING_SOURCE", "gem5") or "gem5").lower()
if _TIMING_SOURCE not in ("gem5", "perf"):
    raise RuntimeError(
        f"PIE_TIMING_SOURCE must be 'gem5' or 'perf', got {_TIMING_SOURCE!r}"
    )

_GEM5_REF_TABLE_PATH = (
    os.environ.get("PIE_GEM5_REF_TABLE")
    or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data_pie", "gem5_ref_table.json")
)
_GEM5_REFS: dict[str, dict] | None = None
_WARNED_GEM5_REF: set[str] = set()


def _gem5_ref(problem_id: str) -> dict | None:
    """Lazy-load the gem5 reference table; one warn per missing problem.

    An unreadable/empty TABLE raises: silently training a whole run with
    all-zero rewards is strictly worse than crashing at the first batch.
    """
    global _GEM5_REFS
    if _GEM5_REFS is None:
        try:
            with open(_GEM5_REF_TABLE_PATH) as f:
                _GEM5_REFS = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"PIE_TIMING_SOURCE=gem5 but the gem5 ref table is unreadable "
                f"at {_GEM5_REF_TABLE_PATH} ({e}). Build it with "
                f"precompute/build_gem5_ref_table.py or set PIE_GEM5_REF_TABLE."
            ) from e
        if not _GEM5_REFS:
            raise RuntimeError(
                f"gem5 ref table at {_GEM5_REF_TABLE_PATH} is empty; every "
                f"reward would silently be 0. Build it first."
            )
    ref = _GEM5_REFS.get(problem_id)
    if ref is None and problem_id not in _WARNED_GEM5_REF:
        _WARNED_GEM5_REF.add(problem_id)
        print(f"[pie_reward_fn] WARN: no gem5 ref entry for {problem_id}; "
              f"its gate-pass rollouts get score=0.", file=sys.stderr, flush=True)
    return ref


def compute_score(
    solution_str: str,
    ground_truth: str,
    timeout_per_test: int = 30,
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


# ---------------------------------------------------------------------------
# Test-case subsampling. Two modes (PIE_TC_SAMPLE_MODE):
#   "fixed" (default): seed = hash(problem_id) — the SAME subset every call,
#       matching reference tables built offline (spec §13 GO-WITH-MITIGATION).
#   "random": a fresh nonce is drawn at the START of every compute_scores_batch
#       call and XOR'd into the seed, so each training step grades/times a
#       DIFFERENT random subset (prevents overfitting to fixed test cases).
#       Within one call the gate and the gem5 phase still share the subset.
#       Under gem5 timing this requires the ref table to carry per-tc tick
#       arrays for ALL cases (src_per_tc_all / tgt_per_tc_all) so the spd
#       ratio is computed on the same inputs the candidate ran.
# The nonce is a module global: reward calls are serialized per worker
# process (one batch per step), so there is no concurrent-call hazard.
# ---------------------------------------------------------------------------

_TC_SAMPLE_MODE = (os.environ.get("PIE_TC_SAMPLE_MODE", "fixed") or "fixed").lower()
if _TC_SAMPLE_MODE not in ("fixed", "random"):
    raise RuntimeError(
        f"PIE_TC_SAMPLE_MODE must be 'fixed' or 'random', got {_TC_SAMPLE_MODE!r}"
    )
_TC_SAMPLE_NONCE: int = 0


def _subsample_indices(n_cases: int, k: int, problem_id: str) -> list[int]:
    """Indices of the k-subset for this problem (sorted). See mode docs above.
    random.sample's selection depends only on population LENGTH and the seed,
    so this picks the identical positions the legacy case-tuple sampling did."""
    if n_cases <= k:
        return list(range(n_cases))
    seed = int(hashlib.md5(problem_id.encode()).hexdigest()[:8], 16)
    if _TC_SAMPLE_MODE == "random":
        seed ^= _TC_SAMPLE_NONCE
    return sorted(random.Random(seed).sample(range(n_cases), k))


def _subsample_cases(
    cases: list[tuple[str, str]], k: int, problem_id: str
) -> list[tuple[str, str]]:
    """Subsample to at most k test cases for a problem (see _subsample_indices)."""
    if len(cases) <= k:
        return cases
    return [cases[i] for i in _subsample_indices(len(cases), k, problem_id)]


def grade_dedup_batch(
    dedup_tasks: dict[str, dict],
    test_case_dir: str,
    *,
    timeout_per_test: float = 30.0,
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
        K = _MEASUREMENT_SAMPLES_PER_TEST
        if K <= 1:
            passed, elapsed_ns = _run_and_compare(
                exe_path, inp, exp, float(timeout_per_test)
            )
            return key, idx, passed, elapsed_ns
        # Stage 2: median-of-K. All K must pass for the test to count as a
        # pass; short-circuit on first failure to avoid wasted work on a
        # broken program.
        samples: list[int] = []
        for _ in range(K):
            passed, elapsed_ns = _run_and_compare(
                exe_path, inp, exp, float(timeout_per_test)
            )
            if not passed:
                return key, idx, False, elapsed_ns
            samples.append(elapsed_ns)
        samples.sort()
        med = samples[len(samples) // 2]  # upper median for even K (K=3 → samples[1])
        return key, idx, True, med

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
    timeout_per_test: int = 30,
    memory_limit_mb: int = 512,
    num_workers: int | None = None,
    global_timeout: float = 300.0,  # wall-clock cap on the phase-3 test-run pool.
                                    # Bumped 60→300 after -O0 switch: -O0 programs
                                    # are 10-50× slower, and at G=16 × 256 prompts
                                    # the 60s cap was clipping ~30% of batches.
                                    # One pathological program still cannot hold the
                                    # batch past this; un-completed grades return
                                    # score 0 with error="batch timeout".
    measurement_max_tests: int | None = None,
    **kwargs,
) -> list[dict]:
    """Batch reward with code-dedup. Returns one dict per input in the
    original order, each `{"score": float, "acc": float, "speedup": float}`.

    Reward depends on the timing source (env `PIE_TIMING_SOURCE`, default
    `gem5`) and `PIE_REWARD_MODE` (default `log2_speedup`):
      - gem5 (default): speedup = src_gem5_ticks / cand_gem5_ticks via the
        precomputed gem5 reference table (PIE_GEM5_REF_TABLE); reward = the
        PIE_REWARD_MODE transform of that speedup.
      - perf (LEGACY, used by the #46 overfit scripts which pin
        PIE_TIMING_SOURCE=perf): speedup = src_instr / cand_instr with
        src_instr from per-rollout ground_truth["src_instr_count"], reward
        = max(0, log2 speedup) on gate-pass.
    Either way score is 0 unless the rollout passes the correctness gate.
    NOTE: the production pie-gem5 RL runs use reward/gem5_reward.py, not this.
    (The per-rollout src list runs parallel to completion_keys — different
    prompts that dedup to the same candidate code can carry different src refs.)

    Dedup key = `md5(code + "::" + problem_id)`. The key MUST include
    problem_id because two identical C++ programs submitted for different
    problems must be graded against different test cases. Two rollouts with
    identical code for the SAME problem_id collapse to one compile/gate/
    measure cycle.

    Extra kwargs (e.g., `reference_table_dir` from legacy callers) are
    absorbed into **kwargs and silently ignored.
    """
    if not reward_inputs:
        return []

    # random mode: one fresh nonce per batch -> a new test-case subset each
    # training step, shared by the gate and the gem5 timing phase
    global _TC_SAMPLE_NONCE
    if _TC_SAMPLE_MODE == "random":
        _TC_SAMPLE_NONCE = int.from_bytes(os.urandom(8), "little")

    test_case_dir = _resolve_paths(kwargs)

    # ---- Phase 1: extract + dedup, capture per-rollout src_instr_count ----
    # completion_keys[i] = dedup key for rollout i (or None if unparseable)
    # src_instrs[i]      = src_instr_count for rollout i (or None)
    completion_keys: list[str | None] = []
    src_instrs: list[int | None] = []
    dedup_tasks: dict[str, dict] = {}  # key → {"code", "problem_id", "split"}

    for item in reward_inputs:
        code = extract_code(item["solution_str"])
        if not code:
            completion_keys.append(None)
            src_instrs.append(None)
            continue
        try:
            gt = json.loads(item["ground_truth"])
        except (json.JSONDecodeError, TypeError, KeyError):
            completion_keys.append(None)
            src_instrs.append(None)
            continue
        problem_id = gt.get("problem_id")
        if not problem_id:
            completion_keys.append(None)
            src_instrs.append(None)
            continue
        split = gt.get("split", "train")
        src_instr = gt.get("src_instr_count")
        if src_instr is None and _TIMING_SOURCE == "perf":
            # only the perf formula needs src_instr_count; gem5 uses the
            # ref table instead
            _warn_missing_src_instr_once()
        key = hashlib.md5(f"{code}::{problem_id}".encode()).hexdigest()
        completion_keys.append(key)
        src_instrs.append(src_instr if isinstance(src_instr, (int, float)) and src_instr > 0 else None)
        if key not in dedup_tasks:
            dedup_tasks[key] = {"code": code, "problem_id": problem_id, "split": split}

    # ---- Phase 2: grade (two-level scheduler; correctness gate) ----
    dedup_results = grade_dedup_batch(
        dedup_tasks,
        test_case_dir=test_case_dir,
        timeout_per_test=float(timeout_per_test),
        num_workers=num_workers,
        measurement_max_tests=measurement_max_tests,
        global_timeout=float(global_timeout) if global_timeout else None,
    )

    # ---- Phase 2.5 (gem5 timing source): time unique gate-passers ----
    # The native gate above established correctness; gem5 only measures time.
    # Inputs are the SAME deterministic K-subset the gate and the precomputed
    # src/tgt references used (_subsample_cases is seeded by problem_id).
    gem5_ticks: dict[str, dict] = {}
    g5_subset: dict[str, tuple[list[int], int]] = {}  # key -> (tc indices timed, n_cases_total)
    if _TIMING_SOURCE == "gem5":
        from experiment11_code_optim.measurement.gem5_backend import gem5_time_batch
        if measurement_max_tests is None:
            env_cap = os.environ.get("PIE_MEASUREMENT_MAX_TESTS")
            k_cap = int(env_cap) if env_cap else None
        else:
            k_cap = measurement_max_tests
        g5_tasks: dict[str, dict] = {}
        for key, task in dedup_tasks.items():
            if not dedup_results[key]["all_pass"]:
                continue
            all_cases = _get_test_cases(test_case_dir, task["problem_id"])
            idxs = _subsample_indices(
                len(all_cases), k_cap if k_cap is not None else len(all_cases),
                task["problem_id"])
            g5_tasks[key] = {"code": task["code"],
                             "inputs": [all_cases[i][0] for i in idxs]}
            g5_subset[key] = (idxs, len(all_cases))
            if _GEM5_TC_BUDGET_FACTOR > 0:
                _ref = _gem5_ref(task["problem_id"])
                per_tc = (_ref or {}).get("src_per_tc_all")
                if per_tc and len(per_tc) == len(all_cases):
                    # budget = factor x mean src tick over the CHOSEN subset
                    mean_src = sum(per_tc[i] for i in idxs) / len(idxs)
                    g5_tasks[key]["max_tc_ticks"] = int(_GEM5_TC_BUDGET_FACTOR * mean_src)
                elif _ref and _ref.get("src_gem5_ticks") and _ref.get("k"):
                    g5_tasks[key]["max_tc_ticks"] = int(
                        _GEM5_TC_BUDGET_FACTOR * _ref["src_gem5_ticks"] / _ref["k"])
        gem5_ticks = gem5_time_batch(g5_tasks)
        # Per-batch visibility: silent gem5 censoring is the historical bug
        # class; one summary line per call makes a drifting failure rate
        # impossible to miss in the trainer log.
        if g5_tasks:
            n_ok = sum(1 for r in gem5_ticks.values() if r.get("ticks"))
            errs: dict[str, int] = {}
            for r in gem5_ticks.values():
                if not r.get("ticks"):
                    e = (r.get("error") or "unknown").split(":")[0]
                    errs[e] = errs.get(e, 0) + 1
            print(f"[pie_reward_fn] gem5 timing: {n_ok}/{len(g5_tasks)} ok"
                  + (f", failures: {errs}" if errs else ""),
                  file=sys.stderr, flush=True)

    # ---- Phase 3: scatter, compute reward per rollout ----
    out: list[dict] = []
    for i, key in enumerate(completion_keys):
        if key is None:
            out.append(dict(_ZERO))
            continue
        r = dedup_results[key]
        if not r["all_pass"]:
            out.append(dict(_ZERO))
            continue

        if _TIMING_SOURCE == "gem5":
            g5 = gem5_ticks.get(key)
            pid = dedup_tasks[key]["problem_id"]
            ref = _gem5_ref(pid)
            if (g5 is None or g5.get("ticks") is None or g5["ticks"] <= 0
                    or ref is None or not ref.get("src_gem5_ticks")):
                # gate passed but timing or reference unavailable -> no
                # speedup claim; acc stays 1.0 so pass-rate metrics are honest.
                # Timeout-class failures under pass_plus_logspd take the
                # floor, not 0 — otherwise timing out beats being measured
                # slow (reward inversion).
                err = (g5 or {}).get("error") or ""
                score = (_GEM5_TIMEOUT_FLOOR
                         if _REWARD_MODE == "pass_plus_logspd"
                         and ("timeout" in err or "budget exceeded" in err)
                         else 0.0)
                out.append({"score": float(score), "acc": 1.0, "speedup": 0.0})
                continue
            # spd is only valid if src ref and candidate cover the SAME test
            # cases. Preferred: per-tc arrays for ALL cases (src_per_tc_all)
            # -> src time summed over exactly the indices the candidate ran
            # (required for PIE_TC_SAMPLE_MODE=random). Legacy fallback:
            # whole-subset totals, valid only when the runtime subset matches
            # the table's fixed k/n.
            idxs, n_total = g5_subset.get(key, (None, None))
            src_per_tc = ref.get("src_per_tc_all")
            tgt_per_tc = ref.get("tgt_per_tc_all")
            src_ticks = tgt_ticks = None
            if src_per_tc and n_total is not None and len(src_per_tc) == n_total:
                src_ticks = sum(src_per_tc[i] for i in idxs)
                if tgt_per_tc and len(tgt_per_tc) == n_total:
                    tgt_ticks = sum(tgt_per_tc[i] for i in idxs)
            elif (_TC_SAMPLE_MODE == "fixed"
                  and (ref.get("k") is None or ref.get("k") == len(idxs or []))
                  and (ref.get("n_cases_total") is None
                       or ref.get("n_cases_total") == n_total)):
                src_ticks = ref["src_gem5_ticks"]
                tgt_ticks = ref.get("tgt_gem5_ticks")
            if not src_ticks:
                if pid not in _WARNED_GEM5_REF:
                    _WARNED_GEM5_REF.add(pid)
                    print(f"[pie_reward_fn] WARN: gem5 ref subset mismatch for "
                          f"{pid} (mode={_TC_SAMPLE_MODE}, ref k={ref.get('k')}/"
                          f"n={ref.get('n_cases_total')}, per_tc_all="
                          f"{bool(src_per_tc)}, runtime k={len(idxs or [])}/"
                          f"n={n_total}); rebuild the ref table with "
                          f"build_gem5_ref_table.py. score=0.",
                          file=sys.stderr, flush=True)
                out.append({"score": 0.0, "acc": 1.0, "speedup": 0.0})
                continue
            cand_ticks = g5["ticks"]
            spd = src_ticks / cand_ticks
            if _REWARD_MODE == "spd":
                reward = spd
            elif _REWARD_MODE == "pass_plus_logspd":
                reward = 1.0 + math.log(spd)
            elif _REWARD_MODE == "gated_logspd":
                reward = math.log(spd) if spd > 1.0 else 0.0
            elif _REWARD_MODE == "tgt_relative":
                if not tgt_ticks:
                    out.append({"score": 0.0, "acc": 1.0, "speedup": float(spd)})
                    continue
                reward = tgt_ticks / cand_ticks
            elif _REWARD_MODE == "pass_plus_logspeedup":
                reward = 1.0 + math.log(1.0 + spd)
            elif _REWARD_MODE == "gated_logspeedup":
                reward = math.log(spd) if spd > 1.0 else 0.0
            else:  # "log2_speedup" default
                reward = max(0.0, math.log2(spd))
            out.append({"score": float(reward), "acc": 1.0, "speedup": float(spd)})
            continue

        # ---- perf timing source (PIE_TIMING_SOURCE=perf; legacy) ----
        if r["elapsed_ns"] is None or r["elapsed_ns"] <= 0:
            out.append(dict(_ZERO))
            continue
        cand_instr = r["elapsed_ns"]  # name kept across backends; under PIE_BACKEND=perf this is retired-instructions
        src_instr = src_instrs[i]
        if src_instr is None:
            # Gate-passed but no src_instr_count in ground_truth — log already
            # warned once. Record gate-pass for acc but emit 0 reward / 0.0
            # speedup placeholder so the metrics don't claim spurious speedups.
            out.append({"score": 0.0, "acc": 1.0, "speedup": 0.0})
            continue
        ratio = src_instr / cand_instr
        if _REWARD_MODE == "pass_plus_logspeedup":
            reward = 1.0 + math.log(1.0 + ratio)  # r = 1[pass] + ln(1+speedup)
        elif _REWARD_MODE == "gated_logspeedup":
            reward = math.log(ratio) if ratio > 1.0 else 0.0  # r = 1[pass & speedup>1]·ln(speedup)
        else:
            reward = max(0.0, math.log2(ratio))
        out.append({"score": float(reward), "acc": 1.0, "speedup": float(ratio)})

    return out


# ---------------------------------------------------------------------------
# Grade-path warm-up: prime ThreadPoolExecutor + fork/exec + SIGSTOP wrapper
# FS cache by running one synthetic grade_dedup_batch at module load.
#
# Without this, Gate 4 trial-1 (first grade_dedup_batch call after module
# import) showed ~14/16 spurious gate-fails while trials 2-N were
# bit-identical. Likely cause: cold libc thread-creation paths in the new
# ThreadPoolExecutor + cold FS cache for the wrapper binary under the
# production threading pattern. The warm-up runs the SAME code path that
# production uses, with one trivial synthetic task, so each subsequent
# production batch starts warm.
#
# Gated on PIE_BACKEND=perf — under walltime there's no such cold-trial-1
# anomaly (the original codepath does the warm-up implicitly via thousands
# of historical runs under the perf-event-stats backend), and we want to
# avoid adding latency to the walltime fallback path.
# ---------------------------------------------------------------------------

def _grade_warm_up() -> None:
    """Run one synthetic grade_dedup_batch to prime cold paths. No-op if
    PIE_BACKEND != "perf" or if compile fails (the warm-up is best-effort)."""
    if MEASUREMENT_BACKEND != "perf":
        return
    warmup_pid = "_pie_perf_warmup_"
    warmup_code = (
        "#include <iostream>\n"
        "int main(){ std::cout << \"warm\\n\"; return 0; }\n"
    )
    # Inject one synthetic test case into the cache; clean up after.
    global _TEST_CASE_CACHE_DIR
    had_entry = warmup_pid in _TEST_CASE_CACHE
    _TEST_CASE_CACHE[warmup_pid] = [("", "warm\n")]
    saved_dir = _TEST_CASE_CACHE_DIR
    if _TEST_CASE_CACHE_DIR is None:
        _TEST_CASE_CACHE_DIR = "_pie_perf_warmup_dir"

    key = hashlib.md5(f"{warmup_code}::{warmup_pid}".encode()).hexdigest()
    tasks = {key: {"code": warmup_code, "problem_id": warmup_pid, "split": "val"}}
    try:
        grade_dedup_batch(
            tasks,
            test_case_dir=_TEST_CASE_CACHE_DIR,
            measurement_max_tests=1,
            timeout_per_test=10.0,
            # Match the production thread-pool size so libc's pthread paths
            # warm up at production scale. 64 trivial tasks would be wasteful;
            # we submit 1 task and rely on grade_dedup_batch's min(n_workers,
            # len(sub_tasks)) clamping — so pool spawn is for 1 thread.
            # Instead, run multiple synthetic tasks to force broad warm-up.
            num_workers=64,
        )
    except Exception:
        # Warm-up failure must not block module import.
        pass
    finally:
        if not had_entry:
            _TEST_CASE_CACHE.pop(warmup_pid, None)
        _TEST_CASE_CACHE_DIR = saved_dir


# Second helper: spawn 64 worker threads to spread libc pthread-create's
# allocator state across the warm-up. The ThreadPoolExecutor inside
# grade_dedup_batch clamps to len(sub_tasks); a single warmup task gives
# only 1 thread. Spawning 64 explicit warmup threads forces broad warm-up.
def _warm_thread_pool(n: int = 64) -> None:
    if MEASUREMENT_BACKEND != "perf":
        return
    try:
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda _: None, range(n)))
    except Exception:
        pass


_warm_thread_pool(64)
_grade_warm_up()


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
