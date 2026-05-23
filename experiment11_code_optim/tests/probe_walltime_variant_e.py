"""Variant E noise probe — K=10 back-to-back per program.

Diagnostic for whether median-of-K averaging would help reduce reward
noise. Runs the SAME compiled binary K=10 times back-to-back on N
concurrent threads; logs every elapsed_ns individually.

Per-thread metrics:
    warmup_factor = elapsed[0] / median(elapsed[1:])
        - >> 1 → first call is meaningfully slower (cold cache / loader);
                 median-of-K will help by discarding the cold sample.
        - ≈ 1  → no warmup effect; every call has the same expected cost.
    tail_spread   = (max(elapsed[1:]) - min(elapsed[1:])) / median(elapsed[1:])
        - < 0.3 → noise is "narrow"; median-of-K = strong noise reduction.
        - > 1.0 → noise is fat-tailed; median-of-K only marginally helps,
                  arithmetic mean might be worse, real fix is environmental.

This bypasses grade_dedup_batch's parallel-grading layer to isolate the
WITHIN-program noise. The N threads run independently — each one does
its own compile + 10 sequential runs — so we still see concurrency effects
on the OS scheduler, but not parallel-grade-aggregation noise.

Usage:
    python experiment11_code_optim/tests/probe_walltime_variant_e.py \
        --n-threads 64 --k 10 --label variant_e \
        --output-json /path/to/out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiment11_code_optim.measurement.core import (  # noqa: E402
    _compile, _run_and_compare, setup_pch, _scratch_dir,
)
from experiment11_code_optim.reward.pie_reward_fn import (  # noqa: E402
    _ensure_thread_pinned, preload_test_cases, _get_test_cases,
)


def load_known_good() -> tuple[str, str]:
    val_jsonl = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/val.jsonl"
    with open(val_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("problem_id") == "p02587":
                return rec["problem_id"], rec["tgt_code"]
        f.seek(0)
        rec = json.loads(next(f))
        return rec["problem_id"], rec["tgt_code"]


def one_thread_workload(args: tuple[int, str, str, list[tuple[str, str]], str, float, int]
                        ) -> dict:
    """Compile a fresh binary into a fresh tmpdir, run K times back-to-back
    against THE SAME test case (the first one), record every elapsed_ns.
    Returns a per-thread result dict."""
    (tid, pid, code, cases, pch, per_test_timeout_s, k) = args
    _ensure_thread_pinned()
    compile_dir = _scratch_dir() or tempfile.gettempdir()
    td = tempfile.mkdtemp(dir=compile_dir, prefix="pie_ve_")
    try:
        exe, err = _compile(code, td, pch_include_path=pch)
        if exe is None:
            return {"tid": tid, "ok": False, "err": err}
        # Use first test case repeatedly — we want WITHIN-program noise,
        # not different-input noise.
        inp, exp = cases[0]
        elapsed_list: list[int] = []
        all_pass = True
        for _ in range(k):
            passed, elapsed_ns = _run_and_compare(exe, inp, exp, per_test_timeout_s)
            elapsed_list.append(elapsed_ns)
            if not passed:
                all_pass = False
        return {"tid": tid, "ok": all_pass, "elapsed_ns": elapsed_list,
                "core": sorted(os.sched_getaffinity(0))}
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def summarize(per_thread: list[dict]) -> dict:
    """Aggregate per-thread metrics into the global summary."""
    ok = [t for t in per_thread if t.get("ok")]
    if not ok:
        return {"ok": False, "n_threads_ok": 0, "n_threads_total": len(per_thread)}

    warmup_factors: list[float] = []
    tail_spreads: list[float] = []
    all_first: list[int] = []
    all_post: list[int] = []  # elapsed[1:] from every thread, pooled

    for t in ok:
        e = t["elapsed_ns"]
        if len(e) < 2:
            continue
        first = e[0]
        rest = e[1:]
        rest_sorted = sorted(rest)
        med_rest = rest_sorted[len(rest_sorted) // 2]
        if med_rest == 0:
            continue
        warmup_factors.append(first / med_rest)
        tail_spreads.append((rest_sorted[-1] - rest_sorted[0]) / med_rest)
        all_first.append(first)
        all_post.extend(rest)

    def pct(xs: list[float | int], p: int) -> float:
        if not xs:
            return float("nan")
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))])

    summary = {
        "ok": True,
        "n_threads_total": len(per_thread),
        "n_threads_ok": len(ok),
        "warmup_factor": {
            "median": round(statistics.median(warmup_factors), 3) if warmup_factors else None,
            "p25": round(pct(warmup_factors, 25), 3),
            "p75": round(pct(warmup_factors, 75), 3),
            "min": round(min(warmup_factors), 3) if warmup_factors else None,
            "max": round(max(warmup_factors), 3) if warmup_factors else None,
        },
        "tail_spread": {  # ((max - min) / median) on elapsed[1:] per thread
            "median": round(statistics.median(tail_spreads), 3) if tail_spreads else None,
            "p25": round(pct(tail_spreads, 25), 3),
            "p75": round(pct(tail_spreads, 75), 3),
            "min": round(min(tail_spreads), 3) if tail_spreads else None,
            "max": round(max(tail_spreads), 3) if tail_spreads else None,
        },
        "first_call_ms": {
            "median": round(statistics.median(all_first) / 1e6, 2) if all_first else None,
            "p10": round(pct(all_first, 10) / 1e6, 2),
            "p90": round(pct(all_first, 90) / 1e6, 2),
        },
        "post_calls_ms": {
            "median": round(statistics.median(all_post) / 1e6, 2) if all_post else None,
            "p10": round(pct(all_post, 10) / 1e6, 2),
            "p90": round(pct(all_post, 90) / 1e6, 2),
        },
    }
    # Pooled rel_spread on the post-warmup samples — directly comparable
    # to the K=1 probe's rel_spread metric, but only counting runs 2..K.
    if all_post:
        s = sorted(all_post)
        med = s[len(s) // 2]
        summary["post_pool_rel_spread"] = round((s[-1] - s[0]) / med, 3) if med else None
        summary["post_pool_n"] = len(all_post)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-threads", type=int, default=64)
    ap.add_argument("--k", type=int, default=10,
                    help="Back-to-back runs per thread")
    ap.add_argument("--label", type=str, default="variant_e")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    os.environ.setdefault("PIE_COMPILE_SCRATCH",
                          f"/tmp/{os.environ.get('USER', 'u')}_pie_scratch")
    os.makedirs(os.environ["PIE_COMPILE_SCRATCH"], exist_ok=True)
    if "PIE_PCH_INCLUDE_PATH" not in os.environ:
        os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
        print(f"[ve] built PCH at {os.environ['PIE_PCH_INCLUDE_PATH']}", flush=True)
    pch = os.environ["PIE_PCH_INCLUDE_PATH"]

    n_cases = preload_test_cases(os.environ.get(
        "PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests"))
    print(f"[ve] preloaded {n_cases} test cases", flush=True)

    pid, code = load_known_good()
    cases = _get_test_cases(os.environ.get(
        "PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests"), pid)
    if not cases:
        print(f"[ve] ERROR: no cases for {pid}", flush=True)
        sys.exit(1)

    cpuset = os.environ.get("PIE_REWARD_CPUSET", "<unset>")
    print(f"[ve] label={args.label} n_threads={args.n_threads} k={args.k} "
          f"cpuset={cpuset!r} pid={pid}", flush=True)

    workloads = [(i, pid, code, cases, pch, 6.0, args.k) for i in range(args.n_threads)]
    t0 = time.monotonic()
    per_thread: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.n_threads) as pool:
        futs = [pool.submit(one_thread_workload, w) for w in workloads]
        for f in as_completed(futs):
            per_thread.append(f.result())
    wall = time.monotonic() - t0

    summary = summarize(per_thread)
    summary["label"] = args.label
    summary["wall_seconds"] = round(wall, 2)
    summary["cpuset"] = cpuset
    summary["n_threads"] = args.n_threads
    summary["k"] = args.k

    print(f"[ve:{args.label}] {json.dumps(summary, indent=2)}", flush=True)
    if args.output_json:
        out = {"summary": summary, "per_thread": per_thread}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[ve] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
