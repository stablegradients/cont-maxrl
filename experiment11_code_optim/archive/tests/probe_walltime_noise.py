"""Wall-clock noise probe (Stage 0 of the noise-fix plan).

Runs a known-good val.jsonl tgt program N times concurrently through the
production reward grader (grade_dedup_batch) and reports the distribution
of elapsed_ns. Should be invoked under a real SLURM allocation with the
SAME env vars the training job uses (PIE_PCH_INCLUDE_PATH, VERL_REWARD_WORKERS,
PIE_REWARD_CPUSET, PIE_MEASUREMENT_MAX_TESTS, etc.).

Each "task" is the same C++ program with a unique trailing-whitespace
suffix, so the dedup key in `grade_dedup_batch` doesn't collapse them and
we get N independent concurrent gradings. The dedup behavior itself is
not what we're probing — we're measuring per-grade wall-clock variance
under the production concurrency pattern.

Usage:
    PYTHONPATH=. python experiment11_code_optim/tests/probe_walltime_noise.py [N=50]

Recommended N=50 with VERL_REWARD_WORKERS=64. Higher N gives tighter
percentile estimates but runs longer.

Output:
    N, problem_id, n_tests, n_all_pass, median_ms, p10, p25, p75, p90, min, max,
    rel_spread (= (max-min)/median), CV (= stdev/median),
    plus the same after PIE_REWARD_CPUSET pinning if that env was set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiment11_code_optim.measurement.core import setup_pch  # noqa: E402
from experiment11_code_optim.reward.pie_reward_fn import (  # noqa: E402
    grade_dedup_batch, preload_test_cases,
)


def load_known_good() -> tuple[str, str, str]:
    """Find a val.jsonl record whose tgt_code reliably passes the strict
    gate at K=5. Returns (problem_id, code, split)."""
    val_jsonl = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/val.jsonl"
    with open(val_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("problem_id") == "p02587":
                return rec["problem_id"], rec["tgt_code"], "val"
        # fallback: just take the first
        f.seek(0)
        rec = json.loads(next(f))
        return rec["problem_id"], rec["tgt_code"], "val"


def make_probe_tasks(pid: str, code: str, split: str, n: int) -> dict[str, dict]:
    """Build N tasks with the same logical program but unique dedup keys.
    Trailing whitespace doesn't change behavior but does change the hash."""
    tasks = {}
    for i in range(n):
        unique_code = code + ("\n" + " " * (i + 1))
        key = hashlib.md5(f"probe_{i}".encode()).hexdigest()
        tasks[key] = {"code": unique_code, "problem_id": pid, "split": split}
    return tasks


def run_probe(n: int = 50, label: str = "baseline") -> dict:
    pid, code, split = load_known_good()
    print(f"[probe:{label}] target = ({pid}, {split}), |code| = {len(code)}", flush=True)

    tasks = make_probe_tasks(pid, code, split, n)
    test_case_dir = os.environ.get(
        "PIE_TEST_CASE_DIR", "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
    )

    t0 = time.monotonic()
    results = grade_dedup_batch(
        tasks,
        test_case_dir=test_case_dir,
        measurement_max_tests=int(os.environ.get("PIE_MEASUREMENT_MAX_TESTS", "5")),
    )
    wall = time.monotonic() - t0

    elapsed_ns = [r["elapsed_ns"] for r in results.values()
                  if r["all_pass"] and r["elapsed_ns"] is not None]
    n_pass = len(elapsed_ns)
    if n_pass < 2:
        print(f"[probe:{label}] FAIL: only {n_pass}/{n} tasks passed the gate "
              f"(can't compute spread)", flush=True)
        return {"label": label, "n": n_pass, "ok": False}

    elapsed_ns.sort()
    med = statistics.median(elapsed_ns)
    p10 = elapsed_ns[max(0, n_pass // 10)]
    p25 = elapsed_ns[n_pass // 4]
    p75 = elapsed_ns[3 * n_pass // 4]
    p90 = elapsed_ns[min(n_pass - 1, 9 * n_pass // 10)]
    mn = elapsed_ns[0]
    mx = elapsed_ns[-1]
    cv = statistics.pstdev(elapsed_ns) / med
    rel_spread = (mx - mn) / med

    out = {
        "label": label,
        "ok": True,
        "n": n_pass,
        "problem_id": pid,
        "wall_seconds": round(wall, 2),
        "median_ms": round(med / 1e6, 2),
        "p10_ms": round(p10 / 1e6, 2),
        "p25_ms": round(p25 / 1e6, 2),
        "p75_ms": round(p75 / 1e6, 2),
        "p90_ms": round(p90 / 1e6, 2),
        "min_ms": round(mn / 1e6, 2),
        "max_ms": round(mx / 1e6, 2),
        "cv": round(cv, 3),
        "rel_spread": round(rel_spread, 3),
    }
    print(f"[probe:{label}] {json.dumps(out, indent=2)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--output-json", type=str, default=None,
                    help="If given, write the full result dict as JSON.")
    args = ap.parse_args()

    # Make sure PCH is built (so compile-time variance is small)
    os.environ.setdefault("PIE_COMPILE_SCRATCH", f"/tmp/{os.environ.get('USER', 'u')}_pie_scratch")
    os.makedirs(os.environ["PIE_COMPILE_SCRATCH"], exist_ok=True)
    if "PIE_PCH_INCLUDE_PATH" not in os.environ:
        os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
        print(f"[probe] built PCH at {os.environ['PIE_PCH_INCLUDE_PATH']}", flush=True)

    # Preload test cases (pickle fast-path if available)
    n_cases = preload_test_cases(os.environ.get(
        "PIE_TEST_CASE_DIR", "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"))
    print(f"[probe] preloaded {n_cases} test cases", flush=True)

    cpuset = os.environ.get("PIE_REWARD_CPUSET", "<unset>")
    workers = os.environ.get("VERL_REWARD_WORKERS", "64")
    print(f"[probe] PIE_REWARD_CPUSET={cpuset!r} VERL_REWARD_WORKERS={workers}",
          flush=True)

    result = run_probe(n=args.n, label=args.label)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[probe] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
