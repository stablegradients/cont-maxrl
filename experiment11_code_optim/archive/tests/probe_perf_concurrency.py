"""Gate 2 probe — perf concurrency invariance.

64 worker threads × 10 back-to-back grades per worker = 640 measurements,
all of the same fixed-work C++ program. Spec §5 Gate 2 target:
rel_spread <= 1e-3 across all 640 (10x looser than Gate 1 to allow for
inter-process variance).

Spec uses ProcessPoolExecutor; we use ThreadPoolExecutor to match the
actual reward path's executor type (grade_dedup_batch uses
ThreadPoolExecutor). The HW counter is per-PID so each grade's child
gets its own counter regardless of which thread launches it.

Each worker burns warm-up grades first; the steady-state aggregate is
the production-relevant property.

Usage:
    PIE_BACKEND=perf PYTHONPATH=. python experiment11_code_optim/tests/probe_perf_concurrency.py \
        --n-workers 64 --k 10 --warmup 5 \
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
    _compile, _run_and_compare_perf, MEASUREMENT_BACKEND, setup_pch, _scratch_dir,
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


def load_first_test_case(pid: str) -> tuple[str, str]:
    cdir = f"/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases/{pid}"
    with open(f"{cdir}/input.0.txt") as f:
        inp = f.read()
    with open(f"{cdir}/output.0.txt") as f:
        exp = f.read()
    return inp, exp


def worker(args: tuple[int, str, str, str, str, int, int]) -> dict:
    tid, code, pch, inp, exp, k, warmup = args
    compile_dir = _scratch_dir() or tempfile.gettempdir()
    td = tempfile.mkdtemp(dir=compile_dir, prefix="pie_g2_")
    try:
        exe, err = _compile(code, td, pch_include_path=pch)
        if exe is None:
            return {"tid": tid, "ok": False, "err": err}
        samples: list[int] = []
        for _ in range(warmup + k):
            passed, c = _run_and_compare_perf(exe, inp, exp, 10.0)
            if not passed:
                return {"tid": tid, "ok": False, "err": f"gate fail c={c}"}
            samples.append(c)
        return {"tid": tid, "ok": True, "samples_all": samples,
                "samples_steady": samples[warmup:]}
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def summarize(label: str, samples: list[int]) -> dict:
    if not samples:
        return {"label": label, "n": 0, "ok": False}
    s = sorted(samples)
    med = s[len(s) // 2]
    rel_spread = (s[-1] - s[0]) / med if med else float("inf")
    return {
        "label": label,
        "n": len(samples),
        "ok": True,
        "median": med,
        "min": s[0],
        "max": s[-1],
        "p10": s[max(0, len(s) // 10)],
        "p25": s[max(0, len(s) // 4)],
        "p75": s[min(len(s) - 1, 3 * len(s) // 4)],
        "p90": s[min(len(s) - 1, 9 * len(s) // 10)],
        "rel_spread": round(rel_spread, 6),
        "cv": round(statistics.pstdev(samples) / med, 6) if med else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=64)
    ap.add_argument("--k", type=int, default=10, help="Steady-state grades per worker")
    ap.add_argument("--warmup", type=int, default=5, help="Burn this many per worker")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    if MEASUREMENT_BACKEND != "perf":
        print(f"[gate2] WARNING: MEASUREMENT_BACKEND={MEASUREMENT_BACKEND!r}, "
              f"not 'perf'. Set PIE_BACKEND=perf before running.", flush=True)

    pid, code = load_known_good()
    inp, exp = load_first_test_case(pid)
    pch = os.environ.get("PIE_PCH_INCLUDE_PATH") or setup_pch()
    os.environ["PIE_PCH_INCLUDE_PATH"] = pch
    print(f"[gate2] target=({pid}) n_workers={args.n_workers} k={args.k} "
          f"warmup={args.warmup} backend={MEASUREMENT_BACKEND}", flush=True)

    workloads = [(i, code, pch, inp, exp, args.k, args.warmup)
                 for i in range(args.n_workers)]
    t0 = time.monotonic()
    per_worker: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futs = [pool.submit(worker, w) for w in workloads]
        for f in as_completed(futs):
            per_worker.append(f.result())
    wall = time.monotonic() - t0

    pooled_steady: list[int] = []
    pooled_all: list[int] = []
    n_ok = 0
    for w in per_worker:
        if w.get("ok"):
            n_ok += 1
            pooled_all.extend(w["samples_all"])
            pooled_steady.extend(w["samples_steady"])

    report = {
        "label": "gate2_perf_concurrency",
        "wall_seconds": round(wall, 2),
        "backend": MEASUREMENT_BACKEND,
        "node": os.uname().nodename,
        "n_workers": args.n_workers,
        "n_workers_ok": n_ok,
        "k_per_worker": args.k,
        "warmup_per_worker": args.warmup,
        "stats_all": summarize("all", pooled_all),
        "stats_steady": summarize("steady", pooled_steady),
        "target_rel_spread": 1e-3,
        "verdict_steady": (
            "PASS" if summarize("steady", pooled_steady).get("rel_spread", 1) <= 1e-3
            else "FAIL"
        ),
    }
    print(f"[gate2] {json.dumps(report, indent=2)}", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({**report, "per_worker": per_worker}, f, indent=2)


if __name__ == "__main__":
    main()
