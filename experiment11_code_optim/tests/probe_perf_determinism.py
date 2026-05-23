"""Gate 1 probe — single-thread perf determinism.

Compile a known-good val.jsonl tgt_code once, then run it sequentially
50× through `_run_and_compare_perf`. Reports min/max/median/rel_spread.

Target: rel_spread <= 1e-4 (spec §5 Gate 1).

The first ~15 grades drift upward by a few percent (Popen libc paths,
fd pool, kernel mm bookkeeping warming gradually). Production worker
threads do thousands of grades each, so the steady-state determinism
is the production-relevant property. We report both:
  - rel_spread_all      — every sample
  - rel_spread_steady   — samples after the warm-up window

Usage:
    PIE_BACKEND=perf PYTHONPATH=. python experiment11_code_optim/tests/probe_perf_determinism.py \
        --n 50 --warmup 15 --output-json /path/to/out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiment11_code_optim.measurement.core import (  # noqa: E402
    _compile, _run_and_compare_perf, MEASUREMENT_BACKEND, setup_pch,
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
    """Read test 0 input + expected for the probe target."""
    cdir = f"/work/nvme/bgmo/sramasubramanian/pie_data/tests/{pid}"
    with open(f"{cdir}/input.0.txt") as f:
        inp = f.read()
    with open(f"{cdir}/output.0.txt") as f:
        exp = f.read()
    return inp, exp


def summarize(label: str, samples: list[int]) -> dict:
    s = sorted(samples)
    if not s:
        return {"label": label, "n": 0, "ok": False}
    med = s[len(s) // 2]
    rel_spread = (s[-1] - s[0]) / med if med else float("inf")
    return {
        "label": label,
        "n": len(s),
        "ok": True,
        "median": med,
        "min": s[0],
        "max": s[-1],
        "p10": s[max(0, len(s) // 10)],
        "p90": s[min(len(s) - 1, 9 * len(s) // 10)],
        "rel_spread": round(rel_spread, 6),
        "cv": round(statistics.pstdev(samples) / med, 6) if med else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Total samples")
    ap.add_argument("--warmup", type=int, default=15,
                    help="Burn this many leading samples before computing steady-state")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    if MEASUREMENT_BACKEND != "perf":
        print(f"[gate1] WARNING: MEASUREMENT_BACKEND={MEASUREMENT_BACKEND!r}, "
              f"not 'perf'. Set PIE_BACKEND=perf before running.", flush=True)

    pid, code = load_known_good()
    inp, exp = load_first_test_case(pid)
    print(f"[gate1] target=({pid}); first-test |input|={len(inp)} |expected|={len(exp)}",
          flush=True)

    pch = os.environ.get("PIE_PCH_INCLUDE_PATH") or setup_pch()
    os.environ["PIE_PCH_INCLUDE_PATH"] = pch

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(
        dir=os.environ.get("PIE_COMPILE_SCRATCH", None), prefix="pie_gate1_"
    ) as td:
        exe, err = _compile(code, td, pch_include_path=pch)
        if exe is None:
            print(f"[gate1] FAIL: compile failed: {err}", flush=True)
            sys.exit(1)

        samples: list[int] = []
        for i in range(args.n):
            passed, c = _run_and_compare_perf(exe, inp, exp, 10.0)
            if not passed:
                print(f"[gate1] FAIL: gate failed at i={i} count={c}", flush=True)
                sys.exit(1)
            samples.append(c)

    wall = time.monotonic() - t0

    s_all = summarize("all", samples)
    s_steady = summarize(f"steady (skip first {args.warmup})", samples[args.warmup:])
    s_steady_short = summarize("first_3", samples[:3])

    report = {
        "label": "gate1_perf_determinism",
        "wall_seconds": round(wall, 2),
        "backend": MEASUREMENT_BACKEND,
        "node": os.uname().nodename,
        "all_samples": samples,
        "stats_all": s_all,
        "stats_steady": s_steady,
        "stats_first_3": s_steady_short,
        "target_rel_spread": 1e-4,
        "verdict_steady": "PASS" if s_steady.get("rel_spread", 1) <= 1e-4 else "FAIL",
        "verdict_all": "PASS" if s_all.get("rel_spread", 1) <= 1e-4 else "FAIL",
    }
    print(f"[gate1] {json.dumps(report, indent=2)}", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
