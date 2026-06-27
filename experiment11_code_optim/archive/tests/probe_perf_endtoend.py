"""Gate 4 probe — end-to-end stability through compute_scores_batch.

Load 16 known-good tgt_code rollouts for ONE problem_id from val.jsonl
(if fewer than 16 distinct, repeat to fill). Pretend they're a rollout
batch: build reward_inputs and run compute_scores_batch through 10
independent trials.

Per program report rel_spread of the FINAL SCORE (the percentile output
from the perf-units reference table). Target: every program's score
rel_spread <= 1e-3 AND no program returns 0.5 (the missing-reference stub).

This is the load-bearing gate. If Gates 1-3 pass but Gate 4 fails, the
bug is in the chain compute_scores_batch → reference lookup → score, not
in the measurement layer.

Requires reference_table_perf to be built (PIE_REFERENCE_TABLE_DIR points
at it). The manifest backend check will refuse to load if it's still
walltime.

Usage:
    PIE_BACKEND=perf PIE_REFERENCE_TABLE_DIR=.../reference_table_perf \
    PIE_REFERENCE_TABLE_PICKLE=.../reference_table_perf.pkl \
    PYTHONPATH=. python experiment11_code_optim/tests/probe_perf_endtoend.py \
        --n-trials 10 --pid p02587 --output-json /path/to/out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiment11_code_optim.measurement.core import (  # noqa: E402
    MEASUREMENT_BACKEND, setup_pch,
)


def load_codes_for_pid(pid: str, n_target: int) -> list[str]:
    """Pull up to n_target distinct tgt_code rollouts for the given pid
    from any of train/val/test jsonls. Falls back to repeating the same
    one if not enough distinct entries exist."""
    candidates: list[str] = []
    seen: set[str] = set()
    for split in ("val", "train", "test"):
        path = f"/work/nvme/bgmo/sramasubramanian/pie_data/raw/{split}.jsonl"
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("problem_id") != pid:
                    continue
                code = rec.get("tgt_code")
                if not code or code in seen:
                    continue
                seen.add(code)
                candidates.append(code)
                if len(candidates) >= n_target:
                    return candidates
    if not candidates:
        return []
    while len(candidates) < n_target:
        candidates.append(candidates[len(candidates) % len(candidates)])
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--n-rollouts", type=int, default=16)
    ap.add_argument("--pid", type=str, default="p02587")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    if MEASUREMENT_BACKEND != "perf":
        print(f"[gate4] WARNING: MEASUREMENT_BACKEND={MEASUREMENT_BACKEND!r}", flush=True)

    # Ensure PCH built (so compile cost doesn't dominate first trial).
    pch = os.environ.get("PIE_PCH_INCLUDE_PATH") or setup_pch()
    os.environ["PIE_PCH_INCLUDE_PATH"] = pch

    # Resolve ref-table dir & confirm it's the perf-units one.
    ref_dir = os.environ.get(
        "PIE_REFERENCE_TABLE_DIR",
        "/work/nvme/bgmo/sramasubramanian/pie_data/reference_table_perf",
    )
    if not os.path.isdir(ref_dir):
        print(f"[gate4] FAIL: reference_table_dir does not exist: {ref_dir}", flush=True)
        sys.exit(1)
    manifest_path = os.path.join(ref_dir, "MANIFEST.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            m = json.load(f)
        print(f"[gate4] reference table manifest: backend={m.get('backend')}, "
              f"K_cap={m.get('K_cap')}", flush=True)
        if m.get("backend") != "perf":
            print(f"[gate4] FAIL: ref table not perf-units (backend="
                  f"{m.get('backend')})", flush=True)
            sys.exit(1)

    codes = load_codes_for_pid(args.pid, args.n_rollouts)
    if not codes:
        print(f"[gate4] FAIL: no codes found for pid={args.pid}", flush=True)
        sys.exit(1)
    print(f"[gate4] loaded {len(codes)} distinct rollouts for pid={args.pid}", flush=True)

    from experiment11_code_optim.reward.pie_reward_fn import (
        compute_scores_batch, preload_test_cases,
    )
    preload_test_cases(os.environ.get(
        "PIE_TEST_CASE_DIR", "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"))

    # Build reward_inputs — each rollout fenced with ```cpp.
    reward_inputs = []
    for code in codes:
        sol = f"```cpp\n{code}\n```"
        gt = json.dumps({"problem_id": args.pid, "split": "val"})
        reward_inputs.append({"solution_str": sol, "ground_truth": gt})

    # Run n_trials independent grading passes. Each pass is one
    # compute_scores_batch call; the dedup layer collapses identical codes
    # internally, but here we want to test stability of the FULL chain
    # against natural per-call variation.
    trials: list[list[dict]] = []
    t0 = time.monotonic()
    for t in range(args.n_trials):
        res = compute_scores_batch(
            reward_inputs,
            test_case_dir=os.environ.get(
                "PIE_TEST_CASE_DIR",
                "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases",
            ),
            reference_table_dir=ref_dir,
        )
        trials.append(res)
        print(f"[gate4] trial {t+1}/{args.n_trials} acc={[r['acc'] for r in res]}",
              flush=True)
    wall = time.monotonic() - t0

    # Check ref-table presence directly — 0.5 can be a LEGITIMATE percentile
    # (midpoint of a reference distribution like n=2 or any case where
    # bisect_left = bisect_right at the median). The spec's "no spurious 0.5
    # stub" check is really "the (split, pid) entry has refs"; we verify that
    # via the loaded cache rather than inferring from the score.
    from experiment11_code_optim.reward.pie_reward_fn import (
        _REF_TABLE_CACHE,
    )
    has_ref = bool(_REF_TABLE_CACHE.get(("val", args.pid)))

    # Per-rollout stats across trials
    per_rollout: list[dict] = []
    n_fail = 0
    n_stub = 0
    for i in range(len(reward_inputs)):
        scores = [trials[t][i]["score"] for t in range(args.n_trials)]
        accs = [trials[t][i]["acc"] for t in range(args.n_trials)]
        if all(a == 0.0 for a in accs):
            verdict_reason = "all-trial gate fail"
            n_fail += 1
        elif not has_ref and all(s == 0.5 for s in scores):
            verdict_reason = "stub-0.5 (no reference data)"
            n_stub += 1
        else:
            verdict_reason = "ok"
        s_sorted = sorted(scores)
        med = s_sorted[len(s_sorted) // 2]
        rel_spread = (s_sorted[-1] - s_sorted[0]) / med if med else 0.0
        per_rollout.append({
            "i": i,
            "scores": scores,
            "median_score": med,
            "rel_spread": round(rel_spread, 6),
            "stdev_score": round(statistics.pstdev(scores), 6),
            "accs": accs,
            "verdict_reason": verdict_reason,
        })

    # Verdict aggregation
    max_rel_spread = max((r["rel_spread"] for r in per_rollout), default=0.0)
    pass_target = 1e-3
    verdict = "PASS" if max_rel_spread <= pass_target and n_stub == 0 else "FAIL"

    report = {
        "label": "gate4_perf_endtoend",
        "backend": MEASUREMENT_BACKEND,
        "node": os.uname().nodename,
        "n_trials": args.n_trials,
        "n_rollouts": len(reward_inputs),
        "wall_seconds": round(wall, 2),
        "per_rollout": per_rollout,
        "max_rel_spread": round(max_rel_spread, 6),
        "n_rollouts_with_stub_score": n_stub,
        "n_rollouts_all_trial_fail": n_fail,
        "target_max_rel_spread": pass_target,
        "verdict": verdict,
    }
    print(f"[gate4] max_rel_spread={max_rel_spread:.6f} (target <= {pass_target})  "
          f"stub_count={n_stub}  verdict={verdict}", flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
