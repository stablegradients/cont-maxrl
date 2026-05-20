"""Aggregate 4 GO/NO-GO shard JSONs into a single verdict.

Run after `go_no_go.py` has been launched in 4-way data-parallel mode
(see eval/run_go_no_go_dp4.sh). Reads shard0.json..shard3.json, combines
the per-problem detail, and reports the combined verdict using the same
≥8/16 threshold and 20% GO floor as the single-shot script.

Usage:
    python -m experiment11_code_optim.eval.aggregate_go_no_go_shards \\
        --shard_glob '/work/nvme/.../sft_ckpts/.../go_no_go_shard*.json' \\
        --output_json /work/nvme/.../sft_ckpts/.../go_no_go.json
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_glob", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--pass_threshold", type=int, default=8)
    ap.add_argument("--go_fraction", type=float, default=0.20)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shard_glob))
    if not paths:
        print(f"FATAL: no shards matched {args.shard_glob!r}", file=sys.stderr)
        return 2

    per_problem_passes: list[int] = []
    per_problem_avg_score: list[float] = []
    per_problem_pids: list[str] = []
    n_completions = None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if n_completions is None:
            n_completions = d["n_completions"]
        elif d["n_completions"] != n_completions:
            print(f"FATAL: shard {p} has n_completions={d['n_completions']} "
                  f"!= {n_completions}", file=sys.stderr)
            return 2
        per_problem_passes.extend(d["per_problem_passes"])
        per_problem_avg_score.extend(d["per_problem_avg_score"])
        per_problem_pids.extend(d.get("per_problem_pids", ["?"] * len(d["per_problem_passes"])))

    n_problems = len(per_problem_passes)
    n_passing_problems = sum(1 for c in per_problem_passes if c >= args.pass_threshold)
    fraction = n_passing_problems / max(n_problems, 1)
    verdict = "GO" if fraction >= args.go_fraction else "NO-GO"

    hist = Counter(per_problem_passes)

    print()
    print("=" * 60)
    print(f"§9 GO/NO-GO RESULT (aggregated from {len(paths)} shards): {verdict}")
    print("=" * 60)
    print(f"  Problems passing threshold (≥{args.pass_threshold}/{n_completions}): "
          f"{n_passing_problems}/{n_problems} = {100*fraction:.1f}%")
    print(f"  Required: ≥ {100*args.go_fraction:.0f}%")
    print()
    print(f"  Pass-count distribution (passes/{n_completions}: # problems):")
    for k in range(n_completions + 1):
        c = hist.get(k, 0)
        if c == 0:
            continue
        bar = "#" * min(60, c)
        print(f"    {k:>2}/{n_completions}  ({c:>3}) {bar}")
    print()
    n_pass_at_least_1 = sum(1 for c in per_problem_passes if c >= 1)
    mean_score = sum(per_problem_avg_score) / max(n_problems, 1)
    print(f"  Mean per-problem avg score: {mean_score:.3f}")
    print(f"  Problems with ≥1 pass:       {n_pass_at_least_1}/{n_problems} = "
          f"{100*n_pass_at_least_1/max(n_problems,1):.1f}%")

    with open(args.output_json, "w") as f:
        json.dump({
            "verdict": verdict,
            "n_problems": n_problems,
            "n_completions": n_completions,
            "pass_threshold": args.pass_threshold,
            "go_fraction": args.go_fraction,
            "n_passing_problems": n_passing_problems,
            "fraction_passing": fraction,
            "per_problem_passes": per_problem_passes,
            "per_problem_avg_score": per_problem_avg_score,
            "per_problem_pids": per_problem_pids,
            "pass_count_histogram": dict(hist),
            "n_shards_aggregated": len(paths),
            "shard_paths": paths,
        }, f, indent=2)
    print(f"  -> wrote {args.output_json}")

    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
