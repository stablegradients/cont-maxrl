"""Aggregate the sharded best_reward@k test eval into per-arm curves + a plot.

Contiguous prompt-sharding means each prompt's full n=1024 completions live in
ONE shard, so each shard already holds that prompt's full-n best@k. This just
averages per-prompt best@k / pass@k across all 878 test prompts per arm, checks
coverage, and plots GRPO vs GOS vs RLOO. Flags incompleteness rather than
reporting partial numbers as final.

Usage:
  python -m experiment11_code_optim.eval.besteval_aggregate \
    --eval_root /work/nvme/bgtv/sramasubramanian/cont-maxrl/besteval \
    --runs grpo_step500 gos_step300 rloo_step500 --expect_prompts 878
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
from statistics import mean

K_DEFAULT = [1, 4, 16, 64, 256]
COLOR = {"grpo": "#1f77b4", "gos": "#d62728", "rloo": "#2ca02c", "base": "#7f7f7f"}


def load_run(run_dir: Path):
    recs, shards_done, shards_expected = [], 0, None
    for d in sorted(glob.glob(str(run_dir / "shard_*.done.json"))):
        shards_done += 1
        info = json.loads(open(d).read())
        if shards_expected is None and "n_shards" in info:
            shards_expected = info["n_shards"]
    seen = set()
    for f in sorted(glob.glob(str(run_dir / "shard_*.jsonl"))):
        if f.endswith(".gens.jsonl"):
            continue  # rollout-dump sidecars, not score records
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("problem_id"), r.get("src_id"))
            if key in seen:
                continue  # defensive dedup (of16/of128 overlap, re-run races)
            seen.add(key)
            recs.append(r)
    return recs, shards_done, shards_expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="<arm>_step<step> subdir names under eval_root")
    ap.add_argument("--k_values", type=int, nargs="+", default=K_DEFAULT)
    ap.add_argument("--expect_prompts", type=int, default=878)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.eval_root)
    out = Path(args.out or (root / "besteval_summary"))
    out.mkdir(parents=True, exist_ok=True)
    ks = args.k_values

    summary = {}
    for run in args.runs:
        arm = run.split("_step")[0]
        step = run.split("_step")[1] if "_step" in run else "?"
        recs, sd, se = load_run(root / run)
        npr = len(recs)
        complete = (npr >= args.expect_prompts) and (se is None or sd >= se)
        bestk = {str(k): mean([r["best_at_k"][str(k)] for r in recs]) for k in ks} if recs else {}
        passk = {str(k): mean([r["pass_at_k"][str(k)] for r in recs]) for k in ks} if recs else {}
        n_comp = recs[0]["n"] if recs else 0
        summary[run] = {"arm": arm, "step": step, "n_prompts": npr,
                        "expect_prompts": args.expect_prompts,
                        "shards_done": sd, "shards_expected": se,
                        "complete": complete, "n_completions": n_comp,
                        "best_at_k": bestk, "pass_at_k": passk}
        flag = "" if complete else "  <-- INCOMPLETE (partial; not final)"
        print(f"\n=== {run} (n={n_comp}, prompts={npr}/{args.expect_prompts}, "
              f"shards={sd}/{se}){flag} ===")
        print("  best_reward@k: " + "  ".join(f"k={k}:{bestk.get(str(k),0):.3f}" for k in ks))
        print("  pass@k:        " + "  ".join(f"k={k}:{passk.get(str(k),0):.3f}" for k in ks))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] -> {out}/summary.json")

    # plot best@k + pass@k
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        for run, s in summary.items():
            if not s["best_at_k"]:
                continue
            c = COLOR.get(s["arm"], None)
            lab = f"{s['arm']} @{s['step']}" + ("" if s["complete"] else " (partial)")
            ls = "-" if s["complete"] else "--"
            axes[0].plot(ks, [s["best_at_k"][str(k)] for k in ks], marker="o", color=c, ls=ls, label=lab)
            axes[1].plot(ks, [s["pass_at_k"][str(k)] for k in ks], marker="o", color=c, ls=ls, label=lab)
        for ax, t, yl in ((axes[0], "best_reward@k (gem5 speedup)", "E[max speedup over k]"),
                          (axes[1], "pass@k (correctness)", "pass@k")):
            ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
            ax.set_xlabel("k (rollouts)"); ax.set_ylabel(yl); ax.set_title(t)
            ax.grid(alpha=0.3); ax.legend()
        fig.suptitle("PIE test set — full-batch gem5 RL: GRPO vs GOS vs RLOO  "
                     "(n=1024/prompt, unbiased best@k, gem5 reward)", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = out / "besteval_test.png"; fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"[plot] -> {p}")
    except Exception as e:
        print(f"[plot] skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
