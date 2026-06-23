"""Diff the per-arm rollout dumps (compare_rollouts.py) into a side-by-side
comparison: per highest-headroom problem, how grpo@500 / gos@300 / rloo@500
rollouts differ in correctness, speedup distribution, and what the best rollout
actually does. Emits a markdown report + a per-problem speedup strip plot + the
single best correct rollout's code per (problem, arm).

Usage:
  python -m experiment11_code_optim.eval.compare_rollouts_aggregate \
    --out_dir /work/nvme/bgtv/sramasubramanian/cont-maxrl/rollout_compare
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path

ARMS = ["grpo", "gos", "rloo"]
COLOR = {"grpo": "#1f77b4", "gos": "#d62728", "rloo": "#2ca02c"}
PAIRS = "/work/nvme/bgtv/sramasubramanian/pie_rl_dataset/pairs_test.parquet"


def _ref_ratios():
    """(problem_id,src_id) -> known src/tgt + ceiling speedup, for context."""
    out = {}
    try:
        import pandas as pd
        df = pd.read_parquet(PAIRS)
        for r in df.itertuples():
            out[(r.problem_id, r.src_id)] = (float(r.our_speedup_usable),
                                             float(r.our_fastest_speedup),
                                             getattr(r, "tgt_id", ""))
    except Exception as e:
        print(f"[ref] skipped: {e}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_show", type=int, default=1, help="best rollouts to inline per arm")
    args = ap.parse_args()
    out = Path(args.out_dir)

    # load each arm's dump
    data = {}
    for f in sorted(glob.glob(str(out / "rollouts_*.json"))):
        arm = Path(f).name.split("_")[1]
        data[arm] = json.loads(open(f).read())
    if not data:
        print(f"no rollouts_*.json under {out}"); return 2

    ref = _ref_ratios()
    # problem order = by descending known ratio
    probs = []
    for arm in data:
        for rec in data[arm]:
            probs.append((rec["problem_id"], rec["src_id"]))
    probs = sorted(set(probs), key=lambda ps: -ref.get(ps, (0, 0, ""))[0])

    def rec_for(arm, ps):
        for rec in data.get(arm, []):
            if (rec["problem_id"], rec["src_id"]) == ps:
                return rec
        return None

    md = []
    md.append("# Rollout comparison — GRPO@500 vs GOS@300 vs RLOO@500\n")
    md.append("Highest src/tgt-ratio test problems, n=32 rollouts/arm, temp=1.0. "
              "Speedup = gem5 src/cand on the biggest usable case (the eval reward); "
              "correctness gated on ALL usable cases. `max_correct` = fastest rollout "
              "that passes every case.\n")

    # ---- summary table ----
    md.append("## Summary\n")
    md.append("| problem | known src/tgt | ceiling | arm | pass | best@32 | max_correct | mean reward |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ps in probs:
        kr, cl, tgt = ref.get(ps, (0, 0, ""))
        for ai, arm in enumerate(ARMS):
            rec = rec_for(arm, ps)
            head = f"{ps[0]}/{ps[1]}" if ai == 0 else ""
            kc = f"{kr:.1f}×" if ai == 0 else ""
            cc = f"{cl:.1f}×" if ai == 0 else ""
            if rec is None:
                md.append(f"| {head} | {kc} | {cc} | {arm} | – | – | – | – |"); continue
            md.append(f"| {head} | {kc} | {cc} | **{arm}** | {rec['n_pass']}/{rec['n']} | "
                      f"{rec['best_at_n']:.2f}× | {rec['max_correct_speedup']:.2f}× | "
                      f"{rec['mean_score']:.2f} |")
    md.append("")

    # ---- per-problem detail ----
    for ps in probs:
        kr, cl, tgt = ref.get(ps, (0, 0, ""))
        md.append(f"## {ps[0]} / {ps[1]}  (known src/tgt {kr:.1f}×, ceiling {cl:.1f}×, human tgt {tgt})\n")
        for arm in ARMS:
            rec = rec_for(arm, ps)
            if rec is None:
                md.append(f"### {arm}: (missing)\n"); continue
            ro = rec["rollouts"]
            sp = [r["speedup"] for r in ro]
            corr = [r for r in ro if r["acc"] == 1.0]
            # failure reason histogram for the non-passing rollouts
            fails = {}
            for r in ro:
                if r["acc"] != 1.0:
                    tag = (r["reason"] or "").split(":")[0][:32] or "?"
                    fails[tag] = fails.get(tag, 0) + 1
            fail_str = ", ".join(f"{k}×{v}" for k, v in sorted(fails.items(), key=lambda x: -x[1])) or "—"
            md.append(f"### {arm} @{rec['step']}  —  pass {rec['n_pass']}/{rec['n']}, "
                      f"best@32 {rec['best_at_n']:.2f}×, max_correct {rec['max_correct_speedup']:.2f}×")
            md.append(f"- correct-rollout speedups: "
                      + (", ".join(f"{r['speedup']:.2f}×" for r in corr[:10]) + (" …" if len(corr) > 10 else "")
                         if corr else "_none correct_"))
            md.append(f"- failure modes (of {rec['n']-rec['n_pass']} non-passing): {fail_str}")
            # inline the best correct rollout's code
            best = corr[0] if corr else (ro[0] if ro else None)
            if best is not None:
                code = best.get("code") or "(no code extracted)"
                tag = f"{best['speedup']:.2f}× correct" if best["acc"] == 1.0 else f"FAIL ({best['reason'][:60]})"
                md.append(f"- best rollout ({tag}):\n")
                md.append("```cpp")
                md.append(code.strip()[:2400])
                md.append("```\n")
                # also dump full best code to a file
                cf = out / f"best_{ps[0]}_{ps[1]}_{arm}.cpp"
                cf.write_text((best.get("code") or "") + "\n")
        md.append("")

    rep = out / "COMPARISON.md"
    rep.write_text("\n".join(md))
    print(f"[report] -> {rep}")

    # ---- strip plot: 32 speedups per arm, one panel per problem ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        npr = len(probs)
        fig, axes = plt.subplots(1, npr, figsize=(5.0 * npr, 5.2), squeeze=False)
        for col, ps in enumerate(probs):
            ax = axes[0][col]
            kr, cl, _ = ref.get(ps, (0, 0, ""))
            for xi, arm in enumerate(ARMS):
                rec = rec_for(arm, ps)
                if rec is None:
                    continue
                ys = [r["speedup"] for r in rec["rollouts"]]
                xs = [xi + (h - len(ys)/2) * 0.012 for h in range(len(ys))]
                cols_pts = [COLOR[arm] if r["acc"] == 1.0 else "#cccccc" for r in rec["rollouts"]]
                ax.scatter(xs, ys, s=22, c=cols_pts, edgecolors="none", alpha=0.8)
                mc = rec["max_correct_speedup"]
                ax.hlines(mc, xi-0.3, xi+0.3, color=COLOR[arm], lw=2)
            ax.axhline(kr, color="k", ls="--", lw=1, alpha=0.6, label=f"known tgt {kr:.1f}×")
            ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
            ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a.upper() for a in ARMS])
            ax.set_yscale("log"); ax.set_ylabel("gem5 speedup (×, log)")
            ax.set_title(f"{ps[0]}/{ps[1]}  (ceiling {cl:.1f}×)")
            ax.grid(alpha=0.3, axis="y"); ax.legend(loc="upper right", fontsize=8)
        fig.suptitle("PIE highest-headroom test problems — 32 rollouts/arm "
                     "(grey = fails correctness; bar = best correct speedup)", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = out / "rollout_compare.png"; fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"[plot] -> {p}")
    except Exception as e:
        print(f"[plot] skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
