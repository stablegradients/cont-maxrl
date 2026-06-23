"""Plots for the final RL datasets (pie_rl_dataset), per split:

  (1) corr_vs_K_<split>.png  — subsample fidelity vs K. For each program, full
      agg = sum of ALL its timed per-tc ticks; K-estimate = (n/K)*sum(K random
      cases). Sweep K; plot log-log r AND within-±2%/±5% band fraction vs K, for
      src and tgt. (Bands are the honest predictive metric; r is inflated by shared
      terms.) High fidelity reflects within-program per-case uniformity (documented).

  (2) vs_pie_agg_<split>.png — our gem5 agg vs PIE agg_runtime, src+tgt scatter +
      ratio histogram. TEST aligns index-for-index on PIE's released `tests` indices
      (fair) and adds a per-test-case our-vs-tc2time panel. TRAIN/VAL sum all timed
      cases vs PIE's smaller unreleased subset (labelled caveat; NO circular rescale).

Usage: python rl_dataset_plots.py [--splits train,val,test]
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = "/work/nvme/bgtv/sramasubramanian/pie_rl_dataset"
OUTDIR = "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/docs/plots/rl_dataset"
PS = 1e12
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
NDRAW = 5     # random subsample draws averaged per (program, K)


def load_programs(split):
    """unique (pid,sid)->{ticks(list), per(dict idx->tick), pie_agg, role, tests, tc2time}."""
    df = pd.read_parquet(f"{DS}/pairs_{split}.parquet")
    P = {}
    tests_by_pid = {}
    for r in df.itertuples():
        if split == "test" and r.problem_id not in tests_by_pid and r.pie_tests_case_ids:
            tests_by_pid[r.problem_id] = set(int(i) for i in json.loads(r.pie_tests_case_ids))
        for role, sid, tk, agg, t2t in (
            ("src", r.src_id, r.gem5_src_per_tc_ticks, r.pie_src_agg_runtime,
             getattr(r, "pie_src_code_tc2time", None)),
            ("tgt", r.tgt_id, r.gem5_tgt_per_tc_ticks, r.pie_tgt_agg_runtime,
             getattr(r, "pie_tgt_code_tc2time", None))):
            key = (r.problem_id, sid, role)
            if key in P:
                continue
            per = {int(k): int(v) for k, v in json.loads(tk).items()} if tk else {}
            if not per:
                continue
            P[key] = {"per": per, "ticks": list(per.values()), "pie_agg": agg,
                      "tc2time": (json.loads(t2t) if isinstance(t2t, str) else None),
                      "pid": r.problem_id}
    return P, tests_by_pid


# ----------------------------- (1) corr vs K -----------------------------
def plot_corr_vs_K(split, P):
    rng = random.Random(0)
    res = {role: {"r": [], "w2": [], "w5": []} for role in ("src", "tgt")}
    for role in ("src", "tgt"):
        # input-sensitive programs only (n>=10); flat-uniform progs trivially exact
        progs = [v["ticks"] for (p, s, ro), v in P.items() if ro == role and len(v["ticks"]) >= 10]
        for K in KS:
            xs, ys = [], []
            for ticks in progs:
                n = len(ticks)
                if n < K:
                    continue
                full = sum(ticks) / PS
                est = np.mean([(n / K) * sum(rng.sample(ticks, K)) for _ in range(NDRAW)]) / PS
                xs.append(full); ys.append(est)
            if len(xs) < 3:
                res[role]["r"].append(np.nan); res[role]["w2"].append(np.nan); res[role]["w5"].append(np.nan); continue
            xs, ys = np.array(xs), np.array(ys)
            rel = np.abs(ys - xs) / xs
            res[role]["r"].append(np.corrcoef(np.log(xs), np.log(ys))[0, 1])
            res[role]["w2"].append(100 * np.mean(rel <= 0.02))
            res[role]["w5"].append(100 * np.mean(rel <= 0.05))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    col = {"src": "#1f77b4", "tgt": "#d62728"}
    for role in ("src", "tgt"):
        axes[0].plot(KS, res[role]["r"], "o-", color=col[role], label=f"{role}")
    axes[0].set_xlabel("K (subsampled cases)"); axes[0].set_ylabel("r (log-log: K-estimate vs full agg)")
    axes[0].set_xscale("log"); axes[0].set_title(f"{split}: subsample correlation vs K")
    axes[0].grid(alpha=0.25, which="both"); axes[0].legend()
    axes[0].set_ylim(min(0.9, np.nanmin([min(res[r]['r']) for r in res])) - 0.01, 1.001)
    for role in ("src", "tgt"):
        axes[1].plot(KS, res[role]["w5"], "o-", color=col[role], label=f"{role} within ±5%")
        axes[1].plot(KS, res[role]["w2"], "s--", color=col[role], alpha=0.6, label=f"{role} within ±2%")
    axes[1].set_xlabel("K (subsampled cases)"); axes[1].set_ylabel("% programs within band")
    axes[1].set_xscale("log"); axes[1].set_title(f"{split}: estimate fidelity vs K (n≥10-case progs)")
    axes[1].grid(alpha=0.25, which="both"); axes[1].legend(fontsize=8); axes[1].set_ylim(0, 101)
    ntr = sum(1 for (p, s, ro), v in P.items() if ro == "src" and len(v["ticks"]) >= 10)
    fig.suptitle(f"Subsample fidelity vs K — {split} (src n≈{ntr} input-sensitive programs, "
                 f"{NDRAW} draws/avg)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTDIR, exist_ok=True)
    p = f"{OUTDIR}/corr_vs_K_{split}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    # K=5,10 readout
    i5, i10 = KS.index(5), KS.index(10)
    print(f"  {split} corr_vs_K: src K=5 r={res['src']['r'][i5]:.4f} ±5%={res['src']['w5'][i5]:.0f}% | "
          f"K=10 r={res['src']['r'][i10]:.4f} ±5%={res['src']['w5'][i10]:.0f}%  -> {p}")


# ----------------------------- (2) vs PIE agg -----------------------------
def plot_vs_pie(split, P, tests_by_pid):
    aligned = (split == "test")
    ncol = 3 if aligned else 2
    fig, axes = plt.subplots(2, ncol, figsize=(5.3 * ncol, 9.2), squeeze=False)
    for ri, role in enumerate(("src", "tgt")):
        ours, theirs, ptc_o, ptc_t = [], [], [], []
        for (pid, sid, ro), v in P.items():
            if ro != role or v["pie_agg"] in (None,) or v["pie_agg"] <= 0:
                continue
            per = v["per"]
            if aligned and v["tc2time"] and pid in tests_by_pid:
                idxs = [i for i in tests_by_pid[pid] if i in per]
                if len(idxs) < 0.8 * len(tests_by_pid[pid]):
                    continue
                o = sum(per[i] for i in idxs) / PS
                for i in idxs:
                    if str(i) in v["tc2time"]:
                        ptc_o.append(per[i] / PS); ptc_t.append(float(v["tc2time"][str(i)]))
            else:
                o = sum(per.values()) / PS
            if o > 0:
                ours.append(o); theirs.append(float(v["pie_agg"]))
        ours, theirs = np.array(ours), np.array(theirs)
        if len(ours) == 0:
            for ci in range(ncol):
                axes[ri][ci].text(0.5, 0.5, f"no {role}", ha="center"); continue
        ratios = ours / theirs
        r = np.corrcoef(np.log(ours), np.log(theirs))[0, 1]
        r_lin = np.corrcoef(ours, theirs)[0, 1]

        ax = axes[ri][0]
        ax.scatter(theirs, ours, s=7, alpha=0.3, c="#2ca02c", edgecolor="none")
        lim = [min(theirs.min(), ours.min()), max(theirs.max(), ours.max())]
        ax.plot(lim, lim, "k--", lw=0.9, label="y=x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"PIE {role}_agg_runtime [s]"); ax.set_ylabel(f"our gem5 agg, {role} [s]")
        cav = "aligned on PIE `tests` idx (fair)" if aligned else \
              "we sum ALL timed cases vs PIE's smaller unreleased subset"
        ax.set_title(f"{split}/{role.upper()} n={len(ours)} | r(ll)={r:.4f} r(lin)={r_lin:.4f}\n{cav}",
                     fontsize=8.5)
        ax.legend(fontsize=7); ax.grid(alpha=0.2, which="both")

        ax = axes[ri][1]
        ax.hist(np.log10(ratios), bins=60, color="#9467bd", alpha=0.85)
        ax.axvline(0, color="k", lw=1, label="our=PIE")
        ax.axvline(np.log10(np.median(ratios)), color="r", lw=1.2, label=f"median {np.median(ratios):.3f}x")
        ax.set_xlabel("log10(our / PIE)"); ax.set_ylabel(f"# {role} programs")
        ax.set_title(f"{role.upper()} ratio | median {np.median(ratios):.3f}x", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

        if aligned:
            ax = axes[ri][2]
            a, b = np.array(ptc_o), np.array(ptc_t)
            m = (a > 0) & (b > 0); a, b = a[m], b[m]
            if len(a) >= 2:
                rr = np.corrcoef(np.log(a), np.log(b))[0, 1]
                ax.scatter(b, a, s=3, alpha=0.12, c="#d62728", edgecolor="none")
                lim = [min(a.min(), b.min()), max(a.max(), b.max())]
                ax.plot(lim, lim, "k--", lw=0.8)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel(f"PIE {role} tc2time [s]"); ax.set_ylabel(f"our per-tc, {role} [s]")
                ax.set_title(f"{role.upper()} per-test-case | n={len(a)} | r(ll)={rr:.4f}", fontsize=8.5)
                ax.grid(alpha=0.2, which="both")
        print(f"  {split}/{role}: n={len(ours)} median our/PIE={np.median(ratios):.3f} r(ll)={r:.4f}")
    sub = ("aligned index-for-index on PIE's benchmark cases (fair)" if aligned else
           "median ~1.0 = per-case match; any offset is case-COUNT (we keep ~18-23 more), not a timing gap")
    fig.suptitle(f"Our gem5 agg vs PIE agg_runtime — {split} (rows SRC/TGT)\n{sub}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTDIR, exist_ok=True)
    p = f"{OUTDIR}/vs_pie_agg_{split}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"  -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()
    for split in args.splits.split(","):
        P, tests = load_programs(split)
        print(f"{split}: {len(P)} unique role-programs")
        plot_corr_vs_K(split, P)
        plot_vs_pie(split, P, tests)


if __name__ == "__main__":
    main()
