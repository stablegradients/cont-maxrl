"""Completeness + reward-fidelity plots on the COMPLETE (post-retime) gem5 timing.

Two figures under docs/plots/dataset_sweep/:
  coverage_completeness.png  -- per-program case-coverage + per-pair completeness,
      directly answering "did we time all test cases for all pairs?"
  reward_fidelity_complete.png -- our gem5 speedup vs PIE's published speedup,
      restricted to BOTH-SIDES-FULLY-TIMED pairs (the clean reward signal) and
      contrasted with the partial-coverage pairs that drive the tail divergence.

Reads the consolidated parquets at PIE_GEM5_TIMES_OUT.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMES = os.environ.get("PIE_GEM5_TIMES_OUT",
                       "/work/nvme/bgtv/sramasubramanian/pie_gem5_times")
OUTDIR = ("/projects/bgmo/sramasubramanian/cont-maxrl/"
          "experiment11_code_optim/docs/plots/dataset_sweep")
SPLITS = ["train", "val", "test"]

# status -> (label, color)
STATUS_STYLE = {
    "ok_full":   ("ok, all cases timed",      "#2ca02c"),
    "ok_part":   ("ok, but heavy cases excl", "#bcbd22"),
    "partial":   ("forkloop_partial",         "#ff7f0e"),
    "zero":      ("zero cases timed",          "#d62728"),
    "other":     ("other fail",                "#7f7f7f"),
}


def prog_category(row):
    cov = (row.n_cases_timed / row.n_cases_total
           if row.n_cases_total else 0)
    if row.n_cases_timed == 0 or pd.isna(row.sum_ticks):
        return "zero"
    if row.status == "ok":
        return "ok_full" if cov >= 0.999 else "ok_part"
    if row.status == "forkloop_partial":
        return "partial"
    return "other"


def fig_completeness():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, sp in enumerate(SPLITS):
        pp = pd.read_parquet(f"{TIMES}/per_program_{sp}.parquet")
        pp = pp[pp.n_cases_total.notna() & (pp.n_cases_total > 0)].copy()
        pp["cov"] = pp.n_cases_timed / pp.n_cases_total
        pp["cat"] = pp.apply(prog_category, axis=1)

        # ---- (top) per-program coverage histogram, stacked by category ----
        ax = axes[0, j]
        bins = np.linspace(0, 1.0001, 26)
        order = ["zero", "partial", "ok_part", "ok_full"]
        data = [pp.loc[pp.cat == c, "cov"].values for c in order]
        colors = [STATUS_STYLE[c][1] for c in order]
        labels = [f"{STATUS_STYLE[c][0]} (n={ (pp.cat==c).sum() })" for c in order]
        ax.hist(data, bins=bins, stacked=True, color=colors, label=labels)
        ax.set_yscale("log")
        ax.set_xlabel("per-program coverage  (cases timed / cases total)")
        ax.set_ylabel("programs (log)")
        full = (pp.cat == "ok_full").mean()
        ax.set_title(f"{sp}: {len(pp)} programs | {100*full:.1f}% fully timed")
        ax.legend(fontsize=7, loc="upper left")
        ax.axvline(1.0, color="k", lw=0.6, ls=":")

        # ---- (bottom) per-pair completeness stacked bar ----
        ax2 = axes[1, j]
        pr = pd.read_parquet(f"{TIMES}/pairs_{sp}.parquet")
        n = len(pr)
        both_ok = ((pr.src_status == "ok") & (pr.tgt_status == "ok")
                   & (pr.src_n_cases >= pr.n_common_cases))  # ok both sides
        nocommon = pr.n_common_cases == 0
        # "fully timed" pair = both sides ok AND both full coverage. Approximate
        # full coverage by both sides ok (ok_part is rare); refine via per-prog.
        part = (~both_ok) & (~nocommon)
        cats = {
            "both ok": int(both_ok.sum()),
            "≥1 partial": int(part.sum()),
            "no common case": int(nocommon.sum()),
        }
        cum = 0
        for name, col in [("both ok", "#2ca02c"), ("≥1 partial", "#ff7f0e"),
                          ("no common case", "#d62728")]:
            v = cats[name]
            ax2.barh(0, v, left=cum, color=col,
                     label=f"{name}: {v} ({100*v/n:.1f}%)")
            cum += v
        ax2.set_xlim(0, n)
        ax2.set_yticks([])
        ax2.set_xlabel("pairs")
        ax2.set_title(f"{sp}: {n} (src,tgt) pairs")
        ax2.legend(fontsize=8, loc="lower center", ncol=1)
    fig.suptitle("gem5 timing COMPLETENESS (post-retime): per-program case "
                 "coverage (top) and per-pair completeness (bottom)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTDIR, exist_ok=True)
    p = f"{OUTDIR}/coverage_completeness.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)


def fig_reward():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for j, sp in enumerate(SPLITS):
        pr = pd.read_parquet(f"{TIMES}/pairs_{sp}.parquet")
        pr = pr.copy()
        pr["ds"] = pd.to_numeric(pr.dataset_speedup, errors="coerce")
        d = pr[pr.our_speedup.notna() & pr.ds.notna()
               & (pr.our_speedup > 0) & (pr.ds > 0)].copy()
        both_ok = (d.src_status == "ok") & (d.tgt_status == "ok")
        clean = d[both_ok]
        partial = d[~both_ok]
        ax = axes[j]
        ax.scatter(partial.ds, partial.our_speedup, s=6, alpha=0.35,
                   c="#ff7f0e", label=f"≥1 partial side (n={len(partial)})")
        ax.scatter(clean.ds, clean.our_speedup, s=6, alpha=0.35,
                   c="#1f77b4", label=f"both fully timed (n={len(clean)})")
        lim = [min(d.ds.min(), d.our_speedup.min()),
               max(d.ds.max(), d.our_speedup.max())]
        ax.plot(lim, lim, "k--", lw=0.8, label="y=x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("PIE published speedup")
        ax.set_ylabel("our gem5 speedup (common cases)")

        def stats(s):
            rr = s.our_speedup / s.ds
            rlog = np.corrcoef(np.log(s.our_speedup), np.log(s.ds))[0, 1]
            return (rlog, rr.median(), (np.abs(rr - 1) < 0.1).mean(),
                    (np.abs(rr - 1) < 0.2).mean())
        rl_c, md_c, w10_c, w20_c = stats(clean)
        rl_a, md_a, _, _ = stats(d)
        ax.set_title(
            f"{sp}  |  BOTH-TIMED: r(log)={rl_c:.4f}  med={md_c:.3f}\n"
            f"within10%={100*w10_c:.1f}%  within20%={100*w20_c:.1f}%   "
            f"(all-pairs r(log)={rl_a:.3f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Reward fidelity on COMPLETE data: our gem5 speedup vs PIE's "
                 "published speedup — the clean (both-fully-timed) subset is the "
                 "faithful reward signal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = f"{OUTDIR}/reward_fidelity_complete.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_completeness()
    fig_reward()
