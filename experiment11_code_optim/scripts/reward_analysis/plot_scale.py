"""Plot + summarize the Step-4 scale result (reads gem5_scale_out/scale_summary.json).
Panels: absolute-K hist (src+tgt), speedup-K hist, our-vs-dataset speedup scatter (log-log),
K vs dataset-speedup, K vs n_tc, throughput/per-step text. Writes PNG + prints markdown."""
import json, sys, math, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = sys.argv[1] if len(sys.argv) > 1 else "/work/nvme/bgtv/sramasubramanian/gem5_scale_out/scale_summary.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/docs/gem5_native_dynamic/scale.png"
d = json.load(open(J)); s = d["summary"]; rows = d["rows"]
def clip(xs, lo, hi): return [min(max(x, lo), hi) for x in xs]

fig, ax = plt.subplots(2, 3, figsize=(18, 11))

Ksrc = [r["K_src"] for r in rows if r.get("K_src")]
Ktgt = [r["K_tgt"] for r in rows if r.get("K_tgt")]
Kabs = Ksrc + Ktgt
ax[0,0].hist(clip(Kabs, 0.9, 1.1), bins=60, color="#9467bd", alpha=0.85)
ax[0,0].axvline(1.0, color="k", ls="--", label="exact")
if Kabs: ax[0,0].axvline(statistics.median(Kabs), color="r", label=f"median={statistics.median(Kabs):.4f}")
ax[0,0].set_xlabel("K = our_agg / dataset_agg (per program-side)"); ax[0,0].set_ylabel("# sides")
ax[0,0].set_title(f"ABSOLUTE timing match (n={len(Kabs)})\nwithin 2%={s['abs_K_within_2pct']*100:.1f}%  within 5%={s['abs_K_within_5pct']*100:.1f}%")
ax[0,0].legend()

Ksp = [r["K_sp"] for r in rows if r.get("K_sp")]
ax[0,1].hist(clip(Ksp, 0.9, 1.1), bins=60, color="#2ca02c", alpha=0.85)
ax[0,1].axvline(1.0, color="k", ls="--")
if Ksp: ax[0,1].axvline(statistics.median(Ksp), color="r", label=f"median={statistics.median(Ksp):.4f}")
ax[0,1].set_xlabel("K_speedup = our_sp / dataset_sp"); ax[0,1].set_ylabel("# pairs")
ax[0,1].set_title(f"SPEEDUP match (n={len(Ksp)})\nwithin 2%={s['speedup_K_within_2pct']*100:.1f}%  within 5%={s['speedup_K_within_5pct']*100:.1f}%")
ax[0,1].legend()

pts = [(r["data_sp"], r["our_sp"]) for r in rows if r.get("our_sp") and r.get("data_sp")]
if pts:
    x = [a for a, b in pts]; y = [b for a, b in pts]
    ax[0,2].scatter(x, y, s=8, alpha=0.35, color="#1f77b4")
    lo, hi = min(x+y), max(x+y); ax[0,2].plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax[0,2].set_xscale("log"); ax[0,2].set_yscale("log")
    ax[0,2].set_xlabel("PIE dataset speedup"); ax[0,2].set_ylabel("our native-dynamic speedup")
    # log-log pearson
    lx = [math.log(a) for a in x]; ly = [math.log(b) for b in y]
    mx, my = sum(lx)/len(lx), sum(ly)/len(ly)
    num = sum((a-mx)*(b-my) for a, b in zip(lx, ly))
    den = math.sqrt(sum((a-mx)**2 for a in lx)*sum((b-my)**2 for b in ly))
    ax[0,2].set_title(f"our vs PIE speedup (n={len(pts)})\nPearson(loglog)={num/den:.4f}")
    ax[0,2].legend()

if Ksp and pts:
    ax[1,0].scatter([r["data_sp"] for r in rows if r.get("K_sp")], [r["K_sp"] for r in rows if r.get("K_sp")], s=8, alpha=0.35, color="#d62728")
    ax[1,0].axhline(1.0, color="k", ls="--"); ax[1,0].axhline(1.02, color="gray", ls=":"); ax[1,0].axhline(0.98, color="gray", ls=":")
    ax[1,0].set_xscale("log"); ax[1,0].set_xlabel("PIE dataset speedup"); ax[1,0].set_ylabel("K_speedup")
    ax[1,0].set_ylim(0.9, 1.1); ax[1,0].set_title("K_speedup vs dataset speedup (bias check)")

ntc = [(len(r["s_tcs"]), r["K_src"]) for r in rows if r.get("K_src")]
if ntc:
    ax[1,1].scatter([a for a, b in ntc], [b for a, b in ntc], s=8, alpha=0.35, color="#ff7f0e")
    ax[1,1].axhline(1.0, color="k", ls="--"); ax[1,1].set_ylim(0.9, 1.1)
    ax[1,1].set_xlabel("# test cases (src)"); ax[1,1].set_ylabel("K_src")
    ax[1,1].set_title("K vs #test-cases (coverage check)")

ax[1,2].axis("off")
txt = (f"STEP 4 SCALE — native+dynamic gem5, no qemu\n\n"
       f"pairs: {s['n_pairs']} ({s['n_pairs_both_compiled']} both compiled)\n"
       f"unique binaries: {s['n_unique_bins']}  compiled: {s['n_compiled']}\n"
       f"compile: {s['compile_stats']}\n"
       f"evals: {s['n_evals']}  timeouts: {s['timeout_evals']}\n"
       f"output-correct: {s['output_correct_rate']*100:.1f}%\n"
       f"determinism (serial==parallel): {s['determinism_serial_vs_parallel']}\n\n"
       f"ABS  K median={s['abs_K_median']}  <2%={s['abs_K_within_2pct']*100:.1f}%\n"
       f"SPDUP K median={s['speedup_K_median']}  <2%={s['speedup_K_within_2pct']*100:.1f}%\n\n"
       f"throughput: {s['throughput_ev_per_s']} evals/s ({s['node_cores']} cores)\n"
       f"per-step (1024 rollouts):\n"
       f"  K=5 tc  : {s['per_step_est_min']['K5_tc']} min\n"
       f"  K=10 tc : {s['per_step_est_min']['K10_tc']} min\n"
       f"  all ~100: {s['per_step_est_min']['all_tc_~100']} min")
ax[1,2].text(0.0, 1.0, txt, va="top", ha="left", fontsize=11, family="monospace")

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.tight_layout(); plt.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT)
print(json.dumps(s, indent=2))
