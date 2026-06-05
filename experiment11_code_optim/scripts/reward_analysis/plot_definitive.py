"""Plot the definitive native-x86 gem5 reward validation (reads /tmp/gem5_definitive.json).
Six panels: reward corr (K5), ceiling full-tc vs K5, absolute-offset hist, engine-exactness hist,
K5-sampling hist, eval-wall distribution. Writes a PNG + prints a markdown summary."""
import json, sys, math, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gem5_definitive.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/docs/plots/gem5_definitive.png"
d = json.load(open(J))
s = d["summary"]
fig, ax = plt.subplots(2, 3, figsize=(18, 11))

# (1) reward corr K5
cp = d["corr_pts"]
if cp:
    x = [math.exp(p["log_dssp"]) for p in cp]; y = [math.exp(p["log_nsp"]) for p in cp]
    ax[0,0].scatter(x, y, s=10, alpha=0.4, color="#1f77b4")
    lo, hi = min(x+y), max(x+y)
    ax[0,0].plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
    ax[0,0].set_xscale("log"); ax[0,0].set_yscale("log")
    ax[0,0].set_xlabel("PIE dataset speedup"); ax[0,0].set_ylabel("native-gem5 speedup (K=5)")
    ax[0,0].set_title(f"REWARD corr (K=5, output-correct)\nPearson(loglog)={s['reward_pearson_loglog']}  "
                      f"Spearman={s['reward_spearman']}  n={s['reward_n_pairs']}")
    ax[0,0].legend()

# (2) ceiling: full-100 vs K5 on same abs pairs
ce = d.get("ceiling_pts", [])
if ce:
    xd = [math.exp(p["log_dssp"]) for p in ce]
    yf = [math.exp(p["log_full_nsp"]) for p in ce]
    yk = [math.exp(p["log_k5_nsp"]) for p in ce]
    ax[0,1].scatter(xd, yf, s=14, alpha=0.6, color="#2ca02c", label=f"full 100-tc (Sp={s.get('reward_corr_ceiling_fulltc_spearman')})")
    ax[0,1].scatter(xd, yk, s=14, alpha=0.4, color="#d62728", label=f"K=5 (Sp={s.get('reward_corr_k5_on_abs_spearman')})")
    lo, hi = min(xd+yf+yk), max(xd+yf+yk)
    ax[0,1].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0,1].set_xscale("log"); ax[0,1].set_yscale("log")
    ax[0,1].set_xlabel("PIE dataset speedup"); ax[0,1].set_ylabel("native-gem5 speedup")
    ax[0,1].set_title(f"Corr CEILING: full-100-tc vs K=5 (same {s.get('reward_corr_ceiling_n')} pairs)\n"
                      f"-> the gap is pure K=5 sampling cost")
    ax[0,1].legend()

# (3) absolute offset native_s/dataset_s
ap = d.get("abs_pts", [])
if ap:
    r = [p["native_s"]/p["dataset_s"] for p in ap if p["dataset_s"] > 0]
    ax[0,2].hist(r, bins=40, color="#9467bd", alpha=0.8)
    ax[0,2].axvline(1.0, color="k", ls="--", label="exact match")
    ax[0,2].axvline(statistics.median(r), color="r", ls="-", label=f"median={statistics.median(r):.3f}")
    ax[0,2].set_xlabel("native sum_ticks/1e12  /  PIE *_agg_runtime (all 100 tc)")
    ax[0,2].set_ylabel("# program-sides")
    ax[0,2].set_title(f"ABSOLUTE offset (n={len(r)})\ngenerated_test_cases != PIE's agg input set")
    ax[0,2].legend()

# (4) engine exactness native vs qemu
eng = d.get("engine_err_pct", [])
if eng:
    ax[1,0].hist(eng, bins=30, color="#ff7f0e", alpha=0.85)
    ax[1,0].axvline(statistics.mean(eng), color="r", label=f"mean={statistics.mean(eng):.3f}%")
    ax[1,0].set_xlabel("|native - qemu| / qemu  (%)"); ax[1,0].set_ylabel("# evals")
    ax[1,0].set_title(f"ENGINE exactness: native vs same-binary-under-qemu\n"
                      f"mean={s['engine_native_vs_qemu_mean_pct']}%  max={s['engine_native_vs_qemu_max_pct']}%  n={s['engine_exactness_n']}")
    ax[1,0].legend()

# (5) K5 sampling ratio
kr = d.get("k5_ratio", [])
if kr:
    ax[1,1].hist(kr, bins=40, color="#17becf", alpha=0.85)
    ax[1,1].axvline(1.0, color="k", ls="--")
    ax[1,1].axvline(statistics.median(kr), color="r", label=f"median={statistics.median(kr):.3f}")
    ax[1,1].set_xlabel("K=5-scaled-sum / true 100-tc sum (per binary)")
    ax[1,1].set_ylabel("# binaries")
    ax[1,1].set_title(f"K=5 SAMPLING cost (5 of 100 inputs)\nmedian={s['k5_sampling_ratio_median']}")
    ax[1,1].legend()

# (6) eval wall distribution
w = list(d.get("native_walls", {}).values())
if w:
    ax[1,2].hist([min(x, 60) for x in w], bins=40, color="#8c564b", alpha=0.85)
    ax[1,2].axvline(statistics.median(w), color="r", label=f"median={statistics.median(w):.2f}s")
    ax[1,2].set_xlabel("per-eval wall (s, clipped 60)"); ax[1,2].set_ylabel("# evals")
    ax[1,2].set_title(f"throughput {s['throughput_evals_per_s']} evals/s on {s['node_cores']} cores\n"
                      f"per-step 1024xK5 = {s['per_step_1024xK5_min_at_thisnode']} min")
    ax[1,2].legend()

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.tight_layout(); plt.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT)
print(json.dumps(s, indent=2))
