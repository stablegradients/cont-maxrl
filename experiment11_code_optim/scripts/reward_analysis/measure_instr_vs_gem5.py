"""Across ALL problems: does our -O0 perf retired-instruction reward assign a
speedup to the known-fast (tgt) solutions, or are tgt and src instruction counts
basically equal (as in p03688)?

For each problem: src_instr_count (precomputed table, = what the reward uses as
baseline) vs the MIN instruction count over its best gem5-speedup tgt solutions.
instr_speedup = src_instr / min_tgt_instr  (exactly the speedup our reward gives).
Compare to the dataset's gem5 speedup. Save JSON + scatter PNG.
"""
from __future__ import annotations
import json, os, hashlib, collections
os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests")
REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
import sys; sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from experiment11_code_optim.reward import pie_reward_fn as R
from experiment11_code_optim.measurement.core import setup_pch
os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()

K_TGT = int(os.environ.get("K_TGT", "5"))   # measure up to K best-gem5 tgt per problem

# pid -> src_instr_count (the reward's baseline)
df = pd.read_parquet("/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_train.parquet")
src_instr = {}
for r in df["reward_model"]:
    g = json.loads(r["ground_truth"]); src_instr.setdefault(g["problem_id"], g.get("src_instr_count"))

# pid -> list of (tgt_code, gem5_speedup); also collect best gem5 speedup per pid
tgts = collections.defaultdict(list)
gem5_best = {}
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r = json.loads(line); pid = r["problem_id"]
    if pid not in src_instr or not src_instr[pid]:
        continue
    sp = float(r["speedup"])
    tgts[pid].append((sp, r["tgt_code"]))
    gem5_best[pid] = max(gem5_best.get(pid, 0.0), sp)

# build one big dedup task set: top-K best-gem5 distinct tgts per problem
tasks = {}
task_meta = {}   # key -> (pid)
for pid, lst in tgts.items():
    lst = sorted(lst, key=lambda x: -x[0])
    seen = set()
    for sp, code in lst:
        if code in seen: continue
        seen.add(code)
        k = hashlib.md5(f"{code}::{pid}".encode()).hexdigest()
        tasks[k] = {"code": code, "problem_id": pid, "split": "train"}
        task_meta[k] = pid
        if len(seen) >= K_TGT: break

print(f"[measure] {len(src_instr)} problems, grading {len(tasks)} distinct tgt solutions "
      f"(top-{K_TGT} by gem5/problem) ...", flush=True)
res = R.grade_dedup_batch(tasks, test_case_dir=os.environ["PIE_TEST_CASE_DIR"],
                          measurement_max_tests=5, timeout_per_test=30,
                          num_workers=64, global_timeout=None)

# per problem: min tgt instruction count among passing tgts
min_tgt_instr = {}
n_compile_fail = collections.Counter()
n_tgt_total = collections.Counter()
for k, pid in task_meta.items():
    n_tgt_total[pid] += 1
    d = res[k]
    if d["all_pass"] and d["elapsed_ns"] and d["elapsed_ns"] > 0:
        min_tgt_instr[pid] = min(min_tgt_instr.get(pid, 1e18), d["elapsed_ns"])
    else:
        n_compile_fail[pid] += 1

rows = []
for pid in src_instr:
    if pid in min_tgt_instr and src_instr[pid]:
        instr_sp = src_instr[pid] / min_tgt_instr[pid]
        rows.append({"pid": pid, "gem5_speedup": gem5_best.get(pid, 0.0),
                     "instr_speedup_ours": instr_sp,
                     "n_tgt_measured": n_tgt_total[pid],
                     "n_tgt_failed": n_compile_fail[pid]})

out = {"rows": rows, "n_problems_with_data": len(rows),
       "n_problems_total": len(src_instr),
       "n_tgt_graded": len(tasks),
       "n_tgt_failed_total": sum(n_compile_fail.values())}
opath = f"{REPO}/experiment11_code_optim/docs/reward_analysis_instr_vs_gem5.json"
os.makedirs(os.path.dirname(opath), exist_ok=True)
json.dump(out, open(opath, "w"))

# ---- summary stats ----
g = np.array([r["gem5_speedup"] for r in rows])
ins = np.array([r["instr_speedup_ours"] for r in rows])
head = g > 1.3   # problems the dataset says have real (gem5) headroom
print(f"\n===== INSTRUCTION-SPEEDUP (our reward) vs GEM5-SPEEDUP (dataset), n={len(rows)} problems =====")
print(f"  problems with gem5 headroom (>1.3x): {head.sum()} ({100*head.mean():.0f}%)")
print(f"  of those, our reward sees instr-speedup:")
print(f"     median = {np.median(ins[head]):.2f}x")
print(f"     <1.1x (BLIND/flat): {100*np.mean(ins[head]<1.1):.0f}%")
print(f"     >1.3x (visible)   : {100*np.mean(ins[head]>1.3):.0f}%")
print(f"  corr(log gem5, log instr) over all = {np.corrcoef(np.log(g), np.log(ins))[0,1]:+.3f}")
print(f"  tgt compile/runtime failures in our env: {out['n_tgt_failed_total']}/{out['n_tgt_graded']} "
      f"({100*out['n_tgt_failed_total']/out['n_tgt_graded']:.0f}%)")

# ---- scatter ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 7))
blind = (ins < 1.1) & (g > 1.3)
ax.scatter(g[~blind], ins[~blind], s=8, alpha=0.4, c="#1f77b4", label="reward sees some speedup")
ax.scatter(g[blind], ins[blind], s=10, alpha=0.5, c="#d62728", label="BLIND (gem5>1.3x, instr<1.1x)")
lim = [0.8, max(g.max(), ins.max())*1.1]
ax.plot(lim, lim, "k--", lw=1, label="y=x (reward tracks gem5)")
ax.axhline(1.0, color="gray", ls=":", lw=1, label="instr-speedup=1 (no reward)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim([0.5, lim[1]])
ax.set_xlabel("gem5 speedup (dataset; what PIE optimizes)")
ax.set_ylabel("instruction-speedup our reward assigns to the fast solution")
ax.set_title(f"Our -O0 perf-instruction reward vs gem5, all {len(rows)} problems\n"
             f"{100*np.mean(ins[head]<1.1):.0f}% of gem5-headroom problems are BLIND to our reward (red)")
ax.legend(fontsize=8)
ppath = f"{REPO}/experiment11_code_optim/docs/reward_analysis_instr_vs_gem5.png"
fig.tight_layout(); fig.savefig(ppath, dpi=120); print(f"\nwrote {opath}\nwrote {ppath}")
