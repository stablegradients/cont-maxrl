"""CONFOUND 1 test: does grading ALL distinct tgt (not just top-5-by-gem5)
rescue any 'BLIND' problem? Take 25 problems flagged BLIND (gem5>1.3x,
instr<1.1x) in the main JSON that have >5 distinct tgt (so top-5 could have
missed an instruction-fast solution), grade EVERY distinct tgt, take the true
MIN instruction count, recompute instr-speedup = src_instr / true_min_instr.
Report before/after blind fraction.

Per-problem tgt count is capped at CAP (default 200, randomly sampled with a
fixed seed) so one huge problem (p03160 has 868) can't oversubscribe the node.
For problems with <=CAP distinct tgt we grade them ALL.
"""
from __future__ import annotations
import json, os, hashlib, collections, random, sys

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests")
REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from experiment11_code_optim.reward import pie_reward_fn as R
from experiment11_code_optim.measurement.core import setup_pch
os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()

CAP = int(os.environ.get("CAP", "200"))
PIDS = os.environ["PIDS"].split(",")   # comma-sep list of pids to grade

# pid -> src_instr_count (the reward baseline)
df = pd.read_parquet("/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_train.parquet")
src_instr = {}
for r in df["reward_model"]:
    g = json.loads(r["ground_truth"]); src_instr.setdefault(g["problem_id"], g.get("src_instr_count"))

# Collect distinct tgt + their gem5 speedup for the requested pids only.
pidset = set(PIDS)
tgts = collections.defaultdict(dict)   # pid -> {code: gem5_speedup}
gem5_best = {}
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r = json.loads(line); pid = r["problem_id"]
    if pid not in pidset:
        continue
    sp = float(r["speedup"])
    if r["tgt_code"] not in tgts[pid] or sp > tgts[pid][r["tgt_code"]]:
        tgts[pid][r["tgt_code"]] = sp
    gem5_best[pid] = max(gem5_best.get(pid, 0.0), sp)

# Build task set: ALL distinct tgt per pid (capped at CAP, seeded sample).
tasks = {}
task_meta = {}      # key -> pid
gem5_of_key = {}    # key -> gem5 speedup of that tgt
for pid in PIDS:
    codes = list(tgts[pid].keys())
    if len(codes) > CAP:
        codes = random.Random(int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)).sample(codes, CAP)
    for code in codes:
        k = hashlib.md5(f"{code}::{pid}".encode()).hexdigest()
        tasks[k] = {"code": code, "problem_id": pid, "split": "train"}
        task_meta[k] = pid
        gem5_of_key[k] = tgts[pid][code]

print(f"[confound1] {len(PIDS)} problems, grading {len(tasks)} distinct tgt (ALL, cap={CAP}) ...", flush=True)
res = R.grade_dedup_batch(tasks, test_case_dir=os.environ["PIE_TEST_CASE_DIR"],
                          measurement_max_tests=5, timeout_per_test=30,
                          num_workers=16, global_timeout=None)

# per problem: min tgt instr over ALL passing tgt; also track which tgt gave the min
min_instr_all = {}
min_key_all = {}
n_pass = collections.Counter()
n_fail = collections.Counter()
# also: min over only the TOP-5-by-gem5 (to reproduce the original method on the same data)
top5_keys = collections.defaultdict(list)
for pid in PIDS:
    ks = [k for k in task_meta if task_meta[k] == pid]
    ks.sort(key=lambda k: -gem5_of_key[k])
    top5_keys[pid] = set(ks[:5])

min_instr_top5 = {}
for k, pid in task_meta.items():
    d = res[k]
    if d["all_pass"] and d["elapsed_ns"] and d["elapsed_ns"] > 0:
        n_pass[pid] += 1
        if d["elapsed_ns"] < min_instr_all.get(pid, 1e18):
            min_instr_all[pid] = d["elapsed_ns"]; min_key_all[pid] = k
        if k in top5_keys[pid]:
            min_instr_top5[pid] = min(min_instr_top5.get(pid, 1e18), d["elapsed_ns"])
    else:
        n_fail[pid] += 1

rows = []
for pid in PIDS:
    si = src_instr.get(pid)
    if not si or pid not in min_instr_all:
        rows.append({"pid": pid, "src_instr": si, "status": "no_passing_tgt",
                     "n_distinct": len(tgts[pid]), "n_pass": n_pass[pid], "n_fail": n_fail[pid]})
        continue
    instr_all = si / min_instr_all[pid]
    instr_top5 = si / min_instr_top5[pid] if pid in min_instr_top5 else None
    rows.append({
        "pid": pid, "gem5": gem5_best.get(pid, 0.0), "src_instr": si,
        "n_distinct": len(tgts[pid]), "n_graded": n_pass[pid] + n_fail[pid],
        "n_pass": n_pass[pid], "n_fail": n_fail[pid],
        "min_instr_all": min_instr_all[pid],
        "instr_sp_all_tgt": instr_all,
        "instr_sp_top5": instr_top5,
        "gem5_of_min_tgt": gem5_of_key[min_key_all[pid]],
        "rescued": instr_all >= 1.1,   # was BLIND (<1.1); does all-tgt push it >=1.1?
    })

opath = f"{REPO}/experiment11_code_optim/docs/reward_analysis_confound1.json"
json.dump({"rows": rows, "cap": CAP}, open(opath, "w"), indent=1)

ok = [r for r in rows if r.get("instr_sp_all_tgt") is not None]
resc = [r for r in ok if r["rescued"]]
print(f"\n===== CONFOUND 1: ALL-TGT REGRADE of {len(rows)} BLIND/>5-tgt problems =====")
print(f"  problems with >=1 passing tgt: {len(ok)}/{len(rows)}")
print(f"  RESCUED (all-tgt instr-sp >= 1.1, i.e. no longer blind): {len(resc)}/{len(ok)}")
for r in ok:
    flag = "  <-- RESCUED" if r["rescued"] else ""
    t5 = f"{r['instr_sp_top5']:.3f}" if r["instr_sp_top5"] else "n/a"
    print(f"  {r['pid']}: gem5={r['gem5']:.2f} n_distinct={r['n_distinct']} n_graded={r['n_graded']} "
          f"instr_sp_top5={t5} -> instr_sp_ALL={r['instr_sp_all_tgt']:.3f}{flag}")
print(f"\nwrote {opath}")
