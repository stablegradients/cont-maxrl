"""-O0 vs -O3 confound test (LENS: optimization level).

For a fixed sample of 15 problems (10 BLIND, 5 VISIBLE per the JSON), measure
retired-instruction counts under perf for BOTH src_code and the top-K best-gem5
tgt solutions, at a SINGLE compile-flag setting chosen by env var FLAG ("O0"|"O3").

instr_speedup(flag) = src_instr(flag) / min_tgt_instr(flag)   -- matched flags.

Run once per flag (separate process + PCH scratch each), then a third pass joins
the two JSONs and compares to gem5. We measure src FRESH at each flag (do not
reuse parquet -O0 baseline) so each ratio is internally flag-consistent.
"""
from __future__ import annotations
import json, os, hashlib, collections, sys

FLAG = os.environ["FLAG"]  # "O0" or "O3"
assert FLAG in ("O0", "O3")
REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests")
# Separate compile+PCH scratch per flag so the -O0 .gch and -O3 .gch never collide.
scratch = f"/tmp/pie_O0vsO3_{FLAG}"
os.makedirs(scratch, exist_ok=True)
os.environ["PIE_COMPILE_SCRATCH"] = scratch

# --- override COMPILE_FLAGS BEFORE setup_pch (PCH builds with COMPILE_FLAGS) ---
from experiment11_code_optim.measurement import core
core.COMPILE_FLAGS = ["-O0" if FLAG == "O0" else "-O3", "-std=c++17"]
from experiment11_code_optim.reward import pie_reward_fn as R
os.environ["PIE_PCH_INCLUDE_PATH"] = core.setup_pch()
print(f"[setup] FLAG={FLAG} COMPILE_FLAGS={core.COMPILE_FLAGS} backend={core.MEASUREMENT_BACKEND}", flush=True)

import pandas as pd

BLIND = ['p03032','p01288','p03096','p00448','p03013','p03160','p03339','p02279','p02658','p03940']
VIS   = ['p02235','p03527','p02960','p03855','p03768']
SAMPLE = set(BLIND + VIS)
K_TGT = 5

# pid -> src_code, and parquet src_instr_count (O0 reference for sanity only)
df = pd.read_parquet("/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_train.parquet")
src_code = {}
src_instr_parquet = {}
for r in df["reward_model"]:
    g = json.loads(r["ground_truth"])
    pid = g["problem_id"]
    if pid in SAMPLE and pid not in src_code:
        # src_code is not in ground_truth; pulled from raw below. store parquet count.
        src_instr_parquet[pid] = g.get("src_instr_count")

# pull src_code + top-K gem5 tgt from raw jsonl
tgts = collections.defaultdict(list)
gem5_best = {}
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r = json.loads(line); pid = r["problem_id"]
    if pid not in SAMPLE:
        continue
    src_code.setdefault(pid, r["src_code"])
    sp = float(r["speedup"])
    tgts[pid].append((sp, r["tgt_code"]))
    gem5_best[pid] = max(gem5_best.get(pid, 0.0), sp)

# build dedup task set: src + top-K distinct tgt per pid
tasks = {}
meta = {}  # key -> (pid, role)  role in {"src","tgt"}
for pid in SAMPLE:
    ks = hashlib.md5(f"SRC::{src_code[pid]}::{pid}".encode()).hexdigest()
    tasks[ks] = {"code": src_code[pid], "problem_id": pid, "split": "train"}
    meta[ks] = (pid, "src")
    lst = sorted(tgts[pid], key=lambda x: -x[0])
    seen = set()
    for sp, code in lst:
        if code in seen: continue
        seen.add(code)
        k = hashlib.md5(f"TGT::{code}::{pid}".encode()).hexdigest()
        tasks[k] = {"code": code, "problem_id": pid, "split": "train"}
        meta[k] = (pid, "tgt")
        if len(seen) >= K_TGT: break

print(f"[measure] FLAG={FLAG}: {len(SAMPLE)} problems, {len(tasks)} programs "
      f"(src+top{K_TGT} tgt) ...", flush=True)
res = R.grade_dedup_batch(tasks, test_case_dir=os.environ["PIE_TEST_CASE_DIR"],
                          measurement_max_tests=5, timeout_per_test=30,
                          num_workers=16, global_timeout=None)

# aggregate per pid
out = {}
for pid in SAMPLE:
    out[pid] = {"gem5_speedup": gem5_best.get(pid),
                "src_instr_parquet_O0": src_instr_parquet.get(pid),
                "src_instr": None, "src_pass": False,
                "tgt_instrs": [], "n_tgt_pass": 0, "n_tgt_fail": 0}
for k, (pid, role) in meta.items():
    d = res[k]
    ok = d["all_pass"] and d["elapsed_ns"] and d["elapsed_ns"] > 0
    if role == "src":
        out[pid]["src_pass"] = bool(ok)
        out[pid]["src_instr"] = int(d["elapsed_ns"]) if ok else None
    else:
        if ok:
            out[pid]["tgt_instrs"].append(int(d["elapsed_ns"]))
            out[pid]["n_tgt_pass"] += 1
        else:
            out[pid]["n_tgt_fail"] += 1

for pid in SAMPLE:
    o = out[pid]
    if o["src_instr"] and o["tgt_instrs"]:
        o["min_tgt_instr"] = min(o["tgt_instrs"])
        o["instr_speedup"] = o["src_instr"] / o["min_tgt_instr"]
    else:
        o["min_tgt_instr"] = None
        o["instr_speedup"] = None

opath = f"{REPO}/experiment11_code_optim/docs/reward_analysis/O0_vs_O3_{FLAG}.json"
os.makedirs(os.path.dirname(opath), exist_ok=True)
json.dump(out, open(opath, "w"), indent=2)
print(f"[done] FLAG={FLAG} wrote {opath}", flush=True)
for pid in BLIND + VIS:
    o = out[pid]
    sp = f"{o['instr_speedup']:.2f}" if o['instr_speedup'] else "NA"
    print(f"  {pid} gem5={o['gem5_speedup']:.1f} instr_{FLAG}={sp} "
          f"src={o['src_instr']} mintgt={o['min_tgt_instr']} "
          f"tgtfail={o['n_tgt_fail']}", flush=True)
