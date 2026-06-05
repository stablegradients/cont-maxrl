"""CONFOUND 2 test: WHAT KINDS of tgt fail to compile/run in our env, and are
the failing ones plausibly the instruction-fast solutions (which would mean we
undercount speedup -> overstate blindness)?

We grade the top-5-by-gem5 distinct tgt (reproducing the main measurement) for a
sample of problems, and for every FAILING tgt we record the error class. We also
compare, within each problem, whether failing tgts had HIGHER gem5 speedup than
passing tgts (failure correlated with the fast solutions would bias us).
"""
from __future__ import annotations
import json, os, hashlib, collections, sys, re

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests")
REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)
import numpy as np
from experiment11_code_optim.reward import pie_reward_fn as R
from experiment11_code_optim.measurement.core import setup_pch
os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()

PIDS = os.environ["PIDS"].split(",")
K_TGT = int(os.environ.get("K_TGT", "5"))
pidset = set(PIDS)

# collect top-K distinct tgt per pid by gem5, like the main measurement
tgts = collections.defaultdict(list)
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r = json.loads(line); pid = r["problem_id"]
    if pid not in pidset:
        continue
    tgts[pid].append((float(r["speedup"]), r["tgt_code"]))

tasks = {}; meta = {}; gem5_of = {}
for pid in PIDS:
    lst = sorted(tgts[pid], key=lambda x: -x[0]); seen=set()
    for sp, code in lst:
        if code in seen: continue
        seen.add(code)
        k = hashlib.md5(f"{code}::{pid}".encode()).hexdigest()
        tasks[k] = {"code": code, "problem_id": pid, "split": "train"}
        meta[k] = pid; gem5_of[k] = sp
        if len(seen) >= K_TGT: break

print(f"[confound2] {len(PIDS)} problems, grading {len(tasks)} top-{K_TGT} tgt ...", flush=True)
res = R.grade_dedup_batch(tasks, test_case_dir=os.environ["PIE_TEST_CASE_DIR"],
                          measurement_max_tests=5, timeout_per_test=30,
                          num_workers=16, global_timeout=None)

def err_class(d):
    if d["all_pass"]: return "PASS"
    e = (d.get("error") or "")
    if not d.get("compiled"):
        if "compile failed" in e: return "compile_error"
        if "timed out" in e: return "compile_timeout"
        if "no test cases" in e: return "no_tests"
        return "compile_other:"+e[:40]
    # compiled but not all_pass -> runtime/output mismatch or timeout
    if "batch timeout" in e: return "run_timeout"
    return "runtime_or_wrong_output"

classes = collections.Counter()
fail_gem5 = []; pass_gem5 = []
sample_errs = collections.defaultdict(list)
for k, pid in meta.items():
    d = res[k]; c = err_class(d)
    classes[c] += 1
    if c == "PASS": pass_gem5.append(gem5_of[k])
    else:
        fail_gem5.append(gem5_of[k])
        if len(sample_errs[c]) < 4:
            sample_errs[c].append((pid, (d.get("error") or "")[:160]))

print("\n===== FAILURE CLASS BREAKDOWN (top-5 tgt of sampled problems) =====")
tot = sum(classes.values())
for c, n in classes.most_common():
    print(f"  {c:28s}: {n:4d} ({100*n/tot:.1f}%)")
print(f"\n  total tgt graded: {tot}, failed: {tot-classes['PASS']} ({100*(tot-classes['PASS'])/tot:.1f}%)")

print("\n===== Are FAILING tgt the FAST ones? (gem5 of failing vs passing tgt) =====")
fg = np.array(fail_gem5); pg = np.array(pass_gem5)
if len(fg) and len(pg):
    print(f"  mean gem5 of FAILING tgt = {fg.mean():.2f} (median {np.median(fg):.2f}, n={len(fg)})")
    print(f"  mean gem5 of PASSING tgt = {pg.mean():.2f} (median {np.median(pg):.2f}, n={len(pg)})")
    print("  => if failing >> passing, fast solutions are being dropped (bias toward blindness)")

print("\n===== SAMPLE ERROR STRINGS per class =====")
for c, exs in sample_errs.items():
    print(f"  [{c}]")
    for pid, e in exs:
        print(f"     {pid}: {e}")
