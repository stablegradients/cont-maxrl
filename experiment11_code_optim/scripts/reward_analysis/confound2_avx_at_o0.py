"""Does AVX/SIMD-targeted competitive code (the dominant compile-fail class on
aarch64) actually have a LOWER retired-instruction count at -O0 when we strip
the offending pragma so it compiles? If not, then these compile failures cannot
be hiding instruction-fast solutions -> confound 2 does not inflate blindness.

We pick high-gem5 AVX-pragma tgt across several problems, neutralise the
'#pragma GCC target("avx...")' line (comment it out) so g++-14/aarch64 compiles
it, grade it, and compare its -O0 instr count to (a) the src and (b) the best
PASSING non-avx tgt for the same problem.
"""
from __future__ import annotations
import json, os, hashlib, collections, sys, re

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests")
REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)
import numpy as np, pandas as pd
from experiment11_code_optim.reward import pie_reward_fn as R
from experiment11_code_optim.measurement.core import setup_pch
os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()

# src_instr baseline
df = pd.read_parquet("/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_train.parquet")
src_instr = {}
for r in df["reward_model"]:
    g = json.loads(r["ground_truth"]); src_instr.setdefault(g["problem_id"], g.get("src_instr_count"))

PRAGMA_RE = re.compile(r'^\s*#pragma\s+GCC\s+target\b.*$', re.MULTILINE)

def neutralize(code: str) -> str:
    # comment out the target pragma so aarch64 g++ accepts it; behaviour at -O0
    # is unaffected by it anyway (no vectorization at -O0).
    return PRAGMA_RE.sub('// [neutralized avx pragma]', code)

avx_re = re.compile(r'#pragma\s+GCC\s+target')
# gather, per problem, the highest-gem5 AVX tgt and the highest-gem5 non-AVX tgt
best_avx = {}; best_nonavx = {}
PIDS = os.environ["PIDS"].split(",")
pidset = set(PIDS)
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r = json.loads(line); pid = r["problem_id"]
    if pid not in pidset: continue
    sp = float(r["speedup"])
    if avx_re.search(r["tgt_code"]):
        if pid not in best_avx or sp > best_avx[pid][1]:
            best_avx[pid] = (r["tgt_code"], sp)
    else:
        if pid not in best_nonavx or sp > best_nonavx[pid][1]:
            best_nonavx[pid] = (r["tgt_code"], sp)

tasks = {}; meta = {}
for pid in PIDS:
    if pid in best_avx:
        code = neutralize(best_avx[pid][0])
        k = hashlib.md5(f"{code}::{pid}".encode()).hexdigest()
        tasks[k] = {"code": code, "problem_id": pid, "split": "train"}; meta[k] = (pid, "avx_neutralized", best_avx[pid][1])
    if pid in best_nonavx:
        code = best_nonavx[pid][0]
        k = hashlib.md5(f"{code}::{pid}".encode()).hexdigest()
        tasks[k] = {"code": code, "problem_id": pid, "split": "train"}; meta[k] = (pid, "nonavx", best_nonavx[pid][1])

print(f"[avx@O0] grading {len(tasks)} tgt across {len(PIDS)} problems ...", flush=True)
res = R.grade_dedup_batch(tasks, test_case_dir=os.environ["PIE_TEST_CASE_DIR"],
                          measurement_max_tests=5, timeout_per_test=30,
                          num_workers=16, global_timeout=None)

by_pid = collections.defaultdict(dict)
for k,(pid,kind,sp) in meta.items():
    d = res[k]
    by_pid[pid][kind] = (d["all_pass"], d.get("elapsed_ns"), sp, (d.get("error") or "")[:60])

print("\n===== AVX-neutralized vs non-AVX tgt instr-count at -O0 =====")
print("(if AVX tgt is NOT meaningfully fewer instructions, compile-fail of AVX code can't hide speedup)")
for pid in PIDS:
    si = src_instr.get(pid)
    rec = by_pid.get(pid, {})
    line = f"  {pid}: src_instr={si}"
    for kind in ("avx_neutralized","nonavx"):
        if kind in rec:
            ap, e_ns, sp, err = rec[kind]
            if ap and e_ns:
                isp = si/e_ns if si else float('nan')
                line += f" | {kind}(gem5={sp:.2f}): instr={e_ns} instr_sp={isp:.3f}"
            else:
                line += f" | {kind}(gem5={sp:.2f}): FAIL[{err}]"
    print(line)
