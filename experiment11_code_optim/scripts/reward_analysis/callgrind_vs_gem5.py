"""Q4: does a callgrind cache/branch-aware CYCLE estimate track gem5 speedup
better than retired-instruction count? Compile src+best-tgt at -O2, run under
callgrind --cache-sim --branch-sim on the reward's 5 subsampled tests, extract
both Ir (instructions) and CEst (cycle estimate). Correlate each with gem5."""
from __future__ import annotations
import json, os, re, hashlib, subprocess, random
from concurrent.futures import ThreadPoolExecutor
import numpy as np

VALGRIND = "/opt/cray/pe/valgrind4hpc/2.13.6/bin/valgrind"
GPP = "/opt/cray/pe/gcc-native/14/bin/g++"
TESTS = "/work/nvme/bgmo/sramasubramanian/pie_data/tests"
REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
FLAGS = ["-O2", "-std=c++17"]
SCRATCH = os.environ.get("PIE_COMPILE_SCRATCH", "/tmp/cg_scratch"); os.makedirs(SCRATCH, exist_ok=True)

rows = json.load(open(f"{REPO}/experiment11_code_optim/docs/reward_analysis_instr_vs_gem5.json"))["rows"]
by_pid = {r["pid"]: r for r in rows}
blind = [r for r in rows if r["gem5_speedup"] > 2 and r["instr_speedup_ours"] < 1.1]
vis   = [r for r in rows if r["gem5_speedup"] > 2 and r["instr_speedup_ours"] > 1.3]
random.Random(0).shuffle(blind); random.Random(1).shuffle(vis)
sample = blind[:18] + vis[:12]
pids = {r["pid"] for r in sample}

best = {}
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r = json.loads(line); pid = r["problem_id"]
    if pid in pids:
        sp = float(r["speedup"])
        if pid not in best or sp > best[pid][0]:
            best[pid] = (sp, r["src_code"], r["tgt_code"])

def subsample(pid):
    cases, i = [], 0
    while os.path.exists(f"{TESTS}/{pid}/input.{i}.txt"):
        cases.append((open(f"{TESTS}/{pid}/input.{i}.txt").read(),
                      open(f"{TESTS}/{pid}/output.{i}.txt").read())); i += 1
    if len(cases) > 5:
        seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
        cases = random.Random(seed).sample(cases, 5)
    return cases

def cg_measure(code, pid, tag):
    code = re.sub(r'#pragma GCC target.*\n', '', code)
    d = f"{SCRATCH}/{pid}_{tag}_{os.getpid()}"; os.makedirs(d, exist_ok=True)
    src, exe = f"{d}/a.cpp", f"{d}/a.out"
    open(src, "w").write(code)
    if subprocess.run([GPP] + FLAGS + [src, "-o", exe], capture_output=True, timeout=90).returncode != 0:
        return None
    Ir_tot = CEst_tot = 0
    for inp, exp in subsample(pid):
        out = f"{d}/cg.out"
        try:
            p = subprocess.run([VALGRIND, "--tool=callgrind", "--cache-sim=yes", "--branch-sim=yes",
                                f"--callgrind-out-file={out}", "--quiet", exe],
                               input=inp.encode(), capture_output=True, timeout=180)
        except Exception:
            return None
        if p.stdout.decode("utf-8", "ignore").strip() != exp.strip():
            return None  # wrong answer -> not a valid solution here
        txt = open(out).read()
        ev_names = re.search(r'^events:\s+(.+)$', txt, re.M)
        ev_vals = re.search(r'^summary:\s+(.+)$', txt, re.M)
        if not ev_names or not ev_vals: return None
        ev = dict(zip(ev_names.group(1).split(), [int(x) for x in ev_vals.group(1).split()]))
        Ir = ev.get("Ir", 0)
        L1m = ev.get("I1mr", 0) + ev.get("D1mr", 0) + ev.get("D1mw", 0)
        LLm = ev.get("ILmr", 0) + ev.get("DLmr", 0) + ev.get("DLmw", 0)
        Br = ev.get("Bcm", 0) + ev.get("Bim", 0)
        Ir_tot += Ir
        CEst_tot += Ir + 10 * L1m + 100 * LLm + 10 * Br   # cachegrind-style cycle estimate
    return Ir_tot, CEst_tot

def work(pid):
    sp, src, tgt = best[pid]
    s = cg_measure(src, pid, "src"); t = cg_measure(tgt, pid, "tgt")
    if not s or not t: return None
    return {"pid": pid, "gem5": sp, "o0_instr_sp": by_pid[pid]["instr_speedup_ours"],
            "o2_instr_sp": s[0] / t[0], "o2_cycle_sp": s[1] / t[1],
            "was_blind": by_pid[pid]["instr_speedup_ours"] < 1.1}

print(f"[callgrind] {len(best)} problems ({sum(r['instr_speedup_ours']<1.1 for r in sample)} -O0-blind) ...", flush=True)
with ThreadPoolExecutor(max_workers=16) as ex:
    res = [r for r in ex.map(work, list(best)) if r]

g = np.array([r["gem5"] for r in res]); o0 = np.array([r["o0_instr_sp"] for r in res])
o2i = np.array([r["o2_instr_sp"] for r in res]); o2c = np.array([r["o2_cycle_sp"] for r in res])
def corr(x): return np.corrcoef(np.log(np.clip(g,1e-3,None)), np.log(np.clip(x,1e-3,None)))[0,1]
print(f"\n===== n={len(res)} problems (src+tgt both compiled -O2, correct) =====")
print(f"  CORR(log gem5, log X):")
print(f"    -O0 retired-instructions (current reward) : {corr(o0):+.3f}")
print(f"    -O2 retired-instructions                  : {corr(o2i):+.3f}")
print(f"    -O2 callgrind CYCLE estimate (cache+branch): {corr(o2c):+.3f}")
bl = np.array([r["was_blind"] for r in res])
if bl.sum():
    print(f"\n  on the {bl.sum()} -O0-BLIND problems (gem5>2x, -O0 instr<1.1x):")
    print(f"    median -O2 cycle-speedup callgrind assigns = {np.median(o2c[bl]):.2f}x  (vs gem5 median {np.median(g[bl]):.1f}x)")
    print(f"    fraction now >1.3x (RESCUED by callgrind)  = {100*np.mean(o2c[bl]>1.3):.0f}%")
print("\nper-problem:"); print(f"  {'pid':>8}{'gem5':>8}{'o0_instr':>9}{'o2_instr':>9}{'o2_cycle':>9}")
for r in sorted(res, key=lambda r:-r["gem5"]):
    print(f"  {r['pid']:>8}{r['gem5']:8.1f}{r['o0_instr_sp']:9.2f}{r['o2_instr_sp']:9.2f}{r['o2_cycle_sp']:9.2f}")
json.dump(res, open(f"{REPO}/experiment11_code_optim/docs/callgrind_vs_gem5.json", "w"))
