"""Validate PATCHED native-x86 gem5 (no qemu) reproduces PIE's EXACT dataset gem5 timing,
using DYNAMIC binaries (PIE's build) + the MERGED test cases. The vma.cc + elf_object.cc +
syscall_emul.hh patches let native gem5 run dynamic x86 binaries on the 64KB-page aarch64 host.

Per problem: compile src+tgt DYNAMIC (g++-9 -O3 -std=c++17, NO -static, interp baked to the rootfs
ld-2.31.so), run native gem5 Verbatim on each test case in PIE's tc2time, sum sim_ticks/1e12 = agg,
speedup = src_agg/tgt_agg. Compare per-tc to PIE's tc2time and agg/speedup to the dataset.

Usage: python native_dynamic_validate.py [N_PROBLEMS]   (default 6)
"""
import json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

RF    = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
V20   = "/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.opt"   # PATCHED native gem5
ENV   = "/work/nvme/bgtv/sramasubramanian/envs/gem5build"
CFG   = f"{RF}/home/gem5-skylake-config/gem5-configs"
RUNSE = f"{CFG}/run_se_redirect.py"                                          # has redirect_paths
LDREAL= f"{RF}/lib/x86_64-linux-gnu/ld-2.31.so"                              # host-resolvable interp
WD    = "/work/nvme/bgtv/sramasubramanian/qemu_x86"
PROOT = f"{WD}/proot/usr/bin/proot"; QB = f"{WD}/qu10/usr/bin/qemu-x86_64"
MERGED= "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
TESTJL= "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/data_pie/test.jsonl"
WORK  = f"{RF}/home/g5val"; os.makedirs(WORK, exist_ok=True)
GPPENV = {**os.environ, "LD_LIBRARY_PATH": f"{WD}/proot/usr/lib/aarch64-linux-gnu", "PROOT_NO_SECCOMP": "1"}
NATENV = {**os.environ, "LD_LIBRARY_PATH": f"{ENV}/lib", "PYTHONPATH": f"{CFG}:{CFG}/system"}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CORES = sorted(os.sched_getaffinity(0)); W = min(len(CORES), 64)

def sim_ticks(s):
    try:
        for l in open(s):
            if l.startswith("sim_ticks"): return int(l.split()[1])
    except Exception: pass
    return None

def compile_dyn(name, code):
    code = re.sub(r'#pragma GCC target.*\n', '', code)
    open(f"{WORK}/{name}.cpp", "w").write(code)
    cmd = [PROOT, "-q", QB, "-r", RF, "-w", "/home/g5val", "/usr/bin/g++",
           f"/home/g5val/{name}.cpp", "-o", f"/home/g5val/{name}",
           "-O3", "-std=c++17", f"-Wl,--dynamic-linker={LDREAL}"]
    try:
        p = subprocess.run(cmd, env=GPPENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        return p.returncode == 0 and os.path.exists(f"{WORK}/{name}"), p.stderr.decode("utf-8","ignore")[-200:]
    except Exception as e:
        return False, str(e)

def run_native(name, pid, tc, core):
    out = f"/tmp/g5val/{name}_{tc}"; os.makedirs(out, exist_ok=True)
    inp = f"{MERGED}/{pid}/input.{tc}.txt"
    if not os.path.exists(inp): return tc, None, False, "no_input"
    cmd = ["taskset", "-c", str(core), V20, "-d", out, "--stats-file=s.txt", RUNSE, "Verbatim", f"{WORK}/{name}"]
    try:
        with open(inp) as fh:
            p = subprocess.run(cmd, stdin=fh, env=NATENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired:
        return tc, None, False, "timeout"
    tk = sim_ticks(f"{out}/s.txt")
    got = p.stdout.decode("utf-8", "ignore").strip()
    try: exp = open(f"{MERGED}/{pid}/output.{tc}.txt").read().strip()
    except Exception: exp = None
    ok = exp is not None and (exp in got or got == exp)
    subprocess.run(["rm", "-rf", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tc, (tk/1e12 if tk is not None else None), ok, ("ok" if tk is not None else "no_ticks")

def agg_side(name, pid, tc2time):
    tcs = sorted(int(t) for t in tc2time.keys())
    res = {}
    with ThreadPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(run_native, name, pid, tc, CORES[i % W]) for i, tc in enumerate(tcs)]
        for f in as_completed(futs):
            tc, sec, ok, st = f.result(); res[tc] = (sec, ok, st)
    total = 0.0; allok = True; nbad = 0; ntimeout = 0
    for tc in tcs:
        sec, ok, st = res[tc]
        if st == "timeout": ntimeout += 1; total = float("inf")
        if sec is None: allok = False
        else: total += sec
        if not ok: nbad += 1
    return total, res, allok, nbad, ntimeout, tcs

def main():
    rows = []
    for line in open(TESTJL):
        r = json.loads(line)
        if r.get("src_code") and r.get("tgt_code") and r.get("src_code_tc2time") and r.get("tgt_code_tc2time") \
           and os.path.isdir(f"{MERGED}/{r['problem_id']}"):
            rows.append(r)
        if len(rows) >= N: break
    print(f"validating {len(rows)} problems with PATCHED native+DYNAMIC gem5 on MERGED test cases\n", flush=True)
    print("%-9s %-7s %5s %14s %14s %8s  %14s %14s %8s" %
          ("problem", "status", "ntc", "our_src_s", "data_src_s", "K_src", "our_speedup", "data_speedup", "K_sp"), flush=True)
    out = []
    for r in rows:
        pid = r["problem_id"]; t0 = time.time()
        cs, ce = compile_dyn(f"{pid}_s", r["src_code"])
        ct, te = compile_dyn(f"{pid}_t", r["tgt_code"])
        if not (cs and ct):
            print(f"{pid:9} COMPILE_FAIL src={cs} tgt={ct}  {ce[:60] if not cs else te[:60]}", flush=True); continue
        s_tot, s_res, s_ok, s_bad, s_to, s_tcs = agg_side(f"{pid}_s", pid, r["src_code_tc2time"])
        t_tot, t_res, t_ok, t_bad, t_to, t_tcs = agg_side(f"{pid}_t", pid, r["tgt_code_tc2time"])
        d_src = float(r["src_agg_runtime"]); d_tgt = float(r["tgt_agg_runtime"])
        our_sp = (s_tot / t_tot) if t_tot not in (0, float("inf")) else None
        d_sp = d_src / d_tgt if d_tgt else None
        K_src = s_tot / d_src if d_src and s_tot != float("inf") else None
        K_sp = (our_sp / d_sp) if (our_sp and d_sp) else None
        print("%-9s %-7s %5d %14.8f %14.8f %8s  %14s %14.6f %8s  (%.0fs, bad=%d to=%d)" % (
            pid, "OK", len(s_tcs), s_tot, d_src,
            f"{K_src:.4f}" if K_src else "--",
            f"{our_sp:.6f}" if our_sp else "--", d_sp,
            f"{K_sp:.4f}" if K_sp else "--", time.time()-t0, s_bad+t_bad, s_to+t_to), flush=True)
        out.append({"pid": pid, "our_src": s_tot, "our_tgt": t_tot, "data_src": d_src, "data_tgt": d_tgt,
                    "our_sp": our_sp, "data_sp": d_sp, "K_src": K_src, "K_sp": K_sp,
                    "src_bad": s_bad, "tgt_bad": t_bad, "ntc": len(s_tcs)})
    json.dump(out, open("/tmp/native_dynamic_validate.json", "w"), indent=2)
    print(f"\nwrote /tmp/native_dynamic_validate.json", flush=True)
    Ks = [abs(o["K_sp"]-1) for o in out if o.get("K_sp")]
    if Ks: print(f"speedup K: {sum(1 for k in Ks if k<0.02)}/{len(Ks)} within 2%%, max|K-1|={max(Ks):.4f}", flush=True)

if __name__ == "__main__":
    os.makedirs("/tmp/g5val", exist_ok=True)
    main()
