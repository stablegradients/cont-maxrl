"""TRUE 1:1 PIE gem5 reproduction on ARM (DeltaAI GH200) via qemu-x86.

Runs PIE's ACTUAL x86 gem5 v20.1 binary + the EXACT Skylake Verbatim config from the
alexshypula/gem5-skylake:api image, compiling test programs with PIE's OWN g++-9 (-O3
-std=c++17, dynamic) -- all under qemu-x86_64 (qemu 10) + proot on the aarch64 host.

Because gem5 sim_seconds is a deterministic simulation of x86 hardware, reproducing PIE's
exact (binary, config, input) yields PIE's exact dataset number regardless of the host ISA.
This is the 1:1 check the ARM-native gem5 sweep (corr 0.78, K~0.70) could not give.

Per problem: compile src+tgt, run gem5 Verbatim on each test case, sum sim_seconds
(= sim_ticks/sim_freq, PIE's exact metric in benchmarking.py:calc_sim_seconds), and compare
to dataset src_agg_runtime / tgt_agg_runtime. Reports our/dataset ratio per program.
"""
import json, os, re, subprocess, sys, glob, argparse, time

# ---- fixed paths (see project_gem5_build memory) ----
WD   = "/work/nvme/bgtv/sramasubramanian/qemu_x86"
RF   = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
PROOT= f"{WD}/proot/usr/bin/proot"
QEMU = f"{WD}/qu10/usr/bin/qemu-x86_64"            # qemu 10 (needed for 64KB-host page mmap)
PROOT_LIB = f"{WD}/proot/usr/lib/aarch64-linux-gnu"
GEM5 = "/home/gem5/build/X86/gem5.opt"            # guest paths (proot root = RF)
CFG  = "/home/gem5-skylake-config/gem5-configs"
RUNSE= f"{CFG}/run-se.py"
GPP  = "/usr/bin/g++"                              # PIE's g++-9 in the rootfs
RAW  = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"
# PIE's dataset gem5 times were computed over generated_test_cases (summed). That dir lives
# INSIDE the PIE image; it is the test set that reproduces src_agg_runtime / tgt_agg_runtime.
TESTS= f"{RF}/home/pie-perf/data/codenet/generated_test_cases"
GWORK= f"{RF}/home/g5work"                         # host view of guest /home/g5work

ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = PROOT_LIB + ":" + ENV.get("LD_LIBRARY_PATH","")
ENV["PROOT_NO_SECCOMP"] = "1"

def proot(argv, stdin_path=None, timeout=None, extra_env=""):
    """Run argv inside the rootfs (guest paths) via proot+qemu10."""
    cmd = [PROOT, "-q", QEMU, "-r", RF, "-w", "/home/g5work"]
    if extra_env:
        cmd += ["env"] + extra_env.split()
    cmd += argv
    fh = open(stdin_path) if stdin_path else subprocess.DEVNULL
    try:
        p = subprocess.run(cmd, env=ENV, stdin=fh, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    finally:
        if stdin_path: fh.close()

def compile_x86(code, name):
    """Compile C++ -> x86 dynamic binary with PIE's g++-9 -O3 -std=c++17 (PIE's exact flags)."""
    code = re.sub(r'#pragma GCC target.*\n', '', code)
    os.makedirs(GWORK, exist_ok=True)
    open(f"{GWORK}/{name}.cpp", "w").write(code)
    rc, out, err = proot([GPP, f"/home/g5work/{name}.cpp", "-o", f"/home/g5work/{name}",
                          "--std=c++17", "-O3"], timeout=120)
    ok = rc == 0 and os.path.exists(f"{GWORK}/{name}")
    return ok, err.decode("utf-8","ignore")[:300]

def parse_sim_seconds(stats_host_path):
    """PIE metric: sim_ticks / sim_freq (benchmarking.py:calc_sim_seconds)."""
    if not os.path.exists(stats_host_path): return None
    st = {}
    for line in open(stats_host_path):
        line = re.sub("#.*","",line).strip()
        p = line.split()
        if len(p) >= 2 and p[0] in ("sim_ticks","sim_freq","sim_seconds"):
            try: st[p[0]] = float(p[1])
            except: pass
    if "sim_ticks" in st and st.get("sim_freq"):
        return st["sim_ticks"]/st["sim_freq"]
    return st.get("sim_seconds")

def gem5_one(binname, pid, tc, timeout):
    """Run PIE gem5 Verbatim on one test case; return (sim_seconds, correct_bool)."""
    stats_guest = f"/home/g5work/{binname}.{tc}.stats.txt"
    stats_host  = f"{GWORK}/{binname}.{tc}.stats.txt"
    in_path = f"{TESTS}/{pid}/input.{tc}.txt"
    # PIE: gem5.opt --stats-file=S run-se.py Verbatim BIN  < input
    rc, out, err = proot([GEM5, f"--stats-file={stats_guest}", RUNSE, "Verbatim",
                          f"/home/g5work/{binname}"],
                         stdin_path=in_path, timeout=timeout,
                         extra_env=f"PYTHONPATH={CFG}:{CFG}/system")
    ss = parse_sim_seconds(stats_host)
    # correctness: gem5 prints the simulated program's stdout; compare to expected
    got = out.decode("utf-8","ignore").strip()
    exp = open(f"{TESTS}/{pid}/output.{tc}.txt").read().strip() if os.path.exists(f"{TESTS}/{pid}/output.{tc}.txt") else None
    correct = (exp is not None and exp in got)   # gem5 emits warnings around it; substring match
    return ss, correct, err.decode("utf-8","ignore")[-300:]

def agg_program(code, name, pid, k_tests, per_test_timeout):
    ok, cerr = compile_x86(code, name)
    if not ok: return {"status":"compile_fail","err":cerr}
    tcs = sorted(int(re.search(r"input\.(\d+)\.txt",p).group(1))
                 for p in glob.glob(f"{TESTS}/{pid}/input.*.txt"))[:k_tests]
    tot=0.0; n=0; allcorrect=True; per=[]
    for tc in tcs:
        ss, corr, gerr = gem5_one(name, pid, tc, per_test_timeout)
        per.append({"tc":tc,"sim_s":ss,"correct":corr})
        if ss is None: return {"status":"gem5_fail","tc":tc,"err":gerr,"per":per}
        tot+=ss; n+=1; allcorrect = allcorrect and corr
    return {"status":"ok","sum_sim_s":tot,"n":n,"correct":allcorrect,"per":per}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", nargs="+", required=True)
    ap.add_argument("--k_tests", type=int, default=10)
    ap.add_argument("--per_test_timeout", type=int, default=600)
    ap.add_argument("--out", default=f"{GWORK}/skylake_validate.json")
    a = ap.parse_args()
    rows={}
    for line in open(RAW):
        r=json.loads(line); pid=r["problem_id"]
        if pid in a.pids and (pid not in rows or float(r["speedup"])>rows[pid]["speedup"]):
            rows[pid]={"speedup":float(r["speedup"]),"src":r["src_code"],"tgt":r["tgt_code"],
                       "data_src":float(r["src_agg_runtime"]),"data_tgt":float(r["tgt_agg_runtime"])}
    results=[]
    for pid,d in rows.items():
        t0=time.time()
        rs=agg_program(d["src"], f"{pid}_s", pid, a.k_tests, a.per_test_timeout)
        rt=agg_program(d["tgt"], f"{pid}_t", pid, a.k_tests, a.per_test_timeout)
        row={"pid":pid,"data_src":d["data_src"],"data_tgt":d["data_tgt"],"data_sp":d["speedup"],
             "src":rs,"tgt":rt,"sec":round(time.time()-t0,1)}
        if rs.get("status")=="ok" and rt.get("status")=="ok":
            row["our_src"]=rs["sum_sim_s"]; row["our_tgt"]=rt["sum_sim_s"]
            row["our_sp"]=rs["sum_sim_s"]/rt["sum_sim_s"] if rt["sum_sim_s"] else None
            # ratio of OUR aggregate to DATASET aggregate -> K (==1 means exact 1:1)
            row["K_src"]=rs["sum_sim_s"]/d["data_src"] if d["data_src"] else None
            row["K_tgt"]=rt["sum_sim_s"]/d["data_tgt"] if d["data_tgt"] else None
        results.append(row)
        print(json.dumps(row), flush=True)
        json.dump(results, open(a.out,"w"), indent=2)
    print(f"\nwrote {a.out}")

if __name__=="__main__":
    main()
