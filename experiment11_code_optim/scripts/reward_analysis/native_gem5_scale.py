"""Scaling test for the NATIVE-x86 gem5 reward: N programs x K test cases run 64-way parallel.
Answers (1) correctness at scale (native vs qemu reference + determinism + output-correct) and
(2) throughput (per-step viability). Native gem5 needs STATIC x86 binaries (dynamic-loader issue),
compiled with g++-9 under qemu; the gem5 RUN itself is native aarch64 (no qemu)."""
import json, os, re, subprocess, time, random, sys, queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Explicit CPU pinning: one distinct core per concurrent eval (64 cores, 64 workers -> 1:1,
# no oversubscription, no migration jitter). Each task acquires a core, taskset-pins its
# subprocess to it, releases on completion.
CORE_POOL = queue.Queue()
def _init_cores(n):
    while not CORE_POOL.empty(): CORE_POOL.get()
    for c in range(n): CORE_POOL.put(c)
def pin(cmd):
    core = CORE_POOL.get()
    return ["taskset", "-c", str(core)] + cmd, core

RF     = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
V20    = "/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.opt"   # native aarch64, X86 target
QGEM5  = f"{RF}/home/gem5/build/X86/gem5.opt"                                 # PIE x86 binary (qemu ref)
ENVDIR = "/work/nvme/bgtv/sramasubramanian/envs/gem5build"
CFG    = f"{RF}/home/gem5-skylake-config/gem5-configs"
CFGg   = "/home/gem5-skylake-config/gem5-configs"                            # guest path (under proot)
WD     = "/work/nvme/bgtv/sramasubramanian/qemu_x86"
PROOT  = f"{WD}/proot/usr/bin/proot"; QB = f"{WD}/qu10/usr/bin/qemu-x86_64"
GEN    = f"{RF}/home/pie-perf/data/codenet/generated_test_cases"
RAW    = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"
WORK   = f"{RF}/home/g5scale"; os.makedirs(WORK, exist_ok=True)
GPP_PROOT_ENV = {**os.environ, "LD_LIBRARY_PATH": f"{WD}/proot/usr/lib/aarch64-linux-gnu", "PROOT_NO_SECCOMP": "1"}
NAT_ENV = {**os.environ, "LD_LIBRARY_PATH": f"{ENVDIR}/lib", "PYTHONPATH": f"{CFG}:{CFG}/system"}

N_PROG, K, WORKERS = int(sys.argv[1]) if len(sys.argv)>1 else 256, 4, 64

def sim_ticks(stats):
    try:
        for l in open(stats):
            if l.startswith("sim_ticks"): return int(l.split()[1])
    except: pass
    return None

def select():
    best = {}
    for line in open(RAW):
        r = json.loads(line); pid = r["problem_id"]
        ntc = len([f for f in os.listdir(f"{GEN}/{pid}")]) if os.path.isdir(f"{GEN}/{pid}") else 0
        if ntc >= K*2 and os.path.exists(f"{GEN}/{pid}/output.0.txt"):
            sp = float(r["speedup"])
            if pid not in best or sp > best[pid][0]:
                best[pid] = (sp, r["src_code"])
    pids = sorted(best)
    random.Random(0).shuffle(pids)
    return [(p, best[p][1]) for p in pids[:N_PROG]]

def compile_static(item):
    pid, src = item
    src = re.sub(r'#pragma GCC target.*\n', '', src)
    open(f"{WORK}/{pid}.cpp", "w").write(src)
    cmd, core = pin([PROOT, "-q", QB, "-r", RF, "-w", "/home/g5scale", "/usr/bin/g++",
                     f"/home/g5scale/{pid}.cpp", "-o", f"/home/g5scale/{pid}",
                     "-O3", "-std=c++17", "-static"])
    try:
        p = subprocess.run(cmd, env=GPP_PROOT_ENV, capture_output=True, timeout=180)
        return pid, (p.returncode == 0 and os.path.exists(f"{WORK}/{pid}"))
    except Exception:
        return pid, False
    finally:
        CORE_POOL.put(core)

def run_native(pid, tc):
    out = f"/tmp/scale/{pid}_{tc}"; os.makedirs(out, exist_ok=True)
    inp = f"{GEN}/{pid}/input.{tc}.txt"
    cmd, core = pin([V20, "-d", out, "--stats-file=s.txt", f"{CFG}/run-se.py", "Verbatim", f"{WORK}/{pid}"])
    t0 = time.time()
    try:
        with open(inp) as fh:
            p = subprocess.run(cmd, stdin=fh, env=NAT_ENV, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        CORE_POOL.put(core)
        return {"pid": pid, "tc": tc, "ticks": None, "wall": 90, "ok": False, "to": True}
    CORE_POOL.put(core)
    wall = time.time() - t0
    tk = sim_ticks(f"{out}/s.txt")
    got = p.stdout.decode("utf-8", "ignore").strip()
    try: exp = open(f"{GEN}/{pid}/output.{tc}.txt").read().strip()
    except: exp = None
    ok = exp is not None and (exp in got or got == exp)
    subprocess.run(["rm", "-rf", out], capture_output=True)
    return {"pid": pid, "tc": tc, "ticks": tk, "wall": round(wall, 2), "ok": ok}

def run_qemu_ref(pid, tc):
    out = f"/home/g5scale/qref_{pid}_{tc}"
    inp = f"{GEN}/{pid}/input.{tc}.txt"
    try:
        with open(inp) as fh:
            subprocess.run([PROOT, "-q", QB, "-r", RF, "-w", "/home/g5scale",
                            "env", f"PYTHONPATH={CFGg}:{CFGg}/system",
                            QGEM5.replace(RF, ""), "-d", out, "--stats-file=s.txt",
                            f"{CFGg}/run-se.py", "Verbatim", f"/home/g5scale/{pid}"],
                           stdin=fh, env=GPP_PROOT_ENV, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        return pid, tc, None
    return pid, tc, sim_ticks(f"{RF}{out}/s.txt")

def main():
    os.makedirs("/tmp/scale", exist_ok=True)
    _init_cores(WORKERS)
    progs = select()
    print(f"selected {len(progs)} programs (>= {K*2} test cases each)", flush=True)

    # ---- compile (parallel) ----
    t0 = time.time(); compiled = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for pid, ok in ex.map(compile_static, progs):
            if ok: compiled.append(pid)
    print(f"compiled {len(compiled)}/{len(progs)} static x86 in {time.time()-t0:.0f}s", flush=True)

    # ---- NATIVE gem5 runs: compiled x K test cases, 64-way ----
    tasks = [(pid, tc) for pid in compiled for tc in range(K)]
    print(f"running {len(tasks)} native gem5 evals on {WORKERS} cores...", flush=True)
    t0 = time.time(); results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_native, pid, tc) for pid, tc in tasks]
        for i, f in enumerate(as_completed(futs)):
            results.append(f.result())
            if (i+1) % 100 == 0: print(f"  {i+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    wall_total = time.time() - t0
    nat = {(r["pid"], r["tc"]): r["ticks"] for r in results}

    # ---- NO-PARALLELISM-NOISE test: re-run 40 evals SERIALLY (isolated), compare to their
    #      PARALLEL-batch ticks. All-match => sim_ticks is parallelism-invariant (no noise). ----
    det = []; det_detail = []
    for pid, tc in random.Random(1).sample(tasks, min(24, len(tasks))):
        r2 = run_native(pid, tc)
        m = (r2["ticks"] is not None and r2["ticks"] == nat.get((pid, tc)))
        det.append(m)
        if not m: det_detail.append((pid, tc, nat.get((pid, tc)), r2["ticks"]))

    # ---- qemu reference on a sample (correctness vs PIE-exact) ----
    sample = random.Random(2).sample(compiled, min(12, len(compiled)))
    print(f"qemu reference on {len(sample)} programs (tc0)...", flush=True)
    qref = {}
    with ThreadPoolExecutor(max_workers=min(WORKERS, 32)) as ex:
        for pid, tc, tk in ex.map(lambda pt: run_qemu_ref(*pt), [(p, 0) for p in sample]):
            qref[(pid, tc)] = tk

    # ---- metrics ----
    ok_runs = [r for r in results if r["ticks"] is not None]
    out_ok  = sum(1 for r in results if r["ok"])
    errs = []
    for (pid, tc), q in qref.items():
        n = nat.get((pid, tc))
        if q and n: errs.append(abs(n - q) / q * 100)
    summary = {
        "n_programs": len(compiled), "K": K, "workers": WORKERS,
        "n_evals": len(tasks), "n_ok_ticks": len(ok_runs), "n_output_correct": out_ok,
        "wall_total_s": round(wall_total, 1),
        "throughput_evals_per_s": round(len(tasks)/wall_total, 2),
        "per_eval_mean_wall_s": round(sum(r["wall"] for r in ok_runs)/max(len(ok_runs),1), 2),
        "parallel_vs_serial_match": f"{sum(det)}/{len(det)}",
        "noise_mismatches": det_detail,
        "native_vs_qemu_n": len(errs),
        "native_vs_qemu_mean_err_pct": round(sum(errs)/len(errs), 3) if errs else None,
        "native_vs_qemu_max_err_pct": round(max(errs), 3) if errs else None,
        "per_step_1024xK5_on_64cores_min": round(1024*5/ (len(tasks)/wall_total) /60, 1),
        "per_step_1024xK5_on_288cores_min": round(1024*5/ (len(tasks)/wall_total*(288/WORKERS)) /60, 1),
    }
    json.dump({"summary": summary, "results": results,
               "qref": {f"{k[0]}_{k[1]}": v for k, v in qref.items()},
               "errs": errs},
              open("/tmp/scale_results.json", "w"), indent=2)
    print("\n=== SCALING SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
