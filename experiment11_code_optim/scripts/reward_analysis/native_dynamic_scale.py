"""STEP 4 — scale test: reproduce PIE's exact gem5 timing for N pairs (default 1024) on 256 cores,
native+DYNAMIC (patched gem5, NO qemu), MERGED test cases, exact tc2time set, timeout->inf policy.

Reports: per-side K = our_agg/dataset_agg, per-pair speedup K_sp, dataset-match rates (<2%), throughput
(evals/s), per-step in-loop cost estimate, determinism (serial==parallel ticks), compile/timeout rates.

Resumable + checkpointed: ticks cached in OUT_DIR/ticks.json (skip done on restart); compiled binaries
persist in the rootfs work dir. Run:
  python native_dynamic_scale.py [N_PAIRS] [TIMEOUT_S]      (defaults 1024, 120)
"""
import json, os, re, subprocess, sys, time, queue, statistics, random
from concurrent.futures import ThreadPoolExecutor, as_completed

RF    = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
V20   = "/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.opt"   # PATCHED native gem5
ENV   = "/work/nvme/bgtv/sramasubramanian/envs/gem5build"
CFG   = f"{RF}/home/gem5-skylake-config/gem5-configs"
RUNSE = f"{CFG}/run_se_redirect.py"
LDREAL= f"{RF}/lib/x86_64-linux-gnu/ld-2.31.so"
WD    = "/work/nvme/bgtv/sramasubramanian/qemu_x86"
PROOT = f"{WD}/proot/usr/bin/proot"; QB = f"{WD}/qu10/usr/bin/qemu-x86_64"
MERGED= "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
DATA  = "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/data_pie"
WORK  = f"{RF}/home/g5scale"; os.makedirs(WORK, exist_ok=True)
OUTDIR= "/work/nvme/bgtv/sramasubramanian/gem5_scale_out"; os.makedirs(OUTDIR, exist_ok=True)
TICKS_F = f"{OUTDIR}/ticks.json"
GPPENV = {**os.environ, "LD_LIBRARY_PATH": f"{WD}/proot/usr/lib/aarch64-linux-gnu", "PROOT_NO_SECCOMP": "1"}
NATENV = {**os.environ, "LD_LIBRARY_PATH": f"{ENV}/lib", "PYTHONPATH": f"{CFG}:{CFG}/system"}

N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 120     # PIE's 2-min/test policy
CORES = sorted(os.sched_getaffinity(0)); WORKERS = len(CORES)
COMPILE_WORKERS = min(WORKERS, 160)
CORE_POOL = queue.Queue()
for c in CORES: CORE_POOL.put(c)

def sim_ticks(s):
    try:
        for l in open(s):
            if l.startswith("sim_ticks"): return int(l.split()[1])
    except Exception: pass
    return None

# ---------- selection (dedup binaries by submission id) ----------
def select(n):
    pairs = []; seen = set()
    for fn in ("test.jsonl", "val.jsonl", "train.jsonl"):
        for line in open(f"{DATA}/{fn}"):
            r = json.loads(line)
            pid = r.get("problem_id")
            if not (r.get("src_code") and r.get("tgt_code") and r.get("src_code_tc2time")
                    and r.get("tgt_code_tc2time") and os.path.isdir(f"{MERGED}/{pid}")): continue
            key = (r["src_id"], r["tgt_id"])
            if key in seen: continue
            seen.add(key); pairs.append(r)
            if len(pairs) >= n: return pairs
    return pairs

# ---------- compile (dynamic, retry) ----------
def compile_one(args):
    sub_id, code = args
    binp = f"{WORK}/{sub_id}"
    if os.path.exists(binp): return sub_id, True, "cached"
    code = re.sub(r'#pragma GCC target.*\n', '', code)
    open(f"{WORK}/{sub_id}.cpp", "w").write(code)
    cmd = [PROOT, "-q", QB, "-r", RF, "-w", "/home/g5scale", "/usr/bin/g++",
           f"/home/g5scale/{sub_id}.cpp", "-o", f"/home/g5scale/{sub_id}",
           "-O3", "-std=c++17", f"-Wl,--dynamic-linker={LDREAL}"]
    last = ""
    for attempt in range(3):                                  # retry transient qemu cc1plus aborts
        core = CORE_POOL.get()
        try:
            p = subprocess.run(["taskset", "-c", str(core)] + cmd, env=GPPENV,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except Exception as e:
            p = None; last = str(e)
        finally:
            CORE_POOL.put(core)
        if p is not None and p.returncode == 0 and os.path.exists(binp):
            return sub_id, True, ("ok" if attempt == 0 else f"ok_retry{attempt}")
        if p is not None: last = p.stderr.decode("utf-8", "ignore")[-150:]
    return sub_id, False, last

# ---------- native run (timeout -> None ticks = inf) ----------
def run_one(sub_id, pid, tc, serial_core=None):
    out = f"/tmp/g5scale/{sub_id}_{tc}_{'s' if serial_core is not None else 'p'}"
    os.makedirs(out, exist_ok=True)
    inp = f"{MERGED}/{pid}/input.{tc}.txt"
    if not os.path.exists(inp): return (sub_id, tc, None, False, "no_input")
    base = [V20, "-d", out, "--stats-file=s.txt", RUNSE, "Verbatim", f"{WORK}/{sub_id}"]
    core = serial_core if serial_core is not None else CORE_POOL.get()
    try:
        with open(inp) as fh:
            p = subprocess.run(["taskset", "-c", str(core)] + base, stdin=fh, env=NATENV,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT)
        st = "ok"
    except subprocess.TimeoutExpired:
        if serial_core is None: CORE_POOL.put(core)
        return (sub_id, tc, "TIMEOUT", False, "timeout")
    if serial_core is None: CORE_POOL.put(core)
    tk = sim_ticks(f"{out}/s.txt")
    got = p.stdout.decode("utf-8", "ignore").strip()
    try: exp = open(f"{MERGED}/{pid}/output.{tc}.txt").read().strip()
    except Exception: exp = None
    ok = exp is not None and (exp in got or got == exp)
    subprocess.run(["rm", "-rf", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (sub_id, tc, (tk if tk is not None else "NOTICK"), ok, st if tk is not None else "no_ticks")

def main():
    os.makedirs("/tmp/g5scale", exist_ok=True)
    print(f"[scale] node cores={WORKERS} N_PAIRS={N_PAIRS} timeout={TIMEOUT}s out={OUTDIR}", flush=True)
    pairs = select(N_PAIRS)
    print(f"[scale] selected {len(pairs)} pairs", flush=True)

    # binaries: sub_id -> (code, pid, tc_list)  (tc_list from tc2time keys = PIE's exact set)
    bins = {}; pair_meta = []
    for r in pairs:
        pid = r["problem_id"]
        s_tcs = sorted(int(t) for t in r["src_code_tc2time"].keys())
        t_tcs = sorted(int(t) for t in r["tgt_code_tc2time"].keys())
        bins[r["src_id"]] = (r["src_code"], pid, s_tcs)
        bins[r["tgt_id"]] = (r["tgt_code"], pid, t_tcs)
        pair_meta.append({"pid": pid, "src_id": r["src_id"], "tgt_id": r["tgt_id"],
                          "data_src": float(r["src_agg_runtime"]), "data_tgt": float(r["tgt_agg_runtime"]),
                          "data_sp": float(r["src_agg_runtime"]) / float(r["tgt_agg_runtime"]) if float(r["tgt_agg_runtime"]) else None,
                          "s_tcs": s_tcs, "t_tcs": t_tcs})

    # ---- compile (parallel + retry) ----
    t0 = time.time(); okbin = set(); comp_stats = {"ok": 0, "retry": 0, "fail": 0}
    items = list((sid, c) for sid, (c, _, _) in bins.items())
    print(f"[scale] compiling {len(items)} unique binaries ({COMPILE_WORKERS}-way)...", flush=True)
    with ThreadPoolExecutor(max_workers=COMPILE_WORKERS) as ex:
        for i, (sid, ok, info) in enumerate(ex.map(compile_one, items)):
            if ok: okbin.add(sid); comp_stats["ok"] += 1; comp_stats["retry"] += ("retry" in info)
            else: comp_stats["fail"] += 1
            if (i + 1) % 500 == 0: print(f"   compiled {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[scale] compiled {len(okbin)}/{len(items)} in {time.time()-t0:.0f}s  {comp_stats}", flush=True)

    # ---- build eval task list (resume: skip cached ticks) ----
    ticks = {}
    if os.path.exists(TICKS_F):
        try: ticks = json.load(open(TICKS_F))
        except Exception: ticks = {}
        print(f"[scale] resumed {len(ticks)} cached evals", flush=True)
    tasks = []
    for sid, (_, pid, tcl) in bins.items():
        if sid not in okbin: continue
        for tc in tcl:
            if f"{sid}|{tc}" not in ticks: tasks.append((sid, pid, tc))
    print(f"[scale] running {len(tasks)} native evals on {WORKERS} cores...", flush=True)

    # ---- run (parallel, checkpoint) ----
    t0 = time.time(); done = 0; n_to = 0; n_bad = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_one, sid, pid, tc) for sid, pid, tc in tasks]
        for f in as_completed(futs):
            sid, tc, tk, ok, st = f.result()
            ticks[f"{sid}|{tc}"] = {"t": tk, "ok": ok}
            done += 1; n_to += (st == "timeout"); n_bad += (not ok)
            if done % 5000 == 0:
                json.dump(ticks, open(TICKS_F, "w"))
                r = done / (time.time() - t0)
                print(f"   {done}/{len(tasks)} ({time.time()-t0:.0f}s, {r:.1f} ev/s, to={n_to} bad={n_bad})", flush=True)
    json.dump(ticks, open(TICKS_F, "w"))
    wall = time.time() - t0
    thr = len(tasks) / wall if wall > 0 else 0
    print(f"[scale] ran {len(tasks)} evals in {wall:.0f}s = {thr:.1f} ev/s", flush=True)

    # ---- determinism: re-run 60 fast evals serially, compare ticks ----
    pinned = CORES[0]
    cand = [(sid, pid, tc) for sid, (_, pid, tcl) in bins.items() if sid in okbin
            for tc in tcl if isinstance(ticks.get(f"{sid}|{tc}", {}).get("t"), int)]
    dsample = random.Random(7).sample(cand, min(60, len(cand)))
    det_match = 0; det_bad = []
    for sid, pid, tc in dsample:
        _, _, tk2, _, _ = run_one(sid, pid, tc, serial_core=pinned)
        a = ticks[f"{sid}|{tc}"]["t"]
        if isinstance(tk2, int) and tk2 == a: det_match += 1
        else: det_bad.append([sid, tc, a, tk2])

    # ---- aggregate per side + per pair ----
    def side_agg(sid, tcl):
        tot = 0.0; nbad = 0; nto = 0; nmiss = 0
        for tc in tcl:
            e = ticks.get(f"{sid}|{tc}")
            if e is None: nmiss += 1; continue
            t = e["t"]
            if t == "TIMEOUT": nto += 1; tot = float("inf")
            elif isinstance(t, int): tot += t / 1e12
            else: nmiss += 1
            if not e["ok"]: nbad += 1
        return tot, nbad, nto, nmiss
    rows = []
    for pm in pair_meta:
        if pm["src_id"] not in okbin or pm["tgt_id"] not in okbin: continue
        s_tot, s_bad, s_to, s_miss = side_agg(pm["src_id"], pm["s_tcs"])
        t_tot, t_bad, t_to, t_miss = side_agg(pm["tgt_id"], pm["t_tcs"])
        K_src = (s_tot / pm["data_src"]) if (pm["data_src"] and s_tot != float("inf")) else None
        K_tgt = (t_tot / pm["data_tgt"]) if (pm["data_tgt"] and t_tot != float("inf")) else None
        our_sp = (s_tot / t_tot) if (t_tot not in (0, float("inf")) and s_tot != float("inf")) else None
        K_sp = (our_sp / pm["data_sp"]) if (our_sp and pm["data_sp"]) else None
        rows.append({**pm, "our_src": s_tot, "our_tgt": t_tot, "our_sp": our_sp,
                     "K_src": K_src, "K_tgt": K_tgt, "K_sp": K_sp,
                     "bad": s_bad + t_bad, "to": s_to + t_to, "miss": s_miss + t_miss})

    Ksrc = [r["K_src"] for r in rows if r["K_src"]]
    Ktgt = [r["K_tgt"] for r in rows if r["K_tgt"]]
    Ksp  = [r["K_sp"]  for r in rows if r["K_sp"]]
    Kabs = Ksrc + Ktgt
    def within(xs, tol): return sum(1 for x in xs if abs(x - 1) <= tol) / len(xs) if xs else None
    summary = {
        "node_cores": WORKERS, "n_pairs": len(pairs), "n_pairs_both_compiled": len(rows),
        "n_unique_bins": len(bins), "n_compiled": len(okbin), "compile_stats": comp_stats,
        "n_evals": len(ticks), "throughput_ev_per_s": round(thr, 1),
        "timeout_evals": sum(1 for v in ticks.values() if v["t"] == "TIMEOUT"),
        "output_correct_rate": round(sum(1 for v in ticks.values() if v["ok"]) / len(ticks), 4) if ticks else None,
        "determinism_serial_vs_parallel": f"{det_match}/{len(dsample)}", "determinism_mismatch": det_bad[:10],
        "abs_K_median": round(statistics.median(Kabs), 5) if Kabs else None,
        "abs_K_within_2pct": round(within(Kabs, 0.02), 4), "abs_K_within_5pct": round(within(Kabs, 0.05), 4),
        "speedup_K_median": round(statistics.median(Ksp), 5) if Ksp else None,
        "speedup_K_within_2pct": round(within(Ksp, 0.02), 4), "speedup_K_within_5pct": round(within(Ksp, 0.05), 4),
        "speedup_K_n": len(Ksp), "abs_K_n": len(Kabs),
        "per_step_est_min": {
            "K5_tc":  round(1024 * 5  / thr / 60, 1) if thr else None,
            "K10_tc": round(1024 * 10 / thr / 60, 1) if thr else None,
            "all_tc_~100": round(1024 * 100 / thr / 60, 1) if thr else None,
        },
    }
    json.dump({"summary": summary, "rows": rows}, open(f"{OUTDIR}/scale_summary.json", "w"), indent=2)
    print("\n=== STEP 4 SCALE SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\nwrote {OUTDIR}/scale_summary.json", flush=True)

if __name__ == "__main__":
    main()
