"""DEFINITIVE native-x86 gem5 reward validation on a full 256-core Grace node.

Decomposes the headline "is the gem5 speedup reward faithful?" into independently-attributable parts
(per advisor): engine-exactness (native vs same-binary-under-qemu) (+) compile/agg fidelity
(native over ALL test cases vs PIE's published absolute src/tgt_agg_runtime) (+) K=5 sampling cost
(native-K5 vs native-all-tc). Plus determinism-under-load (serial vs 256-way batch ticks),
throughput (-> real per-step cost), and exclusion-rate bias. The reward corr is conditioned on
output-correctness so SE-emulation gaps never silently corrupt the signal.

Native gem5 needs STATIC x86 binaries (dynamic-loader resolves pre-redirect); compile g++-9 -O3 -static
under qemu, but the gem5 RUN is native aarch64 (NO qemu). CPUSET on this node is NON-contiguous
(0-111,144-287) -> pin to os.sched_getaffinity(0), never range(N)."""
import json, os, re, subprocess, time, random, sys, queue, math, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

RF    = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
V20   = "/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.opt"   # native aarch64, X86 target
QGEM5 = f"{RF}/home/gem5/build/X86/gem5.opt"                                 # PIE's x86 gem5 (qemu ref)
ENV   = "/work/nvme/bgtv/sramasubramanian/envs/gem5build"
CFG   = f"{RF}/home/gem5-skylake-config/gem5-configs"
CFGg  = "/home/gem5-skylake-config/gem5-configs"
WD    = "/work/nvme/bgtv/sramasubramanian/qemu_x86"
PROOT = f"{WD}/proot/usr/bin/proot"; QB = f"{WD}/qu10/usr/bin/qemu-x86_64"
GEN   = f"{RF}/home/pie-perf/data/codenet/generated_test_cases"
RAW   = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"
WORK  = f"{RF}/home/g5def"; os.makedirs(WORK, exist_ok=True)
GPPENV = {**os.environ, "LD_LIBRARY_PATH": f"{WD}/proot/usr/lib/aarch64-linux-gnu", "PROOT_NO_SECCOMP": "1"}
NATENV = {**os.environ, "LD_LIBRARY_PATH": f"{ENV}/lib", "PYTHONPATH": f"{CFG}:{CFG}/system"}
OUT_JSON = "/tmp/gem5_definitive.json"

# ---- knobs (override via argv: N_PAIR K N_ABS N_QREF R_DET) ----
N_PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # (src,tgt) pairs for reward corr
K      = int(sys.argv[2]) if len(sys.argv) > 2 else 5      # test cases per side in the RL reward
N_ABS  = int(sys.argv[3]) if len(sys.argv) > 3 else 120    # pairs run over ALL tc (absolute fidelity)
N_QREF = int(sys.argv[4]) if len(sys.argv) > 4 else 200    # (binary,tc) run under qemu (engine exactness)
R_DET  = int(sys.argv[5]) if len(sys.argv) > 5 else 100    # evals re-run serial (determinism)
TC_ABS_CAP = 100                                           # all generated_test_cases (PIE agg is sum over 100)

CORES = sorted(os.sched_getaffinity(0)); WORKERS = len(CORES)
CORE_POOL = queue.Queue()
for c in CORES: CORE_POOL.put(c)
def pin(cmd):
    core = CORE_POOL.get()
    return ["taskset", "-c", str(core)] + cmd, core

def sim_ticks(s):
    try:
        for l in open(s):
            if l.startswith("sim_ticks"): return int(l.split()[1])
    except Exception: pass
    return None

def tcs(pid):
    if not os.path.isdir(f"{GEN}/{pid}"): return []
    idx = []
    for f in os.listdir(f"{GEN}/{pid}"):
        m = re.match(r'input\.(\d+)\.txt$', f)
        if m and os.path.exists(f"{GEN}/{pid}/output.{m.group(1)}.txt"):
            idx.append(int(m.group(1)))
    return sorted(idx)

# ---------------- selection ----------------
def select():
    rows = []
    for line in open(RAW):
        r = json.loads(line)
        if float(r.get("src_agg_runtime", 0)) > 0 and float(r.get("tgt_agg_runtime", 0)) > 0 \
           and r.get("src_code") and r.get("tgt_code"):
            rows.append(r)
    random.Random(0).shuffle(rows)
    sel, seen_pair = [], set()
    for r in rows:
        key = (r["src_id"], r["tgt_id"])
        if key in seen_pair: continue
        if len(tcs(r["problem_id"])) >= K:
            sel.append(r); seen_pair.add(key)
        if len(sel) >= N_PAIR: break
    return sel

# ---------------- compile (static) ----------------
def compile_static(sid, src):
    src = re.sub(r'#pragma GCC target.*\n', '', src)
    open(f"{WORK}/{sid}.cpp", "w").write(src)
    cmd, core = pin([PROOT, "-q", QB, "-r", RF, "-w", "/home/g5def", "/usr/bin/g++",
                     f"/home/g5def/{sid}.cpp", "-o", f"/home/g5def/{sid}",
                     "-O3", "-std=c++17", "-static"])
    try:
        p = subprocess.run(cmd, env=GPPENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        return sid, (p.returncode == 0 and os.path.exists(f"{WORK}/{sid}"))
    except Exception:
        return sid, False
    finally:
        CORE_POOL.put(core)

# ---------------- native run ----------------
def run_native(sid, pid, tc, serial_core=None):
    out = f"/tmp/g5def/{sid}_{tc}_{'s' if serial_core is not None else 'p'}"; os.makedirs(out, exist_ok=True)
    base = [V20, "-d", out, "--stats-file=s.txt", f"{CFG}/run-se.py", "Verbatim", f"{WORK}/{sid}"]
    if serial_core is not None:
        cmd = ["taskset", "-c", str(serial_core)] + base; core = None
    else:
        cmd, core = pin(base)
    t0 = time.time()
    try:
        with open(f"{GEN}/{pid}/input.{tc}.txt") as fh:
            p = subprocess.run(cmd, stdin=fh, env=NATENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        to = False
    except subprocess.TimeoutExpired:
        if core is not None: CORE_POOL.put(core)
        return {"sid": sid, "pid": pid, "tc": tc, "ticks": None, "wall": 120.0, "ok": False, "to": True}
    if core is not None: CORE_POOL.put(core)
    wall = time.time() - t0
    tk = sim_ticks(f"{out}/s.txt")
    got = p.stdout.decode("utf-8", "ignore").strip()
    try: exp = open(f"{GEN}/{pid}/output.{tc}.txt").read().strip()
    except Exception: exp = None
    ok = exp is not None and (exp in got or got == exp)
    subprocess.run(["rm", "-rf", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"sid": sid, "pid": pid, "tc": tc, "ticks": tk, "wall": round(wall, 2), "ok": ok, "to": False}

# ---------------- qemu reference (engine exactness, SAME static binary) ----------------
def run_qemu_ref(sid, pid, tc):
    out = f"/home/g5def/qref_{sid}_{tc}"
    try:
        with open(f"{GEN}/{pid}/input.{tc}.txt") as fh:
            subprocess.run([PROOT, "-q", QB, "-r", RF, "-w", "/home/g5def",
                            "env", f"PYTHONPATH={CFGg}:{CFGg}/system",
                            QGEM5.replace(RF, ""), "-d", out, "--stats-file=s.txt",
                            f"{CFGg}/run-se.py", "Verbatim", f"/home/g5def/{sid}"],
                           stdin=fh, env=GPPENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
    except subprocess.TimeoutExpired:
        return (sid, tc), None
    tk = sim_ticks(f"{RF}{out}/s.txt")
    subprocess.run(["rm", "-rf", f"{RF}{out}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (sid, tc), tk

def pmap(fn, items, workers, tag):
    res = []; t0 = time.time(); n = len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, *it) if isinstance(it, tuple) else ex.submit(fn, it) for it in items]
        for i, f in enumerate(as_completed(futs)):
            res.append(f.result())
            if (i + 1) % 250 == 0: print(f"  [{tag}] {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    return res, time.time() - t0

# ---------------- main ----------------
def main():
    os.makedirs("/tmp/g5def", exist_ok=True)
    print(f"node cores={WORKERS} (noncontig ok) | N_PAIR={N_PAIR} K={K} N_ABS={N_ABS} N_QREF={N_QREF} R_DET={R_DET}", flush=True)
    pairs = select()
    print(f"selected {len(pairs)} pairs", flush=True)

    # unique binaries to compile (src + tgt across all pairs)
    bins = {}
    for r in pairs:
        bins[r["src_id"]] = r["src_code"]; bins[r["tgt_id"]] = r["tgt_code"]
    print(f"compiling {len(bins)} unique static binaries...", flush=True)
    comp, ct = pmap(compile_static, list(bins.items()), WORKERS, "compile")
    okbin = {sid for sid, ok in comp if ok}
    print(f"compiled {len(okbin)}/{len(bins)} in {ct:.0f}s", flush=True)

    # absolute-fidelity subsample: pairs whose BOTH sides compiled
    good_pairs = [r for r in pairs if r["src_id"] in okbin and r["tgt_id"] in okbin]
    abs_pairs = good_pairs[:N_ABS]
    abs_bins = {}
    for r in abs_pairs:
        abs_bins[r["src_id"]] = r["problem_id"]; abs_bins[r["tgt_id"]] = r["problem_id"]

    # ---- native eval task list: K-tc for every compiled binary, + ALL-tc for absolute subsample ----
    tasks = set()
    pid_of = {}
    for r in good_pairs:
        for sid in (r["src_id"], r["tgt_id"]):
            pid_of[sid] = r["problem_id"]
            for tc in tcs(r["problem_id"])[:K]:
                tasks.add((sid, tc))
    abs_alltc = {}  # sid -> list of all tc (capped)
    for sid, pid in abs_bins.items():
        all_tc = tcs(pid)[:TC_ABS_CAP]
        abs_alltc[sid] = all_tc
        for tc in all_tc:
            tasks.add((sid, tc))
    tasks = sorted(tasks)
    print(f"native eval: {len(tasks)} (binary,tc) on {WORKERS} cores "
          f"[{len(good_pairs)} good pairs, {len(abs_pairs)} abs-subsample]...", flush=True)
    nat_res, nt = pmap(lambda s, t: run_native(s, pid_of.get(s, abs_bins.get(s)), t),
                       tasks, WORKERS, "native")
    nat = {(r["sid"], r["tc"]): r for r in nat_res}
    thr = len(tasks) / nt
    # checkpoint the expensive native data immediately (survives a node cutoff before metrics)
    json.dump({"nat": [{"sid": r["sid"], "pid": r["pid"], "tc": r["tc"], "ticks": r["ticks"],
                        "ok": r["ok"], "to": r["to"], "wall": r["wall"]} for r in nat_res],
               "pairs": [{"src": r["src_id"], "tgt": r["tgt_id"], "pid": r["problem_id"],
                          "speedup": float(r["speedup"]), "src_agg": float(r["src_agg_runtime"]),
                          "tgt_agg": float(r["tgt_agg_runtime"])} for r in pairs],
               "abs_sids": list(abs_bins.keys()), "okbin": list(okbin), "throughput": thr,
               "native_wall_s": round(nt, 1)},
              open("/tmp/gem5_definitive_nat.json", "w"))
    print(f"  checkpoint -> /tmp/gem5_definitive_nat.json ({len(nat)} evals, thr={thr:.2f}/s)", flush=True)

    # ---- engine exactness: qemu ref on a sample of the SAME static binaries ----
    qsample = random.Random(2).sample([t for t in tasks if nat[t]["ticks"] is not None],
                                       min(N_QREF, len(tasks)))
    print(f"qemu-ref (engine exactness) on {len(qsample)} (binary,tc)...", flush=True)
    qref_res, qt = pmap(lambda s, t: run_qemu_ref(s, pid_of.get(s, abs_bins.get(s)), t),
                        qsample, min(WORKERS, 200), "qref")
    qref = {k: v for k, v in qref_res}

    # ---- determinism under load: re-run fast successful evals SERIALLY, compare ticks ----
    fast = [t for t in tasks if nat[t]["ticks"] is not None and nat[t]["wall"] < 15 and not nat[t]["to"]]
    dsample = random.Random(3).sample(fast, min(R_DET, len(fast)))
    print(f"determinism: re-running {len(dsample)} evals SERIALLY (isolated)...", flush=True)
    det_match, det_mismatch = 0, []
    for sid, tc in dsample:
        r2 = run_native(sid, pid_of.get(sid, abs_bins.get(sid)), tc, serial_core=CORES[0])
        a = nat[(sid, tc)]["ticks"]
        if r2["ticks"] is not None and r2["ticks"] == a: det_match += 1
        else: det_mismatch.append([sid, tc, a, r2["ticks"]])

    # ================= METRICS =================
    def med(x): return round(statistics.median(x), 4) if x else None
    # (1) engine exactness: native vs qemu, same binary/input
    eng = []
    for (sid, tc), q in qref.items():
        n = nat[(sid, tc)]["ticks"]
        if q and n: eng.append(abs(n - q) / q * 100)
    # (2) absolute compile/agg fidelity: native sum over ALL tc /1e12 vs dataset *_agg_runtime
    abs_pts = []   # (which, native_s, dataset_s)
    for r in abs_pairs:
        for sid, dskey in ((r["src_id"], "src_agg_runtime"), (r["tgt_id"], "tgt_agg_runtime")):
            tcl = abs_alltc.get(sid, [])
            ticks = [nat[(sid, tc)]["ticks"] for tc in tcl]
            okc   = [nat[(sid, tc)]["ok"] for tc in tcl]
            if tcl and all(t is not None for t in ticks) and all(okc):
                abs_pts.append((sid, sum(ticks)/1e12, float(r[dskey]), len(tcl)))
    abs_ratio = [a[1]/a[2] for a in abs_pts if a[2] > 0]
    # (3) K=5 sampling cost: native sum over first K tc vs native sum over ALL tc (same binary)
    k5_ratio = []
    for sid, tcl in abs_alltc.items():
        ka = tcl[:K]
        tall = [nat[(sid, tc)]["ticks"] for tc in tcl]
        tk5  = [nat[(sid, tc)]["ticks"] for tc in ka]
        if tall and all(t is not None for t in tall) and all(t is not None for t in tk5) and sum(tall) > 0:
            k5_ratio.append((sum(tk5)/len(ka)) * len(tcl) / sum(tall))  # scaled K5 estimate vs true sum
    # (4) REWARD corr: native speedup vs dataset speedup, conditioned on output-correct on all K tc
    corr_pts = []      # (log_native_sp, log_ds_sp, ds_sp)
    excl = {"compile": 0, "timeout": 0, "wrong": 0, "ok": 0}
    excl_sp, incl_sp = [], []
    for r in pairs:
        s, t = r["src_id"], r["tgt_id"]; pid = r["problem_id"]; dssp = float(r["speedup"])
        if s not in okbin or t not in okbin:
            excl["compile"] += 1; excl_sp.append(dssp); continue
        ktc = tcs(pid)[:K]
        recs = [nat.get((s, tc)) for tc in ktc] + [nat.get((t, tc)) for tc in ktc]
        if any(x is None for x in recs) or any(x["to"] for x in recs):
            excl["timeout"] += 1; excl_sp.append(dssp); continue
        if not all(x["ok"] for x in recs):
            excl["wrong"] += 1; excl_sp.append(dssp); continue
        ssum = sum(nat[(s, tc)]["ticks"] for tc in ktc)
        tsum = sum(nat[(t, tc)]["ticks"] for tc in ktc)
        if ssum > 0 and tsum > 0:
            nsp = ssum / tsum
            corr_pts.append((math.log(nsp), math.log(dssp), dssp, nsp))
            excl["ok"] += 1; incl_sp.append(dssp)

    def pearson(xy):
        if len(xy) < 3: return None
        xs = [a for a, b in xy]; ys = [b for a, b in xy]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        num = sum((a-mx)*(b-my) for a, b in xy)
        den = math.sqrt(sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys))
        return num/den if den else None
    def spearman(xy):
        if len(xy) < 3: return None
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i]); rk = [0]*len(v)
            for r_, i in enumerate(order): rk[i] = r_
            return rk
        rx = ranks([a for a, b in xy]); ry = ranks([b for a, b in xy])
        return pearson(list(zip(rx, ry)))

    # (4b) corr CEILING: on the abs subsample, native speedup over FULL 100 tc vs over K5 (SAME pairs)
    #      -> isolates K=5 sampling cost on the corr itself (output-correct on every tc, both sides).
    abs_full, abs_k5 = [], []
    for r in abs_pairs:
        s, t = r["src_id"], r["tgt_id"]; dssp = float(r["speedup"])
        stc, ttc = abs_alltc.get(s, []), abs_alltc.get(t, [])
        if not stc or not ttc: continue
        sf_recs = [nat[(s, tc)] for tc in stc]; tf_recs = [nat[(t, tc)] for tc in ttc]
        if any(x["ticks"] is None for x in sf_recs + tf_recs): continue
        if not all(x["ok"] for x in sf_recs + tf_recs): continue
        sf = sum(x["ticks"] for x in sf_recs); tf = sum(x["ticks"] for x in tf_recs)
        s5 = sum(nat[(s, tc)]["ticks"] for tc in stc[:K]); t5 = sum(nat[(t, tc)]["ticks"] for tc in ttc[:K])
        if sf > 0 and tf > 0 and s5 > 0 and t5 > 0:
            abs_full.append((math.log(sf/tf), math.log(dssp)))
            abs_k5.append((math.log(s5/t5), math.log(dssp)))

    loglog = [(a, b) for a, b, _, _ in corr_pts]
    summary = {
        "node_cores": WORKERS, "N_PAIR": N_PAIR, "K": K,
        "n_pairs_selected": len(pairs), "n_unique_bins": len(bins), "n_compiled": len(okbin),
        "n_native_evals": len(tasks), "native_wall_s": round(nt, 1),
        "throughput_evals_per_s": round(thr, 2),
        "native_eval_wall_median_s": med([nat[t]["wall"] for t in tasks if nat[t]["ticks"] is not None]),
        "native_eval_wall_mean_s": round(statistics.mean([nat[t]["wall"] for t in tasks if nat[t]["ticks"] is not None]), 2),
        # decomposition
        "engine_exactness_n": len(eng),
        "engine_native_vs_qemu_mean_pct": round(statistics.mean(eng), 4) if eng else None,
        "engine_native_vs_qemu_max_pct": round(max(eng), 4) if eng else None,
        "abs_fidelity_n": len(abs_pts),
        "abs_native_over_dataset_median": med(abs_ratio),
        "abs_native_over_dataset_p10_p90": [round(statistics.quantiles(abs_ratio, n=10)[0], 4),
                                            round(statistics.quantiles(abs_ratio, n=10)[8], 4)] if len(abs_ratio) >= 10 else None,
        "abs_corr_loglog": round(pearson([(math.log(a[1]), math.log(a[2])) for a in abs_pts if a[1] > 0 and a[2] > 0]), 4) if len(abs_pts) >= 3 else None,
        "k5_sampling_ratio_median": med(k5_ratio),
        "k5_sampling_ratio_p10_p90": [round(statistics.quantiles(k5_ratio, n=10)[0], 4),
                                      round(statistics.quantiles(k5_ratio, n=10)[8], 4)] if len(k5_ratio) >= 10 else None,
        # the reward
        "reward_n_pairs": len(corr_pts),
        "reward_pearson_loglog": round(pearson(loglog), 4) if len(loglog) >= 3 else None,
        "reward_spearman": round(spearman(loglog), 4) if len(loglog) >= 3 else None,
        "reward_ds_speedup_range": [round(min(p[2] for p in corr_pts), 3), round(max(p[2] for p in corr_pts), 3)] if corr_pts else None,
        # corr ceiling (full 100-tc) vs K5 on the SAME abs-subsample pairs -> sampling cost on the corr
        "reward_corr_ceiling_n": len(abs_full),
        "reward_corr_ceiling_fulltc_pearson": round(pearson(abs_full), 4) if len(abs_full) >= 3 else None,
        "reward_corr_ceiling_fulltc_spearman": round(spearman(abs_full), 4) if len(abs_full) >= 3 else None,
        "reward_corr_k5_on_abs_pearson": round(pearson(abs_k5), 4) if len(abs_k5) >= 3 else None,
        "reward_corr_k5_on_abs_spearman": round(spearman(abs_k5), 4) if len(abs_k5) >= 3 else None,
        # determinism
        "determinism_match": f"{det_match}/{len(dsample)}", "determinism_mismatch": det_mismatch,
        # exclusions + bias
        "exclusions": excl,
        "excluded_ds_speedup_median": med(excl_sp), "included_ds_speedup_median": med(incl_sp),
        "output_correct_rate": round(sum(1 for t in tasks if nat[t]["ok"]) / len(tasks), 4),
        "timeout_rate": round(sum(1 for t in tasks if nat[t]["to"]) / len(tasks), 4),
        # cost extrapolation
        "qemu_ref_wall_s": round(qt, 1),
        "per_step_1024xK5_min_at_thisnode": round(1024*K / thr / 60, 1),
    }
    json.dump({"summary": summary,
               "corr_pts": [{"log_nsp": a, "log_dssp": b, "dssp": c, "nsp": d} for a, b, c, d in corr_pts],
               "abs_pts": [{"sid": a, "native_s": b, "dataset_s": c, "ntc": d} for a, b, c, d in abs_pts],
               "ceiling_pts": [{"log_full_nsp": f[0], "log_k5_nsp": k[0], "log_dssp": f[1]}
                               for f, k in zip(abs_full, abs_k5)],
               "k5_ratio": k5_ratio, "engine_err_pct": eng,
               "native_walls": {f"{k[0]}_{k[1]}": nat[k]["wall"] for k in tasks}},
              open(OUT_JSON, "w"), indent=2)
    print("\n=== DEFINITIVE SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\nwrote {OUT_JSON}")

if __name__ == "__main__":
    main()
