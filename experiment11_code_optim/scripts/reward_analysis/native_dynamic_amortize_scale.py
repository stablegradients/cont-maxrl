"""Measure the AMORTIZED all-tc per-step: for N candidate binaries, run startup ONCE then fork a COW child
per test case (run_se_forkloop.py), 256-way across candidates. fork_tick = per-binary min_tc_tick - margin
(from the scale ticks.json, no probe). Reports: amortized vs unamortized wall, exactness vs scale baselines,
projected per-step for 1024 rollouts x all tc. Run: python native_dynamic_amortize_scale.py [N] [MARGIN_FRAC]
"""
import json, os, re, subprocess, sys, time, queue, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

RF   = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
G    = "/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.fast"
ENV  = "/work/nvme/bgtv/sramasubramanian/envs/gem5build"
CFG  = f"{RF}/home/gem5-skylake-config/gem5-configs"
FORKLOOP = f"{CFG}/run_se_forkloop.py"
WORK = f"{RF}/home/g5scale"
MERGED = "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
TICKS = "/work/nvme/bgtv/sramasubramanian/gem5_scale_out/ticks.json"
DATA = "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/data_pie"
NATENV = {**os.environ, "LD_LIBRARY_PATH": f"{ENV}/lib", "PYTHONPATH": f"{CFG}:{CFG}/system"}

N        = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MARGIN   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10   # fork at (1-MARGIN)*min_tc_tick
CORES = sorted(os.sched_getaffinity(0)); WORKERS = len(CORES)
POOL = queue.Queue()
for c in CORES: POOL.put(c)

def sid_to_pid():
    m = {}
    for fn in ("test.jsonl", "val.jsonl", "train.jsonl"):
        for line in open(f"{DATA}/{fn}"):
            r = json.loads(line); m[r["src_id"]] = r["problem_id"]; m[r["tgt_id"]] = r["problem_id"]
    return m

def load_candidates(n):
    T = json.load(open(TICKS))
    by = {}
    for k, v in T.items():
        if isinstance(v["t"], int):
            s, tc = k.split("|"); by.setdefault(s, {})[int(tc)] = v["t"]
    s2p = sid_to_pid()
    cands = []
    for sid, tcs in by.items():
        if len(tcs) < 10 or not os.path.exists(f"{WORK}/{sid}"): continue
        pid = s2p.get(sid)
        if not pid or not os.path.isdir(f"{MERGED}/{pid}"): continue
        # keep tc that exist in merged
        tcl = [(tc, t) for tc, t in sorted(tcs.items()) if os.path.exists(f"{MERGED}/{pid}/input.{tc}.txt")]
        if len(tcl) < 10: continue
        mn = min(t for _, t in tcl)
        cands.append({"sid": sid, "pid": pid, "tcs": tcl, "min_tick": mn,
                      "fork_tick": int(mn * (1 - MARGIN))})
        if len(cands) >= n: break
    return cands

def amortized_run(c):
    core = POOL.get()
    fix = f"/dev/shm/amz_{c['sid']}"; tcf = f"/dev/shm/amztc_{c['sid']}"
    open(tcf, "w").write("\n".join(f"{MERGED}/{c['pid']}/input.{tc}.txt" for tc, _ in c["tcs"]))
    open(fix, "w").write("")
    env = {**NATENV, "GEM5_FIXED_STDIN": fix, "GEM5_TC_LIST": tcf, "GEM5_FORK_TICK": str(c["fork_tick"])}
    out = f"/tmp/amz/{c['sid']}"; os.makedirs(out, exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(["taskset", "-c", str(core), G, "-d", out, "--stats-file=s.txt",
                            FORKLOOP, "Verbatim", f"{WORK}/{c['sid']}"],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        txt = p.stdout.decode("utf-8", "ignore")
    except subprocess.TimeoutExpired:
        POOL.put(core); return {"sid": c["sid"], "wall": 600, "to": True, "amz": {}}
    POOL.put(core)
    wall = time.time() - t0
    amz = {}
    for m in re.finditer(r"TC (\d+) tick (\d+)", txt):
        amz[c["tcs"][int(m.group(1))][0]] = int(m.group(2))
    subprocess.run(["rm", "-rf", out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try: os.remove(fix); os.remove(tcf)
    except OSError: pass
    return {"sid": c["sid"], "wall": wall, "to": False, "amz": amz, "ntc": len(c["tcs"]),
            "fork_tick": c["fork_tick"], "min_tick": c["min_tick"]}

def main():
    os.makedirs("/tmp/amz", exist_ok=True)
    cands = load_candidates(N)
    tot_tc = sum(len(c["tcs"]) for c in cands)
    print(f"[amz] {len(cands)} candidates, {tot_tc} tc total, {WORKERS} cores, margin={MARGIN}", flush=True)
    t0 = time.time(); res = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(amortized_run, c) for c in cands]
        for i, f in enumerate(as_completed(futs)):
            res.append(f.result())
            if (i + 1) % 50 == 0: print(f"  {i+1}/{len(cands)} ({time.time()-t0:.0f}s)", flush=True)
    wall = time.time() - t0
    # exactness vs scale baselines
    base = {c["sid"]: dict(c["tcs"]) for c in cands}
    errs = []; n_exact = 0; n_tc_done = 0; mismatch_bins = 0
    for r in res:
        if r["to"] or not r["amz"]: mismatch_bins += 1; continue
        b = base[r["sid"]]
        for tc, t in r["amz"].items():
            if tc in b and b[tc] > 0:
                e = abs(t - b[tc]) / b[tc] * 100; errs.append(e); n_tc_done += 1
                n_exact += (e < 2.0)
    n_to = sum(1 for r in res if r["to"])
    amz_tc_per_s = n_tc_done / wall if wall else 0
    summary = {
        "n_candidates": len(cands), "total_tc": tot_tc, "amz_tc_evaluated": n_tc_done,
        "wall_s": round(wall, 1), "timeouts": n_to, "bins_no_output": mismatch_bins,
        "amortized_tc_per_s": round(amz_tc_per_s, 1),
        "exactness_within_2pct": round(n_exact / len(errs), 4) if errs else None,
        "exactness_median_err_pct": round(statistics.median(errs), 4) if errs else None,
        "exactness_p99_err_pct": round(statistics.quantiles(errs, n=100)[98], 4) if len(errs) >= 100 else None,
        # per-step projection: 1024 rollouts x (their tc). assume ~100 tc/rollout -> 102400 tc
        "per_step_1024xAllTc_min": round(102400 / amz_tc_per_s / 60, 2) if amz_tc_per_s else None,
    }
    json.dump({"summary": summary, "res": [{k: r[k] for k in ("sid","wall","to","ntc") if k in r} for r in res]},
              open("/work/nvme/bgtv/sramasubramanian/gem5_scale_out/amortize_summary.json", "w"), indent=2)
    print("\n=== AMORTIZED SCALE SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
