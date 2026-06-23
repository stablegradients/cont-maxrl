"""Per-test-case gem5 timing for EVERY unique src/tgt program in a PIE split.

For each unique (problem_id, submission_id) program in the split:
  - cross-compile aarch64->x86_64 g++-9 -O3 (#42)
  - probe the startup tick, then run the fork-amortized forkloop over ALL
    merged test cases for the problem (one startup sim, COW fork per case)
  - per case: keep the absolute completion tick if it exited normally; mark it
    EXCLUDED if it hit the per-case tick budget (paper protocol: cases whose
    gem5 sim exceeds ~2 min are dropped) or exited abnormally
Writes one JSONL line per program to out/<split>/shard_<i>.jsonl:
  {pid, sid, roles:[src|tgt], split, n_cases_total, per_tc:{idx:tick},
   excluded:[idx...], sum_ticks, dataset_agg_runtime}

Sharded + resumable: pass --shard i --nshards N; programs[i::N] are processed,
already-done (pid,sid) lines in the shard file are skipped.

Usage:
  PIE_TEST_CASE_DIR=<merged> python dataset_gem5_sweep.py \
      --split train --shard 0 --nshards 64 [--budget_ticks 24000000000]
"""
from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
from experiment11_code_optim.measurement.gem5_backend import (  # validated low-level
    _GEM5, _SEQ, _NATENV, _FORK_MARGIN, _cross_compile, _probe_startup,
    gem5_available,
)

PIE_OFFICIAL = "/work/nvme/bgtv/sramasubramanian/pie_official/PIE_Dataset"
MERGED = os.environ.get("PIE_TEST_CASE_DIR",
                        "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases")
OUT_ROOT = "/work/nvme/bgtv/sramasubramanian/gem5_dataset_sweep"
# per-case RELATIVE tick budget (from the fork point). ~2.4e10 ticks ~= 120s
# wall of post-startup compute at gem5.fast's ~2e8 ticks/wall-s -> the paper's
# "exclude cases that take >2 min" protocol. Cases over budget are EXCLUDED.
DEFAULT_BUDGET = 24_000_000_000
# Per-PROGRAM wall cap. The forkloop is SEQUENTIAL over ~100 cases, so a
# compute-heavy program (cases each near the 120s per-case budget) can run
# 30min+ and monopolize a worker. Cap it: partial-capture keeps every case
# that finished, the program is flagged forkloop_partial, and no worker is
# held longer than this. 600s lets all but the genuinely heavy tail finish.
FORKLOOP_WALL = float(os.environ.get("PIE_SWEEP_FORKLOOP_WALL", "600"))
# Native host g++ (aarch64) for the PRE-SCREEN: some inputs make a correct-ish
# program loop forever (EOF-spin getchar() at end-of-input, sentinel-less inputs,
# special chars). Such a case stalls the single gem5 forkloop process — the
# per-case budget is in sim-TICKS, but a syscall-spin advances ~no ticks so it
# never trips the budget and instead burns the whole per-program wall. We screen
# every case natively first (fast) and drop the ones that hang, exactly as PIE's
# 120s-timeout protocol excluded them. Native termination is arch-independent for
# these I/O-spin loops, so it faithfully predicts the x86 gem5 run.
_NATIVE_GXX = shutil.which("g++") or "/opt/cray/pe/gcc-native/14/bin/g++"
NATIVE_SCREEN = os.environ.get("PIE_NATIVE_SCREEN", "1") != "0"
NATIVE_SCREEN_WALL = float(os.environ.get("PIE_NATIVE_SCREEN_WALL", "10"))


def tc_indices(pid: str) -> list[int]:
    d = f"{MERGED}/{pid}"
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        m = re.match(r"input\.(\d+)\.txt$", f)
        if m and os.path.exists(f"{d}/output.{m.group(1)}.txt"):
            out.append(int(m.group(1)))
    return sorted(out)


def _worklist_from(jsonl: str) -> dict:
    """Unique programs in a jsonl: {(pid,sid): {pid,sid,code,roles,agg_runtime}}.
    agg_runtime is role-invariant per program (verified), so first-seen wins."""
    progs: dict[tuple, dict] = {}
    for line in open(jsonl):
        r = json.loads(line)
        pid = r["problem_id"]
        for role, idk, codek, aggk in (("src", "src_id", "src_code", "src_agg_runtime"),
                                       ("tgt", "tgt_id", "tgt_code", "tgt_agg_runtime")):
            sid = r.get(idk)
            if not sid:
                continue
            key = (pid, sid)
            e = progs.get(key)
            if e is None:
                progs[key] = {"pid": pid, "sid": sid, "code": r[codek],
                              "roles": {role}, "agg_runtime": r.get(aggk)}
            else:
                e["roles"].add(role)
    return progs


def build_worklist(split: str) -> list[dict]:
    """Programs to time. `synthetic_only` = train_with_synthetic MINUS train
    (the 2,970 net-new synthetic programs), so train is never timed twice."""
    if split == "synthetic_only":
        full = _worklist_from(f"{PIE_OFFICIAL}/train_with_synthetic.jsonl")
        train_keys = set(_worklist_from(f"{PIE_OFFICIAL}/train.jsonl"))
        progs = {k: v for k, v in full.items() if k not in train_keys}
    else:
        progs = _worklist_from(f"{PIE_OFFICIAL}/{split}.jsonl")
    out = list(progs.values())
    for e in out:
        e["roles"] = sorted(e["roles"])
    out.sort(key=lambda e: (e["pid"], e["sid"]))           # deterministic shard
    return out


def _run_capture(argv, env, timeout):
    """Run argv in its own process group; on wall timeout SIGKILL the group
    (gem5 m5.fork()s real children) but RETURN the partial stdout drained so
    far. -> (stdout_bytes, timed_out_bool). Unlike gem5_backend._run_pg this
    keeps the per-case lines the forkloop already flushed before the cap."""
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, _ = p.communicate(timeout=30)
        except Exception:
            out = b""
        return out, True


def _native_screen(code, inp_paths, workdir):
    """Compile `code` with the native host g++ and run each case under a short
    WALL timeout to find inputs the program FAILS on: it either HANGS (EOF-spin,
    sentinel-less input) or CRASHES (segfault/abort -> died by signal). Both
    would otherwise break the single gem5 forkloop (a hang stalls it; a crash
    panics the child, which is then silently dropped). Returns the set of input
    INDICES to exclude, mirroring PIE's protocol (PIE's native run also times
    out hangs and fails on crashes). Native behavior is ~architecture-independent
    for these, so the aarch64 screen faithfully predicts the x86 gem5 run. On
    native-compile failure returns set() (no screening; forkloop runs as before)."""
    if not NATIVE_SCREEN or not inp_paths:
        return set()
    src = os.path.join(workdir, "nat.cpp")
    exe = os.path.join(workdir, "nat.out")
    with open(src, "w") as f:
        f.write(code)
    try:
        cp = subprocess.run([_NATIVE_GXX, "-O2", "-std=gnu++17", "-w", "-o", exe, src],
                            capture_output=True, timeout=90)
    except (subprocess.TimeoutExpired, OSError):
        return set()
    if cp.returncode != 0 or not os.path.exists(exe):
        return set()

    def chk(item):
        idx, path = item
        with open(path, "rb") as fh:
            data = fh.read()
        p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            p.communicate(input=data, timeout=NATIVE_SCREEN_WALL)
            # exclude cases that CRASH (died by signal, e.g. SIGSEGV) too: the
            # program fails that input, so gem5 would also abort it (and PIE's
            # native run would exclude it). returncode < 0 == killed by signal.
            return idx, (p.returncode is not None and p.returncode < 0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                p.communicate(timeout=10)
            except Exception:
                pass
            return idx, True
        except Exception:
            return idx, False           # crash/other: let gem5 see it
    hang = set()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for idx, hung in ex.map(chk, inp_paths):
            if hung:
                hang.add(idx)
    return hang


def forkloop_per_tc(binp, inp_paths, fork_tick, workdir, cpu, budget):
    """Per-case parse. Returns (per_tc {idx->tick}, excluded [idx], complete,
    error). Keeps normally-exited cases and lists budget/abnormal ones; on a
    wall-timeout/crash it still returns the cases captured before the cut
    (complete=False), so compute-heavy programs aren't wholly lost."""
    fix = os.path.join(workdir, "fixed_stdin")
    tcf = os.path.join(workdir, "tc_list")
    shutil.copyfile(inp_paths[0][1], fix)
    with open(tcf, "w") as f:
        f.write("\n".join(p for _, p in inp_paths))   # order == inp_paths order
    out = os.path.join(workdir, "fl_m5out")
    os.makedirs(out, exist_ok=True)
    argv = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [
        _GEM5, "-d", out, "--stats-file=s.txt", _SEQ, "Verbatim", binp,
    ]
    env = {**_NATENV, "GEM5_FIXED_STDIN": fix, "GEM5_TC_LIST": tcf,
           "GEM5_FORK_TICK": str(fork_tick),
           "GEM5_TC_MAX_TICKS": str(int(budget))}      # RELATIVE post-fork budget
    try:
        raw, timed_out = _run_capture(argv, env, FORKLOOP_WALL)
    finally:
        shutil.rmtree(out, ignore_errors=True)
        for p in (fix, tcf):
            try:
                os.remove(p)
            except OSError:
                pass
    so = (raw or b"").decode("utf-8", "ignore")
    ms = re.search(r"STARTUP cause=(.+) tick=(\d+)", so)
    if not ms or "limit reached" not in ms.group(1) or int(ms.group(2)) != fork_tick:
        return {}, [], False, ("forkloop wall timeout" if timed_out
                               else "forkloop bad startup phase")
    pairs = re.findall(r"TC (\d+) tick (\d+) cause (.+)", so)
    per_tc: dict[int, int] = {}
    excluded: list[int] = []
    for pos, tick, cause in pairs:
        idx = inp_paths[int(pos)][0]
        if cause.strip().startswith("exiting"):
            per_tc[idx] = int(tick)
        else:                                  # budget_exceeded or abnormal exit
            excluded.append(idx)
    # complete iff the loop ran every case (sentinel present) and nothing was
    # cut by the wall — mirrors gem5_backend._forkloop_ticks' len-check.
    complete = (not timed_out
                and f"FORKLOOP_DONE ntc={len(inp_paths)}" in so
                and len(pairs) == len(inp_paths))
    return per_tc, sorted(excluded), complete, None


def time_program(e: dict, cpu: int, budget: int, scratch: str) -> dict:
    idxs = tc_indices(e["pid"])
    base = {"pid": e["pid"], "sid": e["sid"], "roles": e["roles"],
            "split": e["_split"], "n_cases_total": len(idxs),
            "dataset_agg_runtime": e["agg_runtime"]}
    if not idxs:
        return {**base, "status": "no_test_cases"}
    inp_paths = [(i, f"{MERGED}/{e['pid']}/input.{i}.txt") for i in idxs]
    workdir = tempfile.mkdtemp(dir=scratch, prefix="ds_")
    try:
        binp, err = _cross_compile(e["code"], workdir, cpu)
        if binp is None:
            return {**base, "status": "compile_fail", "error": err}
        # PRE-SCREEN: drop cases that loop forever natively (they would stall the
        # single gem5 forkloop, whose tick-budget can't catch a syscall-spin).
        # Excluding them mirrors PIE's 120s-timeout protocol.
        hang = _native_screen(e["code"], inp_paths, workdir)
        survivors = [ip for ip in inp_paths if ip[0] not in hang]
        if not survivors:
            return {**base, "status": "native_all_excluded", "no_stdin": False,
                    "per_tc": {}, "excluded": sorted(hang),
                    "native_excluded": sorted(hang), "sum_ticks": 0, "n_captured": 0}
        startup, serr = _probe_startup(binp, survivors[0][1], workdir, cpu)
        if startup is None:
            return {**base, "status": "startup_fail", "error": serr,
                    "native_excluded": sorted(hang)}
        if serr == "no_stdin":               # input-invariant: every case == startup
            return {**base, "status": "ok", "no_stdin": True,
                    "per_tc": {str(i): startup for i in idxs}, "excluded": [],
                    "sum_ticks": startup * len(idxs)}
        if startup <= _FORK_MARGIN + 1000:
            return {**base, "status": "startup_too_small"}
        per_tc, excluded, complete, ferr = forkloop_per_tc(
            binp, survivors, startup - _FORK_MARGIN, workdir, cpu, budget)
        if ferr and not per_tc:
            return {**base, "status": "forkloop_fail", "error": ferr,
                    "native_excluded": sorted(hang)}
        excluded = sorted(set(excluded) | hang)        # budget-excl + native hangs
        # complete = every SURVIVING case ran (kept or budget-excluded); hanging
        # cases are excluded like PIE. Partial only if the forkloop itself was cut.
        return {**base,
                "status": "ok" if complete else "forkloop_partial",
                "no_stdin": False,
                "per_tc": {str(k): v for k, v in per_tc.items()},
                "excluded": excluded, "native_excluded": sorted(hang),
                "sum_ticks": sum(per_tc.values()),
                "n_captured": len(per_tc) + len(excluded)}
    except Exception as ex:
        return {**base, "status": f"error:{type(ex).__name__}", "error": str(ex)[:200]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budget_ticks", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("PIE_SWEEP_WORKERS", "0")) or None)
    args = ap.parse_args()
    assert gem5_available(), "gem5 toolchain missing on this host"

    work = build_worklist(args.split)
    for e in work:
        e["_split"] = args.split
    mine = work[args.shard::args.nshards]
    outdir = f"{OUT_ROOT}/{args.split}"
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/shard_{args.shard}_of_{args.nshards}.jsonl"

    done: set[tuple] = set()
    if os.path.exists(outpath):
        for line in open(outpath):
            try:
                r = json.loads(line)
                done.add((r["pid"], r["sid"]))
            except Exception:
                pass
    todo = [e for e in mine if (e["pid"], e["sid"]) not in done]
    print(f"[{args.split} shard {args.shard}/{args.nshards}] {len(mine)} programs "
          f"({len(done)} already done) -> {len(todo)} to run; budget={args.budget_ticks}",
          flush=True)

    cores = sorted(os.sched_getaffinity(0))
    nw = args.workers or len(cores)
    nw = max(1, min(nw, len(cores), len(todo) or 1))
    pool: queue.Queue = queue.Queue()
    for c in cores[:nw]:
        pool.put(c)
    # /dev/shm scratch: sweep dead-pid leftovers, register cleanup for this run
    for d in glob.glob("/dev/shm/dsweep_*"):
        try:
            pid = int(d.rsplit("_", 1)[-1])
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
        except (ValueError, PermissionError):
            pass
    scratch = f"/dev/shm/dsweep_{os.getpid()}"
    os.makedirs(scratch, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(scratch, ignore_errors=True))

    def run(e):
        cpu = pool.get()
        try:
            return time_program(e, cpu, args.budget_ticks, scratch)
        finally:
            pool.put(cpu)

    t0 = time.time()
    n_ok = n_done = 0
    with open(outpath, "a") as fout, ThreadPoolExecutor(max_workers=nw) as ex:
        for fut in as_completed([ex.submit(run, e) for e in todo]):
            r = fut.result()
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            n_done += 1
            if r.get("status") == "ok":
                n_ok += 1
            if n_done % 100 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(todo) - n_done) / rate if rate else 0
                print(f"  {n_done}/{len(todo)} ({n_ok} ok) "
                      f"{rate:.1f} prog/s eta {eta/60:.0f}m", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[{args.split} shard {args.shard}] DONE {n_ok}/{len(todo)} ok in "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
