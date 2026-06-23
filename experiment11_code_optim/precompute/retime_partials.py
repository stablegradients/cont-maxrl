"""Re-time the forkloop_partial (and other non-ok) programs from the dataset
sweep WITHOUT the per-program wall cap — generous, full coverage like PIE.

These programs are fast in gem5 SIM time (PIE timed them at ~0.1s median) but
slow in WALL time to simulate all ~104 cases, so our 10-min sweep wall
truncated them. Here we give a generous per-program wall (PIE_SWEEP_FORKLOOP_WALL,
the job sets 6h) and disable the per-case budget (huge), so every case of every
program is timed to completion.

Writes to the SAME split dir as the main sweep but with a `zz_retime_` prefix
so the (sorted, last-wins) loaders in dataset_sweep_plots.py / consolidate pick
the re-timed record over the original partial. Resumable + shardable.

Usage: PIE_TEST_CASE_DIR=<merged> PIE_SWEEP_FORKLOOP_WALL=21600 \
       python retime_partials.py --split train --shard 0 --nshards 48
"""
from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import queue
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
# NOTE: dataset_gem5_sweep reads PIE_SWEEP_FORKLOOP_WALL at import — the job
# must export the generous value BEFORE this import (it does).
from experiment11_code_optim.precompute import dataset_gem5_sweep as base

SWEEP = base.OUT_ROOT
# Per-case budget = 24e9 ticks ~= 120s = PIE's EXACT timeout protocol (cases over
# it are excluded, like PIE). The old NO_CASE_LIMIT ("be generous, no cap") let a
# genuinely-heavy case run UNBOUNDED in gem5, so scattered heavy cases exhausted
# the per-program wall and left programs forkloop_partial — and it was LESS
# PIE-faithful, not more. Re-enabling the budget lets the forkloop complete.
CASE_BUDGET = int(os.environ.get("PIE_RETIME_CASE_BUDGET", base.DEFAULT_BUDGET))


def nonok_keys(split: str) -> set:
    """(pid,sid) NOT status=ok in the MAIN SWEEP output (shard_*.jsonl ONLY).
    Reads only the STABLE main-sweep shards (never the growing retime output) so
    the worklist is DETERMINISTIC across staggered job starts. Reading the live
    retime output made work[shard::nshards] differ per job -> some programs were
    dropped, others double-processed. Per-shard resume (the output file's own
    'done' set, below) skips programs a prior pass already flipped to ok."""
    bad, good = set(), set()
    for f in glob.glob(f"{SWEEP}/{split}/shard_*.jsonl"):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            k = (r["pid"], r["sid"])
            (good if r.get("status") == "ok" else bad).add(k)
    return bad - good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    assert base.gem5_available(), "gem5 toolchain missing"

    keys = nonok_keys(args.split)
    work = [e for e in base.build_worklist(args.split)
            if (e["pid"], e["sid"]) in keys]
    for e in work:
        e["_split"] = args.split
    work.sort(key=lambda e: (e["pid"], e["sid"]))
    mine = work[args.shard::args.nshards]

    outdir = f"{SWEEP}/{args.split}"
    os.makedirs(outdir, exist_ok=True)
    tag = os.environ.get("PIE_RETIME_TAG", "zz_retime")
    outpath = f"{outdir}/{tag}_shard_{args.shard}_of_{args.nshards}.jsonl"
    done = set()
    if os.path.exists(outpath):
        for l in open(outpath):
            try:
                r = json.loads(l)
                if r.get("status") == "ok":
                    done.add((r["pid"], r["sid"]))
            except Exception:
                pass
    todo = [e for e in mine if (e["pid"], e["sid"]) not in done]
    print(f"[retime {args.split} {args.shard}/{args.nshards}] {len(mine)} non-ok "
          f"programs, {len(done)} already redone -> {len(todo)} to run; "
          f"per-program wall={base.FORKLOOP_WALL}s, per-case budget={CASE_BUDGET} "
          f"(~120s/PIE), native_screen={base.NATIVE_SCREEN}", flush=True)

    cores = sorted(os.sched_getaffinity(0))
    nw = max(1, min(len(cores), len(todo) or 1))
    pool: queue.Queue = queue.Queue()
    for c in cores[:nw]:
        pool.put(c)
    scratch = f"/dev/shm/dretime_{os.getpid()}"
    for d in glob.glob("/dev/shm/dretime_*"):
        try:
            os.kill(int(d.rsplit("_", 1)[-1]), 0)
        except (ProcessLookupError, ValueError):
            shutil.rmtree(d, ignore_errors=True)
        except PermissionError:
            pass
    os.makedirs(scratch, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(scratch, ignore_errors=True))

    def run(e):
        cpu = pool.get()
        try:
            return base.time_program(e, cpu, CASE_BUDGET, scratch)
        finally:
            pool.put(cpu)

    t0 = time.time(); n_ok = n_done = 0
    with open(outpath, "a") as fout, ThreadPoolExecutor(max_workers=nw) as ex:
        for fut in as_completed([ex.submit(run, e) for e in todo]):
            r = fut.result()
            fout.write(json.dumps(r) + "\n"); fout.flush()
            n_done += 1
            if r.get("status") == "ok":
                n_ok += 1
            if n_done % 20 == 0:
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(todo)} ({n_ok} now ok) "
                      f"eta {(len(todo)-n_done)/rate/60:.0f}m", flush=True)
    print(f"[retime {args.split} {args.shard}] DONE {n_ok}/{len(todo)} now ok in "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
