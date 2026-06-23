"""Augmented native correctness screen — persists PER-PROGRAM compile + fail data
(the piece the original intersection discarded). CPU-only, no gem5.

For every unique program (src or tgt id) of every problem in the split:
  compile (g++ -O2 -std=gnu++17 -w), run ALL merged cases natively (hang-timeout,
  coredumps off), compare stdout to expected (line-wise + 1e-3 float tol).

Writes, sharded by problem, to OUT/<split>/shard_<shard>_of_<nshards>.jsonl:
  {pid, ncases, progs: {sid: {compile_ok: bool, n_fail: int, fails: [idx]}}}

Bad-apple / kept / usable derivation is done downstream at dataset-build time from
this raw per-program data; this script bakes no policy.

  python pie_screen_v2.py --split train --shard 0 --nshards 48
"""
from __future__ import annotations
import argparse
import json
import os
import re
import resource
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
from experiment11_code_optim.measurement.core import _compile, _run_and_compare

OFF = "/work/nvme/bgtv/sramasubramanian/pie_official/PIE_Dataset"
M = "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
OUT = os.environ.get("PIE_SCREEN_V2_OUT",
                     "/work/nvme/bgtv/sramasubramanian/pie_screen_v2")
PCH = os.environ.get("PIE_PCH_INCLUDE_PATH")
NAT = float(os.environ.get("PIE_NAT_TIMEOUT", "4"))
_CC = {}


def _cases(pid):
    if pid in _CC:
        return _CC[pid]
    d = f"{M}/{pid}"
    idx = sorted(int(m.group(1)) for f in os.listdir(d)
                 if (m := re.match(r"input\.(\d+)\.txt$", f))
                 and os.path.exists(f"{d}/output.{m.group(1)}.txt")) if os.path.isdir(d) else []
    _CC[pid] = (idx, {c: open(f"{d}/input.{c}.txt").read() for c in idx},
                {c: open(f"{d}/output.{c}.txt").read() for c in idx})
    return _CC[pid]


def _screen(task):
    """compile + run one program on all cases -> (pid, sid, compile_ok, fails)."""
    pid, sid, code = task
    idx, inp, exp = _cases(pid)
    if not idx:
        return pid, sid, None, None
    td = tempfile.mkdtemp(prefix="sv2_")
    try:
        exe, _ = _compile(code, td, pch_include_path=PCH)
        if exe is None:
            return pid, sid, False, []          # compile fail
        fails = [c for c in idx if not _run_and_compare(exe, inp[c], exp[c], NAT)[0]]
        return pid, sid, True, fails
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    prob = {}
    for l in open(f"{OFF}/{a.split}.jsonl"):
        x = json.loads(l)
        pid = x["problem_id"]
        prob.setdefault(pid, {})
        if x.get("src_id") and x.get("src_code"):
            prob[pid][x["src_id"]] = x["src_code"]
        if x.get("tgt_id") and x.get("tgt_code"):
            prob[pid][x["tgt_id"]] = x["tgt_code"]
    pids = sorted(prob)[a.shard::a.nshards]
    tasks = [(pid, sid, c) for pid in pids for sid, c in prob[pid].items()]
    os.makedirs(f"{OUT}/{a.split}", exist_ok=True)
    outp = f"{OUT}/{a.split}/shard_{a.shard}_of_{a.nshards}.jsonl"
    print(f"[screen_v2 {a.split} {a.shard}/{a.nshards}] {len(pids)} problems, "
          f"{len(tasks)} programs, NAT={NAT}s", flush=True)

    res = {}
    n = 0
    with ProcessPoolExecutor(max_workers=72) as ex:
        for pid, sid, ok, fails in ex.map(_screen, tasks, chunksize=4):
            res.setdefault(pid, {})[sid] = (ok, fails)
            n += 1
            if n % 5000 == 0:
                print(f"  screened {n}/{len(tasks)}", flush=True)

    with open(outp, "w") as fo:
        for pid in pids:
            idx, _, _ = _cases(pid)
            if not idx:
                continue
            progs = {}
            for sid, (ok, fails) in res.get(pid, {}).items():
                if ok is None:
                    continue
                progs[sid] = {"compile_ok": bool(ok),
                              "n_fail": (len(fails) if fails is not None else 0),
                              "fails": (fails or [])}
            fo.write(json.dumps({"pid": pid, "ncases": len(idx), "progs": progs}) + "\n")
    print(f"[screen_v2 {a.split} {a.shard}/{a.nshards}] DONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()
