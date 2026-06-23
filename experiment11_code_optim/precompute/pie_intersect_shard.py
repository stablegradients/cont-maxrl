"""Compute the PIE-faithful CORRECTNESS-intersection kept-case set per problem,
sharded by problem for SLURM. MULTIPROCESS within a job (72 cores), no coredumps.
Saves {pid, nprog, ncases, kept:[idx], dropped} per problem -> reusable; the cheap
coverage classification (kept x timing) is done separately/locally.

  python pie_intersect_shard.py --split train --shard 0 --nshards 48
"""
from __future__ import annotations
import json, os, re, glob, tempfile, shutil, sys, argparse, resource
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
from experiment11_code_optim.measurement.core import _compile, _run_and_compare

OFF = "/work/nvme/bgtv/sramasubramanian/pie_official/PIE_Dataset"
M = "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
OUT = "/work/nvme/bgtv/sramasubramanian/pie_intersection"
PCH = os.environ.get("PIE_PCH_INCLUDE_PATH")
NAT = float(os.environ.get("PIE_NAT_TIMEOUT", "4"))
_CC = {}


def _cases(pid):
    if pid in _CC: return _CC[pid]
    d = f"{M}/{pid}"
    idx = sorted(int(m.group(1)) for f in os.listdir(d)
                 if (m := re.match(r"input\.(\d+)\.txt$", f))
                 and os.path.exists(f"{d}/output.{m.group(1)}.txt")) if os.path.isdir(d) else []
    _CC[pid] = (idx, {c: open(f"{d}/input.{c}.txt").read() for c in idx},
                {c: open(f"{d}/output.{c}.txt").read() for c in idx})
    return _CC[pid]


def _screen(task):
    pid, sid, code = task
    idx, inp, exp = _cases(pid)
    if not idx: return pid, sid, None
    td = tempfile.mkdtemp(prefix="is_")
    try:
        exe, _ = _compile(code, td, pch_include_path=PCH)
        if exe is None: return pid, sid, None
        fail = [c for c in idx if not _run_and_compare(exe, inp[c], exp[c], NAT)[0]]
        return pid, sid, fail
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
        x = json.loads(l); pid = x["problem_id"]; prob.setdefault(pid, {})
        if x.get("src_id") and x.get("src_code"): prob[pid][x["src_id"]] = x["src_code"]
        if x.get("tgt_id") and x.get("tgt_code"): prob[pid][x["tgt_id"]] = x["tgt_code"]
    pids = sorted(prob)[a.shard::a.nshards]
    tasks = [(pid, sid, c) for pid in pids for sid, c in prob[pid].items()]
    os.makedirs(f"{OUT}/{a.split}", exist_ok=True)
    outp = f"{OUT}/{a.split}/shard_{a.shard}_of_{a.nshards}.jsonl"
    print(f"[intersect {a.split} {a.shard}/{a.nshards}] {len(pids)} problems, "
          f"{len(tasks)} programs, NAT={NAT}s", flush=True)
    fails = {}
    n = 0
    with ProcessPoolExecutor(max_workers=72) as ex:
        for pid, sid, f in ex.map(_screen, tasks, chunksize=4):
            fails[(pid, sid)] = f; n += 1
            if n % 5000 == 0: print(f"  screened {n}/{len(tasks)}", flush=True)
    with open(outp, "w") as fo:
        for pid in pids:
            idx, _, _ = _cases(pid)
            if not idx: continue
            thr = max(5, int(0.05 * len(idx)))
            good = [s for s in prob[pid]
                    if fails.get((pid, s)) is not None and len(fails[(pid, s)]) <= thr]
            kept = set(idx)
            for s in good: kept -= set(fails[(pid, s)])
            fo.write(json.dumps({"pid": pid, "nprog": len(prob[pid]),
                                 "ncases": len(idx), "kept": sorted(kept),
                                 "dropped": len(prob[pid]) - len(good)}) + "\n")
    print(f"[intersect {a.split} {a.shard}/{a.nshards}] DONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()
