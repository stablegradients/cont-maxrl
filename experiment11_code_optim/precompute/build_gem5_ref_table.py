"""Build the gem5 reference table (PIE_GEM5_REF_TABLE) for overfit problems.

For each problem: locate the src program embedded in the overfit parquet's
prompt, match it to its train.jsonl pair (src_id, tgt_code), and time BOTH
src and tgt with the SAME pipeline the RL reward uses:
  - inputs = pie_reward_fn._subsample_cases(_get_test_cases(...), K, pid)
    (identical deterministic subset as the gate and the rollout timing)
  - ticks  = measurement.gem5_backend.gem5_time_one (cross-compile + probe +
    fork-amortized forkloop; sum of absolute per-tc completion ticks)

Output: data_pie/gem5_ref_table.json
  {pid: {"src_gem5_ticks", "tgt_gem5_ticks", "k", "n_cases_total",
         "src_id", "tgt_id", "src_per_tc", "tgt_per_tc", "ds_speedup"}}

Usage: python build_gem5_ref_table.py [--pids p04028,p03688,...] [--k 5]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")

PIE_DATA = "/work/nvme/bgmo/sramasubramanian/pie_data"
TRAIN_JSONL = f"{PIE_DATA}/raw/train.jsonl"
# MUST match the PIE_TEST_CASE_DIR the training runs use — the per-tc arrays
# are indexed by this directory's canonical case order.
TEST_CASE_DIR = os.environ.get("PIE_TEST_CASE_DIR", f"{PIE_DATA}/tests")
DEFAULT_PIDS = ["p04028", "p03688", "p03556", "p03045", "p02882"]
OUT = "/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/data_pie/gem5_ref_table.json"

_SLOW_RE = re.compile(r"### Slow Version:\s*```(?:cpp)?\n(.*?)```", re.DOTALL)


def src_from_parquet(pid: str) -> str:
    import pandas as pd
    df = pd.read_parquet(f"{PIE_DATA}/parquet/overfit_{pid}.parquet")
    prompt = df.iloc[0]["prompt"]
    txt = prompt[-1]["content"]
    m = _SLOW_RE.search(txt)
    if not m:
        raise RuntimeError(f"{pid}: no '### Slow Version:' code fence in prompt")
    return m.group(1).strip()


def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def find_pair(pid: str, src_code: str) -> dict:
    """First train.jsonl pair for pid whose src matches the prompt's src."""
    want = _norm(src_code)
    matches = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r.get("problem_id") != pid:
                continue
            if _norm(r.get("src_code", "")) == want:
                matches.append(r)
    if not matches:
        raise RuntimeError(f"{pid}: prompt src not found in train.jsonl")
    if len(matches) > 1:
        print(f"  note: {pid} src matches {len(matches)} pairs; using first "
              f"(tgt_id={matches[0].get('tgt_id')})")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default=",".join(DEFAULT_PIDS))
    # default K from the same env the RL reward reads, so they cannot drift
    ap.add_argument("--k", type=int,
                    default=int(os.environ.get("PIE_MEASUREMENT_MAX_TESTS", "5")))
    args = ap.parse_args()
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]

    from experiment11_code_optim.reward.pie_reward_fn import (
        _get_test_cases, _subsample_cases,
    )
    from experiment11_code_optim.measurement.gem5_backend import (
        gem5_available, gem5_time_batch,
    )
    assert gem5_available(), "gem5 toolchain paths missing on this host"

    table: dict[str, dict] = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            table = json.load(f)
        print(f"extending existing table ({len(table)} entries)")

    tasks: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    for pid in pids:
        src_code = src_from_parquet(pid)
        pair = find_pair(pid, src_code)
        cases = _get_test_cases(TEST_CASE_DIR, pid)
        if not cases:
            raise RuntimeError(f"{pid}: no test cases in {TEST_CASE_DIR}")
        # Time ALL cases (canonical _get_test_cases order). The per-tc arrays
        # let the reward sum src/tgt time over ANY runtime subset — required
        # for PIE_TC_SAMPLE_MODE=random (a fresh subset each training step).
        inputs = [c[0] for c in cases]
        print(f"{pid}: timing ALL {len(cases)} cases; "
              f"src_id={pair.get('src_id')} tgt_id={pair.get('tgt_id')} "
              f"ds_speedup={pair.get('speedup')}")
        # CHUNKED: one sequential forkloop over ~104 cases costs the SUM of
        # per-tc sims (hours for compute-heavy programs); chunks run as
        # parallel gem5 tasks so wall = slowest chunk. Ticks are absolute and
        # deterministic, so chunking does not change any value.
        C = int(os.environ.get("PIE_REF_CHUNK", "8"))
        for role, code in (("src", pair["src_code"]), ("tgt", pair["tgt_code"])):
            for ci in range(0, len(inputs), C):
                tasks[f"{pid}|{role}|{ci}"] = {"code": code,
                                               "inputs": inputs[ci:ci + C]}
        meta[pid] = {
            "k": len(cases), "n_cases_total": len(cases),
            "src_id": pair.get("src_id"), "tgt_id": pair.get("tgt_id"),
            "ds_speedup": pair.get("speedup"),
        }

    print(f"timing {len(tasks)} program-chunks with gem5...")
    results = gem5_time_batch(tasks)

    for pid in pids:
        per_tc = {"src": [], "tgt": []}
        errs = []
        for role in ("src", "tgt"):
            keys = sorted((k for k in results if k.startswith(f"{pid}|{role}|")),
                          key=lambda k: int(k.rsplit("|", 1)[1]))
            for k in keys:
                r = results[k]
                if r["ticks"] is None:
                    errs.append(f"{k}: {r['error']}")
                else:
                    per_tc[role].extend(r["per_tc"])
        if errs or not per_tc["src"] or len(per_tc["src"]) != len(per_tc["tgt"]):
            print(f"  ERROR {pid}: {errs[:3]} — skipped")
            continue
        s_ticks, t_ticks = sum(per_tc["src"]), sum(per_tc["tgt"])
        table[pid] = {
            "src_gem5_ticks": s_ticks, "tgt_gem5_ticks": t_ticks,
            "src_per_tc_all": per_tc["src"], "tgt_per_tc_all": per_tc["tgt"],
            **meta[pid],
        }
        print(f"  {pid}: src={s_ticks/1e9:.3f}e9 tgt={t_ticks/1e9:.3f}e9 "
              f"gem5_speedup={s_ticks/t_ticks:.2f}x "
              f"(dataset says {meta[pid]['ds_speedup']})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as f:
        json.dump(table, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(OUT + ".tmp", OUT)
    print(f"wrote {OUT} ({len(table)} entries)")


if __name__ == "__main__":
    main()
