"""Per-case gem5 probe for heavy problems, implementing the PIE paper's own
test-case exclusion (paper §2: cases causing gem5 timeouts 'above 2 minutes'
are EXCLUDED from the suite). For each case of each problem we time src and
tgt individually under a per-case wall cap; cases where either side exceeds
the cap are reported for exclusion.

Writes a raw JSON: {pid: {case_idx: {"src": ticks|None, "tgt": ticks|None}}}
Usage: PIE_TEST_CASE_DIR=<merged> python probe_percase_gem5.py p04028,p03556
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")

PIE_DATA = "/work/nvme/bgmo/sramasubramanian/pie_data"
TRAIN_JSONL = f"{PIE_DATA}/raw/train.jsonl"
TEST_CASE_DIR = os.environ["PIE_TEST_CASE_DIR"]
OUT = "/work/nvme/bgtv/sramasubramanian/gem5_scale_out/percase_probe.json"
PAIRS = {"p04028": ("s046752289", "s958850010"),
         "p03556": ("s162970044", "s053033100")}

from experiment11_code_optim.precompute.build_gem5_ref_table import src_from_parquet, find_pair
from experiment11_code_optim.reward.pie_reward_fn import _get_test_cases
from experiment11_code_optim.measurement.gem5_backend import gem5_time_batch


def main() -> None:
    pids = sys.argv[1].split(",") if len(sys.argv) > 1 else list(PAIRS)
    tasks: dict[str, dict] = {}
    for pid in pids:
        pair = find_pair(pid, src_from_parquet(pid))
        cases = _get_test_cases(TEST_CASE_DIR, pid)
        for role, code in (("src", pair["src_code"]), ("tgt", pair["tgt_code"])):
            for i, (inp, _exp) in enumerate(cases):
                tasks[f"{pid}|{role}|{i}"] = {"code": code, "inputs": [inp]}
    print(f"{len(tasks)} single-case probes...", flush=True)
    res = gem5_time_batch(tasks)
    out: dict[str, dict] = {}
    for key, r in res.items():
        pid, role, i = key.split("|")
        out.setdefault(pid, {}).setdefault(i, {})[role] = (
            r["per_tc"][0] if r.get("per_tc") else None)
    with open(OUT + ".tmp", "w") as f:
        json.dump(out, f); f.flush(); os.fsync(f.fileno())
    os.replace(OUT + ".tmp", OUT)
    for pid in pids:
        d = out.get(pid, {})
        n = len(d)
        bad = [i for i, v in d.items() if v.get("src") is None or v.get("tgt") is None]
        print(f"{pid}: {n} cases, {len(bad)} exceed cap (would be excluded): "
              f"{sorted(bad, key=int)[:30]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
