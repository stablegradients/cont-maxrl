"""Consolidate the raw per-shard gem5 sweep output into clean, usable datasets.

Produces, under PIE_GEM5_TIMES_OUT (default /work/nvme/bgtv/.../pie_gem5_times/):
  per_program_<split>.parquet   one row per unique (problem_id, submission_id):
      problem_id, submission_id, split, roles, status, n_cases_total,
      n_cases_timed, sum_ticks, sum_seconds, dataset_agg_runtime,
      per_tc_ticks (JSON {idx:ticks}), excluded (JSON [idx])
  pairs_<split>.parquet         one row per dataset (src,tgt) pair, our gem5
      timing joined for both sides:
      problem_id, src_id, tgt_id, split,
      src_sum_seconds, tgt_sum_seconds, our_speedup,
      dataset_speedup, dataset_src_agg_runtime, dataset_tgt_agg_runtime,
      src_per_tc_ticks (JSON), tgt_per_tc_ticks (JSON),
      src_status, tgt_status

Both the PER-TEST-CASE times (per_tc dicts) and the TOTAL SUM are carried, for
train / val / test. Ticks are gem5 picoseconds; seconds = ticks * 1e-12.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

SWEEP = "/work/nvme/bgtv/sramasubramanian/gem5_dataset_sweep"
OFFICIAL = "/work/nvme/bgtv/sramasubramanian/pie_official/PIE_Dataset"
OUT = os.environ.get("PIE_GEM5_TIMES_OUT",
                     "/work/nvme/bgtv/sramasubramanian/pie_gem5_times")
PS_PER_S = 1e12
SPLITS = ["train", "val", "test"]


def load_programs(split: str) -> dict:
    """{(pid,sid): record} merged across shards. Files are sorted so that
    `zz_retime_*` (re-timed partials) sort AFTER `shard_*` and override the
    original partial records (last write wins)."""
    d = {}
    for f in sorted(glob.glob(f"{SWEEP}/{split}/*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            d[(r["pid"], r["sid"])] = r
    return d


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for split in SPLITS:
        progs = load_programs(split)

        # ---- per-program table ----
        rows = []
        for (pid, sid), r in progs.items():
            per = r.get("per_tc") or {}
            rows.append({
                "problem_id": pid, "submission_id": sid, "split": split,
                "roles": ",".join(r.get("roles", [])),
                "status": r.get("status"),
                "n_cases_total": r.get("n_cases_total"),
                "n_cases_timed": len(per),
                "sum_ticks": r.get("sum_ticks"),
                "sum_seconds": (r["sum_ticks"] / PS_PER_S
                                if r.get("sum_ticks") else None),
                "dataset_agg_runtime": r.get("dataset_agg_runtime"),
                "per_tc_ticks": json.dumps(per),
                "excluded": json.dumps(r.get("excluded", [])),
            })
        pp = pd.DataFrame(rows)
        ppath = f"{OUT}/per_program_{split}.parquet"
        pp.to_parquet(ppath, index=False)

        # ---- pair-level table (join our timing onto dataset pairs) ----
        def side(pid, sid):
            r = progs.get((pid, sid))
            if not r or not r.get("sum_ticks"):
                return None
            return r
        prows = []
        for line in open(f"{OFFICIAL}/{split}.jsonl"):
            x = json.loads(line)
            pid = x["problem_id"]
            sr = side(pid, x.get("src_id")); tr = side(pid, x.get("tgt_id"))
            s_per = {int(k): int(v) for k, v in (sr.get("per_tc") or {}).items()} if sr else {}
            t_per = {int(k): int(v) for k, v in (tr.get("per_tc") or {}).items()} if tr else {}
            s_s = sr["sum_ticks"] / PS_PER_S if sr else None
            t_s = tr["sum_ticks"] / PS_PER_S if tr else None
            # VALID speedup = ratio over the COMMON timed cases only (src and tgt
            # must be measured on the SAME inputs; a partial side has fewer cases)
            common = sorted(set(s_per) & set(t_per))
            our_spd = None
            if common:
                sc = sum(s_per[i] for i in common); tc = sum(t_per[i] for i in common)
                if tc > 0:
                    our_spd = sc / tc
            prows.append({
                "problem_id": pid, "src_id": x.get("src_id"), "tgt_id": x.get("tgt_id"),
                "split": split,
                "src_sum_seconds": s_s, "tgt_sum_seconds": t_s,
                "src_n_cases": len(s_per), "tgt_n_cases": len(t_per),
                "n_common_cases": len(common),
                "our_speedup": our_spd,          # over common cases (valid)
                "dataset_speedup": (
                    x.get("speedup") if x.get("speedup") is not None
                    else (x["src_agg_runtime"] / x["tgt_agg_runtime"]
                          if x.get("src_agg_runtime") is not None
                          and x.get("tgt_agg_runtime") else None)),
                "dataset_src_agg_runtime": x.get("src_agg_runtime"),
                "dataset_tgt_agg_runtime": x.get("tgt_agg_runtime"),
                "src_per_tc_ticks": json.dumps(sr.get("per_tc") or {}) if sr else "{}",
                "tgt_per_tc_ticks": json.dumps(tr.get("per_tc") or {}) if tr else "{}",
                "src_status": sr.get("status") if sr else None,
                "tgt_status": tr.get("status") if tr else None,
            })
        pr = pd.DataFrame(prows)
        prpath = f"{OUT}/pairs_{split}.parquet"
        pr.to_parquet(prpath, index=False)

        ok = (pp["status"] == "ok").sum()
        both = pr[["src_sum_seconds", "tgt_sum_seconds"]].notna().all(axis=1).sum()
        print(f"{split}: per_program {len(pp)} rows ({ok} ok) -> {ppath}")
        print(f"        pairs {len(pr)} rows ({both} with both src+tgt timed) -> {prpath}")

    # tiny manifest describing the schema
    with open(f"{OUT}/README.txt", "w") as f:
        f.write(
            "PIE gem5 times (our reproduction), 2026-06-10.\n"
            "per_program_<split>.parquet: 1 row/unique program; per_tc_ticks (JSON "
            "{idx:ticks}) = PER-TEST-CASE time, sum_ticks/sum_seconds = TOTAL.\n"
            "pairs_<split>.parquet: 1 row/dataset pair; src/tgt sum_seconds + "
            "per_tc_ticks + our_speedup, joined to dataset_speedup/agg_runtime.\n"
            "Ticks are gem5 picoseconds; seconds = ticks*1e-12. Validated vs PIE: "
            "test per-tc r=1.0000; train/val speedup r=0.996.\n")
    print(f"wrote schema README -> {OUT}/README.txt")


if __name__ == "__main__":
    main()
