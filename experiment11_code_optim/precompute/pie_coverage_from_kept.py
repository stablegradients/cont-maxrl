"""Final PIE-faithful timing-coverage table, computed from PRECOMPUTED artifacts
only (no recompilation):
  - kept-case set per problem  <- pie_intersection/<split>/shard_*.jsonl
      (case kept iff EVERY accepted, non-bad-apple submission passes it)
  - per-program timed cases     <- pie_gem5_times/per_program_<split>.parquet
      (per_tc_ticks JSON keys = the cases we have a gem5 time for)

For each official (src,tgt) pair classify by the KEPT cases both sides have timed:
  completely = both sides timed EVERY kept case
  partially  = some-but-not-all kept cases timed on both
  none       = zero kept cases timed on both
A case EXCLUDED by the native/budget screen is not "missing" -- if it is not in
the kept set it does not count against coverage (PIE excludes those same cases).

Usage: pie_coverage_from_kept.py            # all splits
"""
import glob
import json

import pandas as pd

OFF = "/work/nvme/bgtv/sramasubramanian/pie_official/PIE_Dataset"
INT = "/work/nvme/bgtv/sramasubramanian/pie_intersection"
TIMES = "/work/nvme/bgtv/sramasubramanian/pie_gem5_times"
SPLITS = ["train", "val", "test"]


def kept_sets(split):
    """{pid: set(kept_idx)} from intersection shards (last write wins)."""
    k = {}
    for f in sorted(glob.glob(f"{INT}/{split}/shard_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k[r["pid"]] = set(r["kept"])
    return k


def timed_sets(split):
    """{(pid,sid): set(timed_idx)} from per_program parquet per_tc_ticks keys."""
    pp = pd.read_parquet(f"{TIMES}/per_program_{split}.parquet")
    t = {}
    for r in pp.itertuples():
        per = json.loads(r.per_tc_ticks) if r.per_tc_ticks else {}
        t[(r.problem_id, r.submission_id)] = set(int(x) for x in per)
    return t


def main():
    print(f"{'split':<7}{'pairs':>8}  {'COMPLETE':>14} {'PARTIAL':>13} "
          f"{'NONE':>11} {'no-kept':>9}  {'med_kept':>9}")
    for sp in SPLITS:
        kept = kept_sets(sp)
        timed = timed_sets(sp)
        pairs = []
        for line in open(f"{OFF}/{sp}.jsonl"):
            x = json.loads(line)
            pairs.append((x["problem_id"], x.get("src_id"), x.get("tgt_id")))
        comp = part = none = nopair = 0
        for pid, s, t in pairs:
            k = kept.get(pid, set())
            if not s or not t or not k:
                nopair += 1
                continue
            both = timed.get((pid, s), set()) & timed.get((pid, t), set())
            cov = len(k & both)
            if cov == len(k):
                comp += 1
            elif cov == 0:
                none += 1
            else:
                part += 1
        N = len(pairs)
        meds = sorted(len(v) for v in kept.values())
        med = meds[len(meds) // 2] if meds else 0
        print(f"{sp:<7}{N:>8}  {comp:>7}({100*comp/N:5.1f}%) {part:>6}({100*part/N:4.1f}%) "
              f"{none:>5}({100*none/N:4.1f}%) {nopair:>4}({100*nopair/N:4.1f}%)  {med:>9}")


if __name__ == "__main__":
    main()
