"""
§14 step 6: build per-problem reference tables for the percentile-rank reward.

For each problem in train/val/test, compile every distinct human-Accepted
source program (tgt_code from the PIE JSONLs) and grade it against the
unified test-case archive. On all-pass, record the elapsed_ns cost.
Sort per problem and write to:

    reference_table_dir/<split>/<problem_id>.json

Per-problem JSON schema (§14 step 6 contract, F1 closed):

    {
        "problem_id": "p02587",
        "split": "val",
        "n_programs_attempted": 12,
        "n_programs_passed": 11,
        "n_programs_compile_failed": 0,
        "n_programs_gate_failed": 1,
        "elapsed_ns_values": [12345, 13456, ..., 25600]   # sorted ascending
    }

Also writes reference_table_dir/MANIFEST.json with coverage stats. Problems
where 0 Accepted programs pass the strict gate are recorded as excluded;
at runtime `_percentile_from_reference_table` falls back to stub 0.5 for
those (open() raises, the except clause returns 0.5).

Run on a compute node with the same env exported by the PIE SLURM scripts:

    export PIE_MEASUREMENT_MAX_TESTS=5
    export PIE_COMPILE_SCRATCH="${TMPDIR:-/tmp}"
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/build_reference_table.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
RAW_ROOT = PIE_DATA_ROOT / "raw"
TESTS_ROOT = PIE_DATA_ROOT / "tests"
DEFAULT_OUTPUT_ROOT = PIE_DATA_ROOT / "reference_table"

SPLIT_JSONLS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def load_distinct_programs(split: str) -> dict[str, set[str]]:
    """{problem_id: {tgt_code, ...}} for one split, dedup-by-string."""
    path = RAW_ROOT / SPLIT_JSONLS[split]
    by_pid: dict[str, set[str]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("problem_id")
            code = r.get("tgt_code")
            if not pid or not code:
                continue
            by_pid.setdefault(pid, set()).add(code)
    return by_pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--splits", default="train,val,test",
                    help="Comma-separated splits to process.")
    ap.add_argument("--limit-problems", type=int, default=0,
                    help="If >0, take only the first N problems per split (smoke testing).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build dedup_tasks and report stats; do NOT grade or write.")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        if s not in SPLIT_JSONLS:
            print(f"FATAL: unknown split {s!r}; choices: {list(SPLIT_JSONLS)}",
                  file=sys.stderr)
            return 2

    # Late imports so --help works without paying setup cost.
    from experiment11_code_optim.measurement.core import setup_pch
    from experiment11_code_optim.reward.pie_reward_fn import (
        grade_dedup_batch,
        preload_test_cases,
    )

    print(f"[setup] Building PCH...")
    pch_path = setup_pch()
    os.environ["PIE_PCH_INCLUDE_PATH"] = pch_path
    print(f"  PCH at {pch_path}")

    print(f"[setup] Preloading test cases from {TESTS_ROOT}...")
    t0 = time.monotonic()
    n_cases = preload_test_cases(str(TESTS_ROOT))
    print(f"  {n_cases} cases loaded in {time.monotonic() - t0:.1f}s")

    K_cap_env = os.environ.get("PIE_MEASUREMENT_MAX_TESTS")
    K_cap = int(K_cap_env) if K_cap_env else None
    print(f"[setup] K_cap = {K_cap} (from PIE_MEASUREMENT_MAX_TESTS env)")

    # ---- Read distinct (pid, code) per split ----
    print(f"[load] Reading distinct tgt_code per problem from splits={splits}...")
    per_split: dict[str, dict[str, set[str]]] = {}
    n_total_pairs = 0
    for split in splits:
        by_pid = load_distinct_programs(split)
        if args.limit_problems > 0:
            sample = list(by_pid.keys())[: args.limit_problems]
            by_pid = {p: by_pid[p] for p in sample}
        per_split[split] = by_pid
        n_pairs = sum(len(v) for v in by_pid.values())
        n_total_pairs += n_pairs
        print(f"  {split}: {len(by_pid)} problems, {n_pairs} distinct (pid, code)")
    print(f"[load] Total: {n_total_pairs} distinct programs to grade")

    # ---- Build dedup_tasks ----
    # Same key scheme as compute_scores_batch: md5(code + "::" + pid).
    # Cross-split (pid, code) collisions: rare but possible. First-write-wins
    # for dedup_tasks (so we only grade once); however we record the per-(split,
    # pid, code) attempt against ALL splits where the triple appears.
    dedup_tasks: dict[str, dict] = {}
    triples: list[tuple[str, str, str, str]] = []  # (split, pid, code, key)
    for split, by_pid in per_split.items():
        for pid, codes in by_pid.items():
            for code in codes:
                key = hashlib.md5(f"{code}::{pid}".encode()).hexdigest()
                if key not in dedup_tasks:
                    dedup_tasks[key] = {"code": code, "problem_id": pid, "split": split}
                triples.append((split, pid, code, key))

    print(f"[dedup] {n_total_pairs} program×split pairs → {len(dedup_tasks)} unique grades "
          f"(ratio {n_total_pairs/max(1,len(dedup_tasks)):.2f}x)")

    if args.dry_run:
        print(f"[dry-run] Skipping grading + writing.")
        return 0

    # ---- Grade via the shared two-level scheduler ----
    print(f"[grade] Running grade_dedup_batch on {len(dedup_tasks)} unique tasks...")
    t0 = time.monotonic()
    dedup_results = grade_dedup_batch(
        dedup_tasks,
        test_case_dir=str(TESTS_ROOT),
        measurement_max_tests=K_cap,
    )
    elapsed = time.monotonic() - t0
    rate = len(dedup_tasks) / max(elapsed, 1e-6)
    print(f"[grade] {len(dedup_tasks)} grades in {elapsed:.1f}s = {rate:.1f} grades/s")

    # ---- Aggregate per (split, problem_id) ----
    # Different splits may share a (pid, code) but the bucket is per (split, pid).
    per_problem: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "attempted": 0,
        "passed": 0,
        "compile_failed": 0,
        "gate_failed": 0,
        "elapsed_ns_values": [],
    })
    for (split, pid, _code, key) in triples:
        r = dedup_results.get(key)
        if r is None:
            continue
        bucket = per_problem[(split, pid)]
        bucket["attempted"] += 1
        if r["all_pass"]:
            bucket["passed"] += 1
            bucket["elapsed_ns_values"].append(r["elapsed_ns"])
        elif not r["compiled"]:
            bucket["compile_failed"] += 1
        else:
            bucket["gate_failed"] += 1

    # ---- Write per-problem files + MANIFEST.json ----
    print(f"[write] Writing per-problem reference tables to {output_root}...")
    output_root.mkdir(parents=True, exist_ok=True)
    n_passing_problems = 0
    excluded: list[str] = []
    for (split, pid), bucket in per_problem.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        bucket["elapsed_ns_values"].sort()
        out = {
            "problem_id": pid,
            "split": split,
            "n_programs_attempted": bucket["attempted"],
            "n_programs_passed": bucket["passed"],
            "n_programs_compile_failed": bucket["compile_failed"],
            "n_programs_gate_failed": bucket["gate_failed"],
            "elapsed_ns_values": bucket["elapsed_ns_values"],
        }
        with open(split_dir / f"{pid}.json", "w") as f:
            json.dump(out, f)
        if bucket["passed"] > 0:
            n_passing_problems += 1
        else:
            excluded.append(f"{split}/{pid}")

    manifest = {
        "schema_version": 1,
        "splits": splits,
        "K_cap": K_cap,
        "n_total_problems": len(per_problem),
        "n_problems_with_at_least_one_passing": n_passing_problems,
        "n_problems_excluded": len(excluded),
        "n_unique_grades_run": len(dedup_tasks),
        "n_total_program_attempts": n_total_pairs,
        "grade_seconds": round(elapsed, 2),
        "grade_throughput_per_s": round(rate, 1),
        "excluded": sorted(excluded),
    }
    with open(output_root / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Serialize the whole table to one pickle so the reward fn can load
    # all 1592 reference distributions via ONE open()+read() instead of
    # opening a JSON file per (split, problem_id) at every reward call.
    # Same Lustre-storm avoidance as tests.pkl.
    import pickle as _pickle
    pkl_path = output_root.with_suffix(".pkl")
    refs: dict[tuple[str, str], list[int]] = {}
    for (split, pid), bucket in per_problem.items():
        refs[(split, pid)] = sorted(bucket["elapsed_ns_values"])
    with open(pkl_path, "wb") as f:
        _pickle.dump(refs, f)
    print(f"  pickle:                   {pkl_path} ({pkl_path.stat().st_size/1e6:.2f} MB)")

    # ---- Summary ----
    print()
    print(f"=== SUMMARY ===")
    print(f"  Total problems:           {manifest['n_total_problems']}")
    print(f"  ≥1 passing reference:     {manifest['n_problems_with_at_least_one_passing']}")
    print(f"  Excluded (0 references):  {manifest['n_problems_excluded']}")
    print(f"  Unique grades run:        {manifest['n_unique_grades_run']}")
    print(f"  Total grade time:         {elapsed:.1f}s")
    print(f"  Throughput:               {rate:.1f} grades/s")
    print(f"  Output:                   {output_root}")
    print(f"  MANIFEST:                 {output_root / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
