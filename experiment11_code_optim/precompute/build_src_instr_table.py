"""Build {record_key: src_instr_count} table for the log2-speedup reward.

For every record in train/val/test, compile `src_code` at -O0 and grade it
against the K=5 deterministically-subsampled tests for `problem_id`. Store
the total retired-instruction count (perf backend) keyed by
`md5(src_code + "::" + problem_id)`.

This table is the denominator-source of the new reward: at training time
the parquet row carries `src_instr_count` inside ground_truth, the reward
fn reads it per-rollout, and computes max(0, log2(src_instr / cand_instr)).

Records where the src_code fails to compile or fails any subsampled test at
-O0 get `src_instr_count = None` and are dropped from the parquet by
build_pie_parquet.py (no usable denominator → no reward signal).

Run on a compute node (perf backend needs perf_event_paranoid=2 and the
SIGSTOP wrapper; both are satisfied on Delta-AI GHX4 nodes):

    export PIE_BACKEND=perf
    export PIE_MEASUREMENT_MAX_TESTS=5
    export PIE_COMPILE_SCRATCH="${TMPDIR:-/tmp}"
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/build_src_instr_table.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
RAW_ROOT = PIE_DATA_ROOT / "raw"
TESTS_ROOT = PIE_DATA_ROOT / "tests"
DEFAULT_OUTPUT = PIE_DATA_ROOT / "src_instr_table_O0.pkl"

SPLIT_JSONLS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def record_key(src_code: str, problem_id: str) -> str:
    """Stable per-record key. The reward fn reads `src_instr_count` from
    ground_truth (NOT this pickle) at training time, so this key is only
    used by build_pie_parquet.py to join pickle → record. Different (src_code,
    problem_id) tuples land in different keys; two records sharing src_code +
    problem_id correctly share the same instruction count."""
    return hashlib.md5(f"{src_code}::{problem_id}".encode()).hexdigest()


def load_records() -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
    """Return (dedup_tasks_for_grading, per_record_triples).

    `dedup_tasks` is keyed by record_key; one grade per unique
    (src_code, problem_id). `per_record_triples` is (split, pid, key) for
    each raw record so build_pie_parquet can join.
    """
    dedup_tasks: dict[str, dict] = {}
    triples: list[tuple[str, str, str]] = []
    for split, fn in SPLIT_JSONLS.items():
        path = RAW_ROOT / fn
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                pid = r.get("problem_id")
                src = r.get("src_code")
                if not pid or not src:
                    continue
                k = record_key(src, pid)
                triples.append((split, pid, k))
                if k not in dedup_tasks:
                    dedup_tasks[k] = {
                        "code": src,
                        "problem_id": pid,
                        "split": split,
                    }
    return dedup_tasks, triples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Pickle path: {record_key: int|None} mapping.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, take only the first N dedup keys (smoke).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build dedup tasks and report stats; do NOT grade or write.")
    args = ap.parse_args()

    # Force -O0 to match training-time compile flags. The reward fn assumes
    # the parquet's src_instr_count was recorded under the same flags as the
    # candidate is compiled with, otherwise the speedup ratio is meaningless.
    os.environ.setdefault("PIE_BACKEND", "perf")  # only perf gives reproducible counts
    if os.environ.get("PIE_BACKEND") != "perf":
        print(f"FATAL: PIE_BACKEND={os.environ.get('PIE_BACKEND')!r} — perf backend "
              f"is required (walltime under -O0 has 28× rel-spread; see "
              f"docs/DATA_ANALYSIS_2026_05_22_23.md)", file=sys.stderr)
        return 2

    # Late imports so --help works without paying setup cost.
    from experiment11_code_optim.measurement.core import setup_pch, COMPILE_FLAGS
    from experiment11_code_optim.reward.pie_reward_fn import (
        grade_dedup_batch,
        preload_test_cases,
    )
    assert "-O0" in COMPILE_FLAGS, (
        f"build_src_instr_table.py requires COMPILE_FLAGS=-O0 in core.py; "
        f"got {COMPILE_FLAGS!r}. The reward fn assumes src_instr_count was "
        f"recorded under the SAME compile flags that candidates use at "
        f"training time, otherwise the log2(src/cand) ratio is garbage."
    )

    print(f"[setup] PIE_BACKEND={os.environ['PIE_BACKEND']} COMPILE_FLAGS={COMPILE_FLAGS!r}")
    print(f"[setup] Building PCH...")
    pch_path = setup_pch()
    os.environ["PIE_PCH_INCLUDE_PATH"] = pch_path

    print(f"[setup] Preloading test cases from {TESTS_ROOT}...")
    t0 = time.monotonic()
    n_cases = preload_test_cases(str(TESTS_ROOT))
    print(f"  {n_cases} cases loaded in {time.monotonic() - t0:.1f}s")

    K_cap_env = os.environ.get("PIE_MEASUREMENT_MAX_TESTS")
    K_cap = int(K_cap_env) if K_cap_env else 5  # default 5 = match RL training
    print(f"[setup] K_cap = {K_cap} (from PIE_MEASUREMENT_MAX_TESTS or default 5)")

    print(f"[load] Reading raw records from {RAW_ROOT}...")
    dedup_tasks, triples = load_records()
    print(f"  {len(triples)} total records → {len(dedup_tasks)} unique (src_code, pid)")
    if args.limit > 0:
        keys = list(dedup_tasks)[: args.limit]
        dedup_tasks = {k: dedup_tasks[k] for k in keys}
        triples = [t for t in triples if t[2] in dedup_tasks]
        print(f"[limit] truncated to {len(dedup_tasks)} unique keys / {len(triples)} records")

    if args.dry_run:
        print("[dry-run] skipping grading + write")
        return 0

    # ---- Grade via the shared two-level scheduler ----
    print(f"[grade] running grade_dedup_batch on {len(dedup_tasks)} src programs...")
    t0 = time.monotonic()
    dedup_results = grade_dedup_batch(
        dedup_tasks,
        test_case_dir=str(TESTS_ROOT),
        measurement_max_tests=K_cap,
        # Override the default 30s per_test_timeout — src_code at -O0 can run
        # 10-50× slower than at -O3, and the prior 6/10s defaults were a 100%
        # timeout on the long-tail programs we still want a count for.
        timeout_per_test=30.0,
        # No global cap on the offline build; we want every grade to complete
        # regardless of one bad program slowing the batch.
        global_timeout=None,
    )
    elapsed = time.monotonic() - t0
    rate = len(dedup_tasks) / max(elapsed, 1e-6)
    print(f"[grade] {len(dedup_tasks)} grades in {elapsed:.1f}s ({rate:.1f} grades/s)")

    # ---- Build {record_key: instr_count_or_None} ----
    table: dict[str, int | None] = {}
    n_pass = 0
    n_fail_compile = 0
    n_fail_gate = 0
    per_split_drop = defaultdict(int)
    for key, task in dedup_tasks.items():
        r = dedup_results.get(key, {})
        if r.get("all_pass") and r.get("elapsed_ns") and r["elapsed_ns"] > 0:
            table[key] = int(r["elapsed_ns"])
            n_pass += 1
        else:
            table[key] = None
            if not r.get("compiled"):
                n_fail_compile += 1
            else:
                n_fail_gate += 1
            per_split_drop[task["split"]] += 1

    # Per-split summary (so build_pie_parquet can predict drops before joining)
    per_split_total = defaultdict(int)
    for split, _pid, key in triples:
        per_split_total[split] += 1
    print()
    print("=== SUMMARY ===")
    print(f"  unique (src_code, pid) graded:       {len(dedup_tasks)}")
    print(f"    pass (have instr_count):           {n_pass}")
    print(f"    compile-failed:                    {n_fail_compile}")
    print(f"    gate-failed:                       {n_fail_gate}")
    for split in SPLIT_JSONLS:
        total = per_split_total[split]
        drop = per_split_drop[split]
        print(f"  {split}: {total - drop}/{total} records keep src_instr_count "
              f"({drop} dropped)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(table, f)
    print(f"\n  output: {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
