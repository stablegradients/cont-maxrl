"""Per-problem tgt spread restricted to TRAIN split only."""
from __future__ import annotations
import hashlib, json, os, sys, time, statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_COMPILE_SCRATCH",
                       f"/tmp/{os.environ.get('USER', 'u')}_pie_scratch_train")
os.makedirs(os.environ["PIE_COMPILE_SCRATCH"], exist_ok=True)

REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)
import experiment11_code_optim.measurement.core as core
core.COMPILE_FLAGS = ["-O0", "-std=c++17"]
from experiment11_code_optim.measurement.core import setup_pch
os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
print("[setup] PCH built (-O0)", flush=True)

from experiment11_code_optim.reward.pie_reward_fn import (
    grade_dedup_batch, preload_test_cases,
)
TEST_CASE_DIR = "/work/nvme/bgmo/sramasubramanian/pie_data/tests"
OUT_DIR = Path(REPO) / "experiment11_code_optim/docs/plots"
preload_test_cases(TEST_CASE_DIR)

# Train tgts only
per_pid_tgts: dict[str, set[str]] = defaultdict(set)
with open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl") as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("problem_id"); tgt = r.get("tgt_code")
        if pid and tgt:
            per_pid_tgts[pid].add(tgt)
n_tgt = sum(len(v) for v in per_pid_tgts.values())
print(f"[setup] {len(per_pid_tgts)} pids, {n_tgt} unique train tgts", flush=True)

dedup_tasks = {}
task_meta = {}
for pid, codes in per_pid_tgts.items():
    for code in codes:
        key = hashlib.md5(f"{code}::{pid}::tgt_train".encode()).hexdigest()
        dedup_tasks[key] = {"code": code, "problem_id": pid, "split": "val"}
        task_meta[key] = pid

t0 = time.monotonic()
results = grade_dedup_batch(
    dedup_tasks, test_case_dir=TEST_CASE_DIR,
    measurement_max_tests=5, timeout_per_test=30.0, num_workers=64,
)
print(f"[grade] done in {time.monotonic() - t0:.1f}s", flush=True)

per_pid_el: dict[str, list[int]] = defaultdict(list)
for k, r in results.items():
    pid = task_meta[k]
    if r.get("all_pass") and r.get("elapsed_ns") and r["elapsed_ns"] > 0:
        per_pid_el[pid].append(int(r["elapsed_ns"]))

with open(OUT_DIR / "pie_train_tgt_only_elapsed_ns_O0.json", "w") as f:
    json.dump({pid: lst for pid, lst in per_pid_el.items()}, f)
print(f"[save] elapsed lists for {len(per_pid_el)} pids", flush=True)

mom = []
for pid, lst in per_pid_el.items():
    if len(lst) >= 2:
        mom.append(max(lst) / min(lst))
mom.sort()
n = len(mom)
print(f"\nper-pid tgt max/min (TRAIN-only, -O0, n={n} pids with ≥2 tgts):")
for p in (10, 25, 50, 75, 90, 95, 99):
    print(f"  p{p:>2d}: {mom[int(p/100*(n-1))]:.3f}x")
print(f"  max: {mom[-1]:.3f}x")
print(f"  mean: {statistics.mean(mom):.2f}x   median: {mom[n//2]:.4f}x")
print()
for thr in (1.5, 2, 3, 5, 10, 50, 100):
    cnt = sum(1 for r in mom if r > thr)
    print(f"  max/min > {thr:>5}x: {cnt:>4d} / {n}  ({100*cnt/n:.1f}%)")
