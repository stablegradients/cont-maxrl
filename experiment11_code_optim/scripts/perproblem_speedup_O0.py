"""Per-problem speedup histogram at -O0 compile, two views:

  (A) src+tgt combined: candidate pool per pid = unique (pid, code) across
      all src and tgt fields in train/val/test
  (B) tgt-only:          candidate pool per pid = unique tgt_code only

For each pid: slowest_g = max(elapsed_ns_in_pool), ratio_i = slowest_g/elapsed_i,
histogram on linear bins [1.0, 1.1, ..., 4.9, 5.0, +inf), normalize per
problem (sum=1), then average across problems.

Runs grading once at -O0 with a single dedup_tasks pool tagged with kind
(src/tgt) so both cuts are computed from the SAME measurements.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_COMPILE_SCRATCH",
                       f"/tmp/{os.environ.get('USER', 'u')}_pie_scratch_O0_v2")
os.makedirs(os.environ["PIE_COMPILE_SCRATCH"], exist_ok=True)

REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)

# Patch COMPILE_FLAGS BEFORE setup_pch
import experiment11_code_optim.measurement.core as core
core.COMPILE_FLAGS = ["-O0", "-std=c++17"]
print(f"[setup] COMPILE_FLAGS={core.COMPILE_FLAGS}", flush=True)

from experiment11_code_optim.measurement.core import setup_pch
os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
print("[setup] PCH built (-O0)", flush=True)

from experiment11_code_optim.reward.pie_reward_fn import (
    grade_dedup_batch, preload_test_cases,
)
TEST_CASE_DIR = "/work/nvme/bgmo/sramasubramanian/pie_data/tests"
OUT_DIR = Path(REPO) / "experiment11_code_optim/docs/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
preload_test_cases(TEST_CASE_DIR)
print("[setup] test cases preloaded", flush=True)

# Build per-pid (src | tgt) unique program set
per_pid_codes: dict[str, dict[str, set[str]]] = defaultdict(
    lambda: {"src": set(), "tgt": set()}
)
for split in ("train", "val", "test"):
    p = f"/work/nvme/bgmo/sramasubramanian/pie_data/raw/{split}.jsonl"
    if not os.path.isfile(p):
        continue
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("problem_id")
            if not pid:
                continue
            if r.get("src_code"):
                per_pid_codes[pid]["src"].add(r["src_code"])
            if r.get("tgt_code"):
                per_pid_codes[pid]["tgt"].add(r["tgt_code"])

n_pids = len(per_pid_codes)
n_src = sum(len(v["src"]) for v in per_pid_codes.values())
n_tgt = sum(len(v["tgt"]) for v in per_pid_codes.values())
print(f"[setup] {n_pids} pids, {n_src} unique src, {n_tgt} unique tgt", flush=True)

# Dedup_tasks keyed by hash(code+pid+kind) — kind in key to avoid src/tgt
# collision (which is rare but possible).
dedup_tasks: dict[str, dict] = {}
task_meta: dict[str, tuple[str, str]] = {}  # key -> (pid, kind)
for pid, byk in per_pid_codes.items():
    for kind in ("src", "tgt"):
        for code in byk[kind]:
            key = hashlib.md5(f"{code}::{pid}::{kind}".encode()).hexdigest()
            dedup_tasks[key] = {"code": code, "problem_id": pid, "split": "val"}
            task_meta[key] = (pid, kind)
print(f"[grade] {len(dedup_tasks)} unique kind-tagged programs", flush=True)

t0 = time.monotonic()
results = grade_dedup_batch(
    dedup_tasks,
    test_case_dir=TEST_CASE_DIR,
    measurement_max_tests=5,
    timeout_per_test=30.0,
    num_workers=64,
)
wall = time.monotonic() - t0
print(f"[grade] done in {wall:.1f}s", flush=True)

# Build per-pid {kind: list[elapsed]}
per_pid_elapsed: dict[str, dict[str, list[int]]] = defaultdict(
    lambda: {"src": [], "tgt": []}
)
n_gated = 0
for key, r in results.items():
    pid, kind = task_meta[key]
    if not r.get("all_pass") or r.get("elapsed_ns") is None or r["elapsed_ns"] <= 0:
        n_gated += 1
        continue
    per_pid_elapsed[pid][kind].append(int(r["elapsed_ns"]))

total_kept = sum(len(v["src"]) + len(v["tgt"]) for v in per_pid_elapsed.values())
print(f"[collect] kept={total_kept} gate-fail={n_gated} pids={len(per_pid_elapsed)}",
      flush=True)

raw_save = OUT_DIR / "pie_all_perproblem_elapsed_ns_O0.json"
with open(raw_save, "w") as f:
    json.dump({pid: {"src": v["src"], "tgt": v["tgt"]}
               for pid, v in per_pid_elapsed.items()}, f)
print(f"[save] {raw_save}", flush=True)

import numpy as np

edges = np.concatenate(([1.0], np.arange(1.1, 5.0001, 0.1), [float("inf")]))
n_bins = len(edges) - 1


def avg_hist(per_pid_lists):
    """per_pid_lists: dict[pid -> list[elapsed_ns]]"""
    acc = np.zeros(n_bins)
    n_p = 0
    pooled = []
    for pid, elapsed in per_pid_lists.items():
        if len(elapsed) < 1:
            continue
        g = max(elapsed)
        ratios = [g / e for e in elapsed]
        pooled.extend(ratios)
        h, _ = np.histogram(ratios, bins=edges)
        s = h.sum()
        if s == 0:
            continue
        acc += h / s
        n_p += 1
    return acc / n_p if n_p else acc, n_p, pooled


pid_combined = {pid: v["src"] + v["tgt"] for pid, v in per_pid_elapsed.items()}
pid_tgt_only = {pid: v["tgt"] for pid, v in per_pid_elapsed.items()}

avg_combined, n_combined, pool_combined = avg_hist(pid_combined)
avg_tgt, n_tgt_p, pool_tgt = avg_hist(pid_tgt_only)
print(f"[avg] combined avg over {n_combined} pids, tgt-only over {n_tgt_p}",
      flush=True)

with open(OUT_DIR / "pie_perproblem_O0_data.json", "w") as f:
    json.dump({
        "edges_finite": [1.0 + 0.1 * i for i in range(n_bins)],
        "avg_combined": avg_combined.tolist(),
        "avg_tgt": avg_tgt.tolist(),
        "n_combined": int(n_combined),
        "n_tgt": int(n_tgt_p),
    }, f)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
bar_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins - 1)]

for ax, title, avg, n_p in (
    (axes[0], f"src + tgt combined  (n={n_combined} pids)", avg_combined, n_combined),
    (axes[1], f"tgt only  (n={n_tgt_p} pids)", avg_tgt, n_tgt_p),
):
    ax.bar(bar_centers, avg[:-1], width=0.09, align="center",
           color="#4c72b0", edgecolor="black", linewidth=0.3, alpha=0.9)
    ax.bar([5.1], [avg[-1]], width=0.18, align="center",
           color="#c44e52", edgecolor="black", linewidth=0.4, alpha=0.9,
           label=f"[5,+inf): {avg[-1]:.3f}")
    ax.set_xticks(np.arange(1.0, 5.01, 0.5).tolist() + [5.1])
    ax.set_xticklabels([f"{x:.1f}" for x in np.arange(1.0, 5.01, 0.5)] + [">5"])
    ax.set_xlim(0.95, 5.25)
    ax.set_xlabel("slowest_g / elapsed   (1.0 = slowest in pid's pool)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
axes[0].set_ylabel("avg per-problem density (sum=1 normalized per pid)")

fig.suptitle("PIE per-problem speedup histogram, averaged across problems  —  "
             "-O0 compile (no compiler optimization)",
             fontsize=13)
plt.tight_layout()
out = OUT_DIR / "pie_perproblem_O0.png"
plt.savefig(out, dpi=120)
print(f"[plot] {out}", flush=True)

# Stats
for label, pool, n_p in [("combined", pool_combined, n_combined),
                          ("tgt-only", pool_tgt, n_tgt_p)]:
    if pool:
        s = sorted(pool)
        print(f"\n{label} pooled quantiles ({n_p} pids, {len(pool)} grades):")
        for p in (10, 25, 50, 75, 90, 95, 99):
            print(f"  p{p:>2d}: {s[int(p/100*(len(s)-1))]:.3f}")
        print(f"  max: {s[-1]:.3f}")
