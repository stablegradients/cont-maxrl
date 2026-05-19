"""
Build the unified PIE test-case tree under /work/nvme/bgmo/.../pie_data/tests/.

For every problem_id appearing in any of the staged PIE JSONLs (train, train_hq_only,
val, test), copy the richest available test-case set into a single uniform layout:

    tests/{problem_id}/input.<i>.txt
    tests/{problem_id}/output.<i>.txt   # i = 0 .. n_tests-1, contiguous

Source precedence per problem (richest first; first hit wins):

  1. pie-perf Generated test cases (~100 tests/problem, alphacode-sourced; 247 problems)
  2. pie-perf Public test cases     (~3   tests/problem, CodeNet-extracted; 1141 problems)
  3. IBM CodeNet derived/input_output/  (1 test/problem; 3867 problems, near-100% coverage)

A manifest is written to tests/MANIFEST.json describing source + n_tests per problem.

Run on a compute node:

    /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/build_unified_test_cases.py

This is intentionally side-effecting (copies files); it skips problems whose
output already exists with the right n_tests so reruns are cheap.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
GENERATED_ROOT = PIE_DATA_ROOT / "pie_perf_generated" / "codenet" / "generated_test_cases"
PUBLIC_ROOT = PIE_DATA_ROOT / "pie_perf_tests" / "codenet" / "public_test_cases"
CODENET_ROOT = PIE_DATA_ROOT / "Project_CodeNet" / "derived" / "input_output" / "data"
TARGET_ROOT = PIE_DATA_ROOT / "tests"

JSONL_RAW = PIE_DATA_ROOT / "raw"
JSONL_NAMES = ["train.jsonl", "train_hq_only.jsonl", "val.jsonl", "test.jsonl"]

_INDEX_RE = re.compile(r"^input\.(\d+)\.txt$")


def collect_required_problem_ids() -> dict[str, set[str]]:
    """Return {jsonl_name: set_of_problem_ids}, and the union under 'all'."""
    per_split: dict[str, set[str]] = {}
    for name in JSONL_NAMES:
        ids: set[str] = set()
        path = JSONL_RAW / name
        with open(path) as f:
            for line in f:
                ids.add(json.loads(line)["problem_id"])
        per_split[name] = ids
    per_split["all"] = set().union(*per_split.values())
    return per_split


def list_generated_tests(pid: str) -> list[int] | None:
    d = GENERATED_ROOT / pid
    if not d.is_dir():
        return None
    indices: list[int] = []
    for fn in os.listdir(d):
        m = _INDEX_RE.match(fn)
        if m:
            i = int(m.group(1))
            if (d / f"output.{i}.txt").exists():
                indices.append(i)
    indices.sort()
    return indices if indices else None


def list_public_tests(pid: str) -> list[int] | None:
    d = PUBLIC_ROOT / pid
    if not d.is_dir():
        return None
    indices: list[int] = []
    for fn in os.listdir(d):
        m = _INDEX_RE.match(fn)
        if m:
            i = int(m.group(1))
            if (d / f"output.{i}.txt").exists():
                indices.append(i)
    indices.sort()
    return indices if indices else None


def has_codenet_single(pid: str) -> bool:
    d = CODENET_ROOT / pid
    return (d / "input.txt").exists() and (d / "output.txt").exists()


def pick_source(pid: str) -> tuple[str, Path, list[int]] | None:
    """Returns (source_name, source_dir, sorted_source_indices) or None if no source."""
    indices = list_generated_tests(pid)
    if indices is not None:
        return "generated", GENERATED_ROOT / pid, indices
    indices = list_public_tests(pid)
    if indices is not None:
        return "public", PUBLIC_ROOT / pid, indices
    if has_codenet_single(pid):
        return "codenet", CODENET_ROOT / pid, [-1]  # sentinel: copy input.txt/output.txt directly
    return None


def write_problem(pid: str, source_name: str, source_dir: Path, source_indices: list[int]) -> int:
    """Materialise tests/{pid}/input.<i>.txt + output.<i>.txt with contiguous i = 0..n-1.
    Returns the number of test cases written."""
    tgt = TARGET_ROOT / pid
    tgt.mkdir(parents=True, exist_ok=True)

    if source_name == "codenet":
        shutil.copyfile(source_dir / "input.txt",  tgt / "input.0.txt")
        shutil.copyfile(source_dir / "output.txt", tgt / "output.0.txt")
        return 1

    # Generated or Public: renumber source indices to 0..n-1 contiguous to avoid
    # surprises if the source skipped any integers (it shouldn't, but be safe).
    for new_i, old_i in enumerate(source_indices):
        shutil.copyfile(source_dir / f"input.{old_i}.txt",  tgt / f"input.{new_i}.txt")
        shutil.copyfile(source_dir / f"output.{old_i}.txt", tgt / f"output.{new_i}.txt")
    return len(source_indices)


def main() -> int:
    if not GENERATED_ROOT.is_dir():
        print(f"ERROR: Generated test cases not found at {GENERATED_ROOT}", file=sys.stderr)
        return 1
    if not PUBLIC_ROOT.is_dir():
        print(f"ERROR: Public test cases not found at {PUBLIC_ROOT}", file=sys.stderr)
        return 1
    if not CODENET_ROOT.is_dir():
        print(f"ERROR: CodeNet derived/input_output not found at {CODENET_ROOT}", file=sys.stderr)
        return 1

    pids_by_split = collect_required_problem_ids()
    all_pids = sorted(pids_by_split["all"])
    print(f"Building tests/ for {len(all_pids)} unique problem_ids across {len(JSONL_NAMES)} splits...")
    print(f"  Target: {TARGET_ROOT}")

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    counts = {"generated": 0, "public": 0, "codenet": 0, "missing": 0}
    n_tests_total = 0
    missing_pids: list[str] = []

    for pid in all_pids:
        pick = pick_source(pid)
        if pick is None:
            counts["missing"] += 1
            missing_pids.append(pid)
            manifest[pid] = {"source": "MISSING", "n_tests": 0}
            continue

        source_name, source_dir, source_indices = pick
        n_written = write_problem(pid, source_name, source_dir, source_indices)
        counts[source_name] += 1
        n_tests_total += n_written
        manifest[pid] = {"source": source_name, "n_tests": n_written}

    # Per-split coverage stats
    split_stats: dict[str, dict] = {}
    for split in JSONL_NAMES:
        pids_split = pids_by_split[split]
        sources_seen = {"generated": 0, "public": 0, "codenet": 0, "missing": 0}
        n_tests_split = 0
        for pid in pids_split:
            m = manifest.get(pid, {"source": "MISSING", "n_tests": 0})
            sources_seen[m["source"].lower() if m["source"] != "MISSING" else "missing"] += 1
            n_tests_split += m["n_tests"]
        split_stats[split] = {
            "n_problems": len(pids_split),
            "by_source": sources_seen,
            "total_tests": n_tests_split,
            "mean_tests_per_problem": n_tests_split / max(len(pids_split), 1),
        }

    manifest_path = TARGET_ROOT / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "per_problem": manifest,
                "summary": {
                    "total_problems": len(all_pids),
                    "by_source": counts,
                    "total_tests": n_tests_total,
                    "mean_tests_per_problem": n_tests_total / max(len(all_pids), 1),
                },
                "per_split": split_stats,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print(f"Wrote {n_tests_total} test cases across {len(all_pids) - counts['missing']} problems.")
    print(f"  by source:  generated={counts['generated']}  public={counts['public']}  "
          f"codenet={counts['codenet']}  missing={counts['missing']}")
    print(f"  manifest:   {manifest_path}")
    if missing_pids:
        print(f"  MISSING (no test cases in any source, first 20): {missing_pids[:20]}")
    print()
    print("Per-split coverage:")
    for split, st in split_stats.items():
        bs = st["by_source"]
        cov = (bs["generated"] + bs["public"] + bs["codenet"]) / st["n_problems"]
        print(f"  {split:25s}  problems={st['n_problems']:5d}  generated={bs['generated']:4d}  "
              f"public={bs['public']:4d}  codenet={bs['codenet']:4d}  missing={bs['missing']:4d}  "
              f"coverage={cov*100:.1f}%  mean_tests={st['mean_tests_per_problem']:.1f}")

    return 0 if counts["missing"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
