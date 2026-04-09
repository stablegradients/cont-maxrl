"""Filter manifest for staged launch (Phase 1 gate / Phase 2 RL).

Usage:
    # Phase 1: supervised gate
    python -m experiment7_depth.filter_jobs \
        --manifest experiment7_depth/scripts/runs/manifest.csv \
        --methods ordinal_ce plain_ce \
        --output_dir experiment7_depth/scripts/runs_gate

    # Phase 2: everything except supervised gate
    python -m experiment7_depth.filter_jobs \
        --manifest experiment7_depth/scripts/runs/manifest.csv \
        --exclude-methods ordinal_ce plain_ce \
        --output_dir experiment7_depth/scripts/runs_phase2
"""

import argparse
import csv
import shutil
from pathlib import Path


def filter_manifest(manifest_path, output_dir, methods=None, exclude_methods=None,
                    K_values=None, seeds=None, append=False):
    src_dir = Path(manifest_path).parent
    out = Path(output_dir)
    if not append:
        out.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    filtered = []
    for row in rows:
        if methods and row["method"] not in methods:
            continue
        if exclude_methods and row["method"] in exclude_methods:
            continue
        if K_values and int(row["K"]) not in K_values:
            continue
        if seeds and int(row["seed"]) not in seeds:
            continue
        filtered.append(row)

    # Copy scripts
    for row in filtered:
        src = src_dir / row["filename"]
        dst = out / row["filename"]
        if src.exists():
            shutil.copy2(src, dst)

    # Write filtered manifest
    mode = "a" if append else "w"
    manifest_out = out / "manifest.csv"
    write_header = not append or not manifest_out.exists()
    with open(manifest_out, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "method", "K", "G", "seed",
                                                 "output_dir", "filename"])
        if write_header:
            writer.writeheader()
        writer.writerows(filtered)

    print(f"Filtered {len(filtered)} jobs to {out}")


def main():
    parser = argparse.ArgumentParser(description="Filter experiment jobs")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--exclude-methods", nargs="+", default=None)
    parser.add_argument("--K", nargs="+", type=int, default=None, dest="K_values")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    filter_manifest(args.manifest, args.output_dir, args.methods, args.exclude_methods,
                    args.K_values, args.seeds, args.append)


if __name__ == "__main__":
    main()
