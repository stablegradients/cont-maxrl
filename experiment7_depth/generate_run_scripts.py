"""Generate all 165 per-job bash scripts for the depth experiment sweep.

Usage:
    python -m experiment7_depth.generate_run_scripts --output_dir experiment7_depth/scripts/runs
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path

SUPERVISED_METHODS = ["ordinal_ce", "plain_ce", "mse"]
RL_METHODS = ["cont_maxrl", "grpo", "reinforce", "binary_maxrl"]
ALL_METHODS = SUPERVISED_METHODS + RL_METHODS

K_VALUES = [10, 100, 1000]
G_VALUES = [4, 16, 64, 256]
SEEDS = [42, 43, 44]

PYTHON_BIN = "/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"


def job_id(method: str, K: int, G: int, seed: int) -> str:
    key = f"{method}|K{K}|G{G}|s{seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:5]
    return f"zx7k_{h}"


SCRIPT_TEMPLATE = """#!/bin/bash
# Job: {job_id} | Method: {method} | K: {K} | G: {G} | Seed: {seed}
set -euo pipefail
export WANDB_PROJECT="cont-maxrl-depth"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${{PYTHON_BIN:-{python_bin}}}"

cd "$REPO_ROOT"

# 4-GPU DDP launch via torchrun. CUDA_VISIBLE_DEVICES set by submit_all.sh.
torchrun --standalone --nproc_per_node=4 \\
    -m experiment7_depth.run \\
    --method {method} \\
    --K {K} \\
    --G {G} \\
    --seed {seed} \\
    --global_batch_size 32 \\
    --epochs 25 \\
    --epochs_hard_cap 35 \\
    --epochs_min_before_es 18 \\
    --early_stop_patience 8 \\
    --lr 2e-4 \\
    --lr_schedule polynomial \\
    --lr_power 0.9 \\
    --momentum 0.9 \\
    --weight_decay 5e-4 \\
    --amp \\
    --data_dir "$REPO_ROOT/data/nyu" \\
    --output_dir "$REPO_ROOT/experiment7_depth/results/{job_id}" \\
    --wandb \\
    --wandb_run_name "{job_id}"
"""


def generate_all(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    def add_run(method, K, G, seed):
        jid = job_id(method, K, G, seed)
        filename = f"{jid}.sh"
        content = SCRIPT_TEMPLATE.format(
            job_id=jid, method=method, K=K, G=G, seed=seed,
            python_bin=PYTHON_BIN,
        )
        filepath = out / filename
        filepath.write_text(content)
        filepath.chmod(0o755)
        manifest_rows.append({
            "job_id": jid, "method": method, "K": K, "G": G,
            "seed": seed, "output_dir": f"experiment7_depth/results/{jid}",
            "filename": filename,
        })

    # Supervised: ordinal_ce and plain_ce at all K values
    for method in ["ordinal_ce", "plain_ce"]:
        for K in K_VALUES:
            for seed in SEEDS:
                add_run(method, K, 0, seed)

    # Supervised: mse (K-agnostic training, use K=0 as placeholder)
    for seed in SEEDS:
        add_run("mse", 0, 0, seed)

    # RL: all methods x K x G x seeds
    for method in RL_METHODS:
        for K in K_VALUES:
            for G in G_VALUES:
                for seed in SEEDS:
                    add_run(method, K, G, seed)

    # Write manifest
    manifest_path = out / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_id", "method", "K", "G", "seed",
                                                 "output_dir", "filename"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} scripts + manifest.csv in {out}")
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description="Generate experiment run scripts")
    parser.add_argument("--output_dir", type=str,
                        default="experiment7_depth/scripts/runs")
    args = parser.parse_args()
    generate_all(args.output_dir)


if __name__ == "__main__":
    main()
