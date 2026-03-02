"""Auto-generate all standalone bash scripts for the experiment sweep.

Generates scripts:
- 9 supervised: 3 methods × 3 seeds
- 75 RL: 5 methods × 5 N values × 3 seeds
- binary recovery (L=2): 5 RL methods × 3 seeds (N=64), plus supervised, plus higher N

Usage:
    python -m experiment5_ordinal.sweep --output_dir experiment5_ordinal/scripts/runs
"""

import argparse
import os
from pathlib import Path

SUPERVISED_METHODS = ["ordinal_ce", "plain_ce", "mse"]
RL_METHODS = ["cont_maxrl", "cont_maxrl_no_bl", "binary_maxrl", "grpo", "reinforce"]
N_VALUES = [16, 64, 256]
SEEDS = [42, 43, 44]

# LR by method type
SUPERVISED_LR = "1e-3"
RL_LR = "1e-3"

SCRIPT_TEMPLATE = """#!/bin/bash
# {description}
# Method: {method} | N: {N} | L: {L} | Seed: {seed}
# Expected runtime: ~{runtime} on A5000
set -euo pipefail

export WANDB_PROJECT="cont-maxrl-ordinal"
export WANDB_RUN_GROUP="{group}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"

cd "$REPO_ROOT"

$PYTHON -m experiment5_ordinal.run \\
    --method {method} \\
    --N {N} \\
    --L {L} \\
    --seed {seed} \\
    --epochs 50 \\
    --batch_size 64 \\
    --lr {lr} \\
    --input_size 224 \\
    --data_dir "$REPO_ROOT/data/utkface" \\
    --output_dir "$REPO_ROOT/experiment5_ordinal/results" \\
    --wandb
"""


def generate_script(
    method: str, N: int, L: int, seed: int, output_dir: Path
) -> str:
    """Generate a single standalone bash script."""
    is_supervised = method in SUPERVISED_METHODS
    lr = SUPERVISED_LR if is_supervised else RL_LR

    if is_supervised:
        filename = f"train_{method}_L{L}_seed{seed}.sh"
        description = f"Supervised: {method}, L={L}, seed={seed}"
        runtime = "10 min"
    else:
        filename = f"train_{method}_N{N}_L{L}_seed{seed}.sh"
        description = f"RL: {method}, N={N}, L={L}, seed={seed}"
        runtime = "25 min"

    group = f"{method}_L{L}_N{N}"

    content = SCRIPT_TEMPLATE.format(
        description=description,
        method=method,
        N=N,
        L=L,
        seed=seed,
        lr=lr,
        group=group,
        runtime=runtime,
    )

    filepath = output_dir / filename
    filepath.write_text(content)
    filepath.chmod(0o755)

    return filename


def generate_all(output_dir: str):
    """Generate all 135+ standalone scripts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated = []

    # === L=100 supervised (15 scripts) ===
    print("Generating supervised scripts (L=100)...")
    for method in SUPERVISED_METHODS:
        for seed in SEEDS:
            fname = generate_script(method, 0, 100, seed, out)
            generated.append(fname)

    # === L=100 RL (100 scripts) ===
    print("Generating RL scripts (L=100)...")
    for method in RL_METHODS:
        for N in N_VALUES:
            for seed in SEEDS:
                fname = generate_script(method, N, 100, seed, out)
                generated.append(fname)

    # === L=2 binary recovery (20 scripts: 4 RL × 5 seeds at N=64, plus supervised) ===
    print("Generating binary recovery scripts (L=2)...")
    # Supervised at L=2
    for method in ["ordinal_ce", "plain_ce"]:
        for seed in SEEDS:
            fname = generate_script(method, 0, 2, seed, out)
            generated.append(fname)
    # RL at L=2 with N=64
    for method in RL_METHODS:
        for seed in SEEDS:
            fname = generate_script(method, 64, 2, seed, out)
            generated.append(fname)
    # Also binary/cont MaxRL at higher N for L=2
    for method in ["binary_maxrl", "cont_maxrl"]:
        for N in [256, 1024]:
            for seed in SEEDS:
                fname = generate_script(method, N, 2, seed, out)
                generated.append(fname)

    print(f"\nGenerated {len(generated)} scripts in {out}")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment scripts")
    parser.add_argument("--output_dir", type=str,
                        default="experiment5_ordinal/scripts/runs")
    args = parser.parse_args()

    generate_all(args.output_dir)
