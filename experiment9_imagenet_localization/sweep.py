"""Generate SLURM submission scripts for the ImageNet localization sweep.

Writes one .sh per configuration plus:
  - submit_all.sh            — submits every script (use only if all stages wanted at once)
  - submit_stage{1..4}.sh    — per-stage launchers

Default output directory:
  /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/

CLI:
  python -m experiment9_imagenet_localization.sweep \
      [--out_dir DIR] \
      [--results_dir DIR] \
      [--data_dir DIR] \
      [--partition NAME] \
      [--conda_env NAME] \
      [--train_subsample N] \
      [--epochs N] \
      [--time HH:MM:SS] \
      [--include_mse]
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = ("gos", "binary_maxrl", "grpo", "rloo", "reinforce", "dg",
           "ordinal_ce", "plain_ce")
K_VALUES = (10, 25, 50, 100)
SEEDS = (42, 43, 44)

N_PRIMARY = 64
N_SWEEP_VALUES = (4, 16, 64, 256)

REPO_DIR = "/zfsauton/scratch/shrinivr/cont-maxrl"
EXPERIMENT_DIR = os.path.join(REPO_DIR, "experiment9_imagenet_localization")

DEFAULT_OUT_DIR = os.path.join(EXPERIMENT_DIR, "sweep_scripts")
DEFAULT_RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
DEFAULT_DATA_DIR = "/zfsauton/datasets/imagenet"
DEFAULT_PARTITION = "general"
DEFAULT_CONDA_ENV = "base"
DEFAULT_EPOCHS = 30
# 0 means "no subsample" — run.py treats --train_subsample absent as full data.
DEFAULT_TRAIN_SUBSAMPLE = 0
DEFAULT_TIME = "48:00:00"

# SLURM job script template.  Variables wrapped in {{}} are literal braces
# (Python format escapes); single-brace {} are format placeholders.
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time}
#SBATCH --output={log_dir}/{job_name}_%j.out

set -euo pipefail

# wandb reads credentials from ~/.netrc (written by `wandb login`), so no
# WANDB_API_KEY env export is needed. The earlier strict `${{VAR:?...}}` check
# was killing jobs before they started.
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="{wandb_group}"

cd "{repo_dir}"

{conda_activate}

python -m experiment9_imagenet_localization.run \\
    --method {method} \\
    --K {K} \\
    --N {N} \\
    --seed {seed} \\
    --epochs {epochs} \\
    --batch_size 128 \\
    --lr 5e-4 \\
    --data_dir "{data_dir}" \\
    --output_dir "{run_output_dir}" \\
    --num_workers 8 \\
{subsample_line}\
    --wandb
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conda_activate_block(conda_env: str) -> str:
    """Return bash snippet that activates *conda_env*.

    If *conda_env* is an empty string, returns an empty string (no
    activation).  Otherwise emits ``source ~/.bashrc; conda activate ENV``.
    """
    if not conda_env:
        return ""
    return f"source ~/.bashrc\nconda activate {conda_env}"


def _job_name(method: str, K: int, N: int, seed: int) -> str:
    """Return an opaque SLURM job name — deterministic 8-char hex derived from
    the (method, K, N, seed) tuple. This keeps `squeue` on a shared cluster
    from revealing which experiment / method is being run. The mapping is
    reproducible so logs can still be matched back to configs locally.
    """
    import hashlib
    key = f"{method}|{K}|{N}|{seed}".encode()
    digest = hashlib.md5(key).hexdigest()[:8]
    return f"j_{digest}"


def _script_filename(method: str, K: int, N: int, seed: int) -> str:
    """Return the .sh filename for a given configuration."""
    return f"{method}_K{K}_N{N}_seed{seed}.sh"


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


def generate_script_content(
    method: str,
    K: int,
    N: int,
    seed: int,
    results_dir: str,
    data_dir: str,
    partition: str,
    conda_env: str,
    train_subsample: int,
    epochs: int,
    time: str,
    log_dir: str,
    repo_dir: str,
) -> str:
    """Return the full text of a SLURM submission script for one config."""
    job_name = _job_name(method, K, N, seed)
    # Per-run output dir so W&B / checkpoints never collide between seeds.
    run_name = f"{method}_K{K}_N{N}_seed{seed}"
    run_output_dir = os.path.join(results_dir, run_name)
    # W&B group: group by (method, K, N) so the three seeds appear together.
    wandb_group = f"{method}_K{K}_N{N}"
    conda_activate = _conda_activate_block(conda_env)

    # train_subsample <= 0 means "no subsample" — omit the flag entirely so
    # run.py falls back to its default (None = full dataset).
    if train_subsample and train_subsample > 0:
        subsample_line = f"    --train_subsample {train_subsample} \\\n"
    else:
        subsample_line = ""

    return SLURM_TEMPLATE.format(
        job_name=job_name,
        partition=partition,
        time=time,
        log_dir=log_dir,
        wandb_group=wandb_group,
        repo_dir=repo_dir,
        conda_activate=conda_activate,
        method=method,
        K=K,
        N=N,
        seed=seed,
        epochs=epochs,
        data_dir=data_dir,
        run_output_dir=run_output_dir,
        subsample_line=subsample_line,
    )


def write_script(
    out_dir: str,
    method: str,
    K: int,
    N: int,
    seed: int,
    results_dir: str,
    data_dir: str,
    partition: str,
    conda_env: str,
    train_subsample: int,
    epochs: int,
    time: str,
    log_dir: str,
    repo_dir: str,
) -> str:
    """Write one .sh file and make it executable; return its absolute path."""
    content = generate_script_content(
        method=method,
        K=K,
        N=N,
        seed=seed,
        results_dir=results_dir,
        data_dir=data_dir,
        partition=partition,
        conda_env=conda_env,
        train_subsample=train_subsample,
        epochs=epochs,
        time=time,
        log_dir=log_dir,
        repo_dir=repo_dir,
    )
    filename = _script_filename(method, K, N, seed)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as fh:
        fh.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------


def stage_groups(include_mse: bool = False) -> dict[int, list[tuple[str, int, int, int]]]:
    """Return a dict mapping stage number → list of (method, K, N, seed) tuples.

    Stage 1 (headline, 24):        gos/binary_maxrl × K × seeds, N=64
    Stage 2 (supervised, 24+12):   ordinal_ce/plain_ce × K × seeds, N=64
                                   (+mse × K × seeds if include_mse)
    Stage 3 (other RL, 48):        grpo/rloo/reinforce/dg × K × seeds, N=64
    Stage 4 (supplementary, 26):   N-sweep (24) + K=2 sanity (2)
    """
    stages: dict[int, list[tuple[str, int, int, int]]] = {1: [], 2: [], 3: [], 4: []}

    # ---------- Stage 1: headline RL ----------
    for method in ("gos", "binary_maxrl"):
        for K in K_VALUES:
            for seed in SEEDS:
                stages[1].append((method, K, N_PRIMARY, seed))

    # ---------- Stage 2: supervised ----------
    for method in ("ordinal_ce", "plain_ce"):
        for K in K_VALUES:
            for seed in SEEDS:
                stages[2].append((method, K, N_PRIMARY, seed))

    if include_mse:
        for K in K_VALUES:
            for seed in SEEDS:
                stages[2].append(("mse", K, N_PRIMARY, seed))

    # ---------- Stage 3: other RL baselines ----------
    for method in ("grpo", "rloo", "reinforce", "dg"):
        for K in K_VALUES:
            for seed in SEEDS:
                stages[3].append((method, K, N_PRIMARY, seed))

    # ---------- Stage 4: supplementary ----------
    # N-sweep: gos/binary_maxrl × K=50 × N ∈ {4,16,64,256} × seeds.
    # N=N_PRIMARY (=64) is skipped because Stage 1 already generates
    # (method, K=50, N=64, seed) for every seed — including it here would
    # cause submit_all.sh and submit_stage4.sh to re-submit the same script.
    for method in ("gos", "binary_maxrl"):
        for N in N_SWEEP_VALUES:
            if N == N_PRIMARY:
                continue
            for seed in SEEDS:
                stages[4].append((method, 50, N, seed))

    # K=2 sanity: gos/binary_maxrl × K=2 × N=64 × seed=42
    for method in ("gos", "binary_maxrl"):
        stages[4].append((method, 2, N_PRIMARY, 42))

    return stages


# ---------------------------------------------------------------------------
# Submit-script writers
# ---------------------------------------------------------------------------


def _write_launcher(path: str, script_paths: list[str], header_comment: str) -> None:
    """Write a launcher shell script that sbatches every script in *script_paths*."""
    lines = [
        "#!/bin/bash",
        "# " + header_comment,
        "set -euo pipefail",
        "",
    ]
    for sp in script_paths:
        lines.append(f'sbatch "{sp}"')
    lines.append("")  # trailing newline
    content = "\n".join(lines)
    with open(path, "w") as fh:
        fh.write(content)
    os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def write_submit_all(out_dir: str, per_stage_scripts: dict[int, list[str]]) -> str:
    """Write submit_all.sh; return its path."""
    all_scripts: list[str] = []
    for stage_num in sorted(per_stage_scripts.keys()):
        all_scripts.extend(per_stage_scripts[stage_num])
    path = os.path.join(out_dir, "submit_all.sh")
    total = len(all_scripts)
    _write_launcher(path, all_scripts, f"Submit all {total} sweep scripts across all stages.")
    return path


def write_submit_stage(out_dir: str, stage_num: int, scripts: list[str]) -> str:
    """Write submit_stage{N}.sh; return its path."""
    path = os.path.join(out_dir, f"submit_stage{stage_num}.sh")
    _write_launcher(
        path,
        scripts,
        f"Submit Stage {stage_num} ({len(scripts)} scripts).",
    )
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help="Directory where generated .sh scripts are written.",
    )
    parser.add_argument(
        "--results_dir",
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for per-run output (checkpoints, logs).",
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory of the ImageNet dataset.",
    )
    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        help="SLURM partition name (confirm with cluster admin before submitting).",
    )
    parser.add_argument(
        "--conda_env",
        default=DEFAULT_CONDA_ENV,
        help="Conda environment to activate. Pass '' to skip activation.",
    )
    parser.add_argument(
        "--train_subsample",
        type=int,
        default=DEFAULT_TRAIN_SUBSAMPLE,
        help="Number of training images to subsample (passed as --train_subsample to run.py).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs per run.",
    )
    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
        help="SLURM wall-clock time limit per job (HH:MM:SS).",
    )
    parser.add_argument(
        "--repo_dir",
        default=REPO_DIR,
        help="Absolute path to the repository root (used as `cd` target in scripts).",
    )
    parser.add_argument(
        "--include_mse",
        action="store_true",
        default=False,
        help=(
            "Include optional MSE regressor baseline in Stage 2 "
            "(adds 4 K-values × 3 seeds = 12 extra scripts; total 134)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    stages = stage_groups(include_mse=args.include_mse)
    per_stage_scripts: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    for stage_num, configs in stages.items():
        for (method, K, N, seed) in configs:
            path = write_script(
                out_dir=out_dir,
                method=method,
                K=K,
                N=N,
                seed=seed,
                results_dir=args.results_dir,
                data_dir=args.data_dir,
                partition=args.partition,
                conda_env=args.conda_env,
                train_subsample=args.train_subsample,
                epochs=args.epochs,
                time=args.time,
                log_dir=log_dir,
                repo_dir=args.repo_dir,
            )
            per_stage_scripts[stage_num].append(path)

    # Write launcher scripts
    write_submit_all(out_dir, per_stage_scripts)
    for stage_num, scripts in per_stage_scripts.items():
        write_submit_stage(out_dir, stage_num, scripts)

    total = sum(len(s) for s in per_stage_scripts.values())
    print(f"Wrote {total} scripts to {out_dir}")
    for stage_num, scripts in sorted(per_stage_scripts.items()):
        print(f"  Stage {stage_num}: {len(scripts)} scripts")
    print(f"Launchers written: submit_all.sh + submit_stage{{1..4}}.sh")


if __name__ == "__main__":
    main()
