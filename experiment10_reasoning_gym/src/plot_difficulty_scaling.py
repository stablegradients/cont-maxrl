"""Sweep A plot: x = difficulty, y = pass@1 — GOS+continuous vs MaxRL+binary.

Pulls runs from W&B (via the wandb API) for each task and plots the
difficulty-scaling curve for the two estimators. Run after training
finishes. For now this is a skeleton — tune the metric name after the
first run is logged to confirm keys match ``val/*``.

Usage:
    python -m experiment10_reasoning_gym.src.plot_difficulty_scaling \\
        --task word_sorting --output figures/sweep_a_word_sorting.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fetch_best_pass1(entity: str, project: str, run_name: str,
                     metric: str = "val/test_score/reward") -> float | None:
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError("wandb not installed in this env") from e
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_name}))
    if not runs:
        return None
    run = runs[0]
    history = run.history(keys=[metric])
    if history.empty:
        return None
    return float(history[metric].max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["word_sorting", "knights_knaves"])
    parser.add_argument("--entity", default="stablegradients")
    parser.add_argument("--metric", default="val/test_score/reward",
                        help="W&B metric key to use for y-axis (max over history).")
    parser.add_argument("--output", default="figures/sweep_a.png")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    if args.task == "word_sorting":
        diffs = [4, 6, 8, 10, 12]
    else:
        diffs = [3, 5, 7]

    project = f"reasoning_gym_{args.task}"

    gos_cont = []
    maxrl_bin = []
    for d in diffs:
        gos_name = f"gos_continuous_diff{d}_G16_batch256_s{args.seed}"
        maxrl_name = f"maxrl_binary_diff{d}_G16_batch256_s{args.seed}"
        gos_cont.append(fetch_best_pass1(args.entity, project, gos_name, args.metric))
        maxrl_bin.append(fetch_best_pass1(args.entity, project, maxrl_name, args.metric))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(diffs, gos_cont, marker="o", label="GOS + continuous")
    ax.plot(diffs, maxrl_bin, marker="s", label="MaxRL + binary")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{args.task}: Sweep A (estimator comparison)")
    ax.legend()
    ax.grid(alpha=0.3)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")
    print(f"GOS+cont:  {gos_cont}")
    print(f"MaxRL+bin: {maxrl_bin}")


if __name__ == "__main__":
    main()
