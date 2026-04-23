"""Sweep B plot: x = difficulty, y = pass@1 — GOS+continuous vs GOS+binary.

Isolates the value of a continuous reward signal while fixing the
estimator. See spec §6.3 "Sweep B".

Usage:
    python -m experiment10_reasoning_gym.src.plot_reward_ablation \\
        --task word_sorting --output figures/sweep_b_word_sorting.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment10_reasoning_gym.src.plot_difficulty_scaling import fetch_best_pass1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["word_sorting", "knights_knaves"])
    parser.add_argument("--entity", default="stablegradients")
    parser.add_argument("--metric", default="val/test_score/reward")
    parser.add_argument("--output", default="figures/sweep_b.png")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    diffs = [4, 6, 8, 10, 12] if args.task == "word_sorting" else [3, 5, 7]
    project = f"reasoning_gym_{args.task}"

    gos_cont = []
    gos_bin = []
    for d in diffs:
        gos_cont.append(fetch_best_pass1(
            args.entity, project,
            f"gos_continuous_diff{d}_G16_batch256_s{args.seed}", args.metric))
        gos_bin.append(fetch_best_pass1(
            args.entity, project,
            f"gos_binary_diff{d}_G16_batch256_s{args.seed}", args.metric))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(diffs, gos_cont, marker="o", label="GOS + continuous")
    ax.plot(diffs, gos_bin, marker="^", label="GOS + binary")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel(args.metric)
    ax.set_title(f"{args.task}: Sweep B (reward type ablation)")
    ax.legend()
    ax.grid(alpha=0.3)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")
    print(f"GOS+cont: {gos_cont}")
    print(f"GOS+bin:  {gos_bin}")


if __name__ == "__main__":
    main()
