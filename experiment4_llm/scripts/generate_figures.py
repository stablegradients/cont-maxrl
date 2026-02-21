"""
Generate paper figures from evaluation results.

Figures:
  A: pass@k curves (money figure)
  B: training curves (pass@1 and pass@10 vs step)
  C: reward distributions at convergence
  D: normalized vs unnormalized MaxRL
  E: scaling with G

Usage:
    python experiment4_llm/scripts/generate_figures.py \
        --results_dir results/ \
        --output_dir figures/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Consistent styling
COLORS = {
    "grpo": "#1f77b4",
    "maxrl_unnorm": "#d62728",
    "maxrl_norm": "#ff7f0e",
    "reinforce_baseline": "#2ca02c",
}
LABELS = {
    "grpo": "GRPO",
    "maxrl_unnorm": "MaxRL (unnorm)",
    "maxrl_norm": "MaxRL (norm)",
    "reinforce_baseline": "REINFORCE+bl",
}
K_VALUES = [1, 2, 5, 10, 20, 50, 100]


def load_eval_results(results_dir: str, model: str, method: str, G: int) -> dict:
    """Load and average eval results across seeds."""
    all_results = []
    for seed in [42, 43, 44]:
        path = (
            Path(results_dir)
            / f"{model}_apps_{method}_g{G}_s{seed}"
            / "eval"
            / "eval_results.json"
        )
        if path.exists():
            with open(path) as f:
                all_results.append(json.load(f))

    if not all_results:
        return {}

    # Average pass@k across seeds
    avg = {}
    std = {}
    for k in K_VALUES:
        values = [
            r["pass_at_k"].get(str(k), r["pass_at_k"].get(k))
            for r in all_results
            if r["pass_at_k"].get(str(k), r["pass_at_k"].get(k)) is not None
        ]
        if values:
            avg[k] = np.mean(values)
            std[k] = np.std(values)

    return {"avg": avg, "std": std, "n_seeds": len(all_results)}


def figure_a_pass_at_k(results_dir: str, output_dir: str, model: str, G: int):
    """Figure A: pass@k curves -- the money figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    methods = ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]

    for method in methods:
        data = load_eval_results(results_dir, model, method, G)
        if not data or not data["avg"]:
            continue

        ks = sorted(data["avg"].keys())
        means = [data["avg"][k] * 100 for k in ks]
        stds = [data["std"][k] * 100 for k in ks]

        ax.plot(
            ks, means,
            color=COLORS[method],
            label=LABELS[method],
            marker="o",
            linewidth=2,
        )
        ax.fill_between(
            ks,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=COLORS[method],
        )

    ax.set_xscale("log")
    ax.set_xlabel("k (number of attempts)", fontsize=12)
    ax.set_ylabel("pass@k (%)", fontsize=12)
    ax.set_title(f"pass@k: {model.upper()} on APPS (G={G})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / f"fig_a_pass_at_k_{model}_g{G}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def figure_e_scaling_with_g(results_dir: str, output_dir: str, model: str):
    """Figure E: pass@10 vs G showing MaxRL improves while GRPO saturates."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    methods = ["grpo", "maxrl_unnorm", "maxrl_norm"]
    group_sizes = [4, 16, 32]

    for method in methods:
        pass10_means = []
        pass10_stds = []
        valid_gs = []

        for G in group_sizes:
            data = load_eval_results(results_dir, model, method, G)
            if data and data["avg"] and 10 in data["avg"]:
                pass10_means.append(data["avg"][10] * 100)
                pass10_stds.append(data["std"][10] * 100)
                valid_gs.append(G)

        if valid_gs:
            ax.errorbar(
                valid_gs, pass10_means, yerr=pass10_stds,
                color=COLORS[method],
                label=LABELS[method],
                marker="o",
                linewidth=2,
                capsize=4,
            )

    ax.set_xlabel("Group size (G)", fontsize=12)
    ax.set_ylabel("pass@10 (%)", fontsize=12)
    ax.set_title(f"pass@10 vs Group Size: {model.upper()} on APPS", fontsize=14)
    ax.set_xticks(group_sizes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / f"fig_e_scaling_g_{model}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--output_dir", type=str, default="figures/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Generating Figures ===\n")

    for model in ["1.5b", "7b", "32b"]:
        print(f"Model: {model}")
        for G in [4, 16, 32]:
            figure_a_pass_at_k(args.results_dir, args.output_dir, model, G)
        figure_e_scaling_with_g(args.results_dir, args.output_dir, model)
        print()

    print("Done!")


if __name__ == "__main__":
    main()
