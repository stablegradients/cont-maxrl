"""
Generate paper figures from evaluation results.

Figures:
  A: pass@k curves (money figure)
  B: training curves (reward vs step, from wandb)
  C: per-problem solve rate distributions
  D: normalized vs unnormalized MaxRL comparison across benchmarks
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


# ── Styling constants ──

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
METHODS = ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]
EVAL_BENCHMARKS = ["apps", "code_contests", "livecodebench"]
TRAIN_DATASETS = ["apps", "taco", "code_contests"]
WANDB_ENTITY = "stablegradients"
WANDB_PROJECT = "cont-maxrl-code"
SEEDS = {
    "1.5b": [42, 43, 44],
    "7b": [42, 43, 44],
    "32b": [42, 43],
}


def load_eval_results(
    results_dir: str,
    model: str,
    train_dataset: str,
    method: str,
    G: int,
    eval_benchmark: str,
) -> dict:
    """Load and average eval results across seeds.

    Path convention:
        results/{model}_{dataset}_{method}_g{G}_s{seed}/eval_{benchmark}/eval_results.json
    """
    seeds = SEEDS.get(model, [42, 43, 44])
    all_results = []
    for seed in seeds:
        exp_name = f"{model}_{train_dataset}_{method}_g{G}_s{seed}"
        path = (
            Path(results_dir)
            / exp_name
            / f"eval_{eval_benchmark}"
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

    return {"avg": avg, "std": std, "n_seeds": len(all_results), "raw": all_results}


def figure_a_pass_at_k(
    results_dir: str,
    output_dir: str,
    model: str,
    train_dataset: str,
    G: int,
    eval_benchmark: str,
):
    """Figure A: pass@k curves -- the money figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    has_data = False

    for method in METHODS:
        data = load_eval_results(
            results_dir, model, train_dataset, method, G, eval_benchmark
        )
        if not data or not data["avg"]:
            continue

        has_data = True
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

    if not has_data:
        plt.close()
        return

    ax.set_xscale("log")
    ax.set_xlabel("k (number of attempts)", fontsize=12)
    ax.set_ylabel("pass@k (%)", fontsize=12)
    title = f"pass@k: {model.upper()} on {eval_benchmark.upper()} (G={G})"
    if train_dataset != eval_benchmark:
        title += f"\ntrained on {train_dataset.upper()}"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = (
        Path(output_dir)
        / f"fig_a_pass_at_k_{model}_{train_dataset}_g{G}_on_{eval_benchmark}.pdf"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def figure_b_training_curves(
    results_dir: str,
    output_dir: str,
    model: str,
    train_dataset: str,
    G: int,
):
    """Figure B: Training reward curves from wandb."""
    try:
        import wandb
        api = wandb.Api()
    except Exception:
        print("    Skipped fig B (wandb not available)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    has_data = False

    for method in METHODS:
        group = f"{model}_{train_dataset}_{method}_g{G}"
        try:
            runs = list(api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"group": group},
            ))
        except Exception:
            continue

        if not runs:
            continue

        # Collect training reward histories
        all_steps = []
        all_rewards = []
        for run in runs:
            try:
                history = run.history(pandas=True)
                if history.empty:
                    continue
                # Find reward metric column (verl uses various names)
                reward_col = None
                for col in history.columns:
                    if "reward" in col.lower() and "mean" in col.lower():
                        reward_col = col
                        break
                if reward_col is None:
                    continue
                steps = history["_step"].values
                rewards = history[reward_col].values
                mask = ~np.isnan(rewards)
                if mask.sum() > 1:
                    all_steps.append(steps[mask])
                    all_rewards.append(rewards[mask])
            except Exception:
                continue

        if not all_steps:
            continue

        has_data = True

        # Interpolate to common step grid and average
        max_step = max(s[-1] for s in all_steps)
        common_steps = np.linspace(0, max_step, num=200)
        interpolated = []
        for s, r in zip(all_steps, all_rewards):
            interpolated.append(np.interp(common_steps, s, r))

        mean_r = np.mean(interpolated, axis=0)
        std_r = np.std(interpolated, axis=0)
        ax.plot(
            common_steps, mean_r,
            color=COLORS[method], label=LABELS[method], linewidth=2,
        )
        ax.fill_between(
            common_steps, mean_r - std_r, mean_r + std_r,
            alpha=0.15, color=COLORS[method],
        )

    if not has_data:
        plt.close()
        print(f"    Skipped fig B (no wandb data for {model}_{train_dataset}_g{G})")
        return

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title(
        f"Training Reward: {model.upper()} on {train_dataset.upper()} (G={G})",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / f"fig_b_training_{model}_{train_dataset}_g{G}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def figure_c_reward_distribution(
    results_dir: str,
    output_dir: str,
    model: str,
    train_dataset: str,
    G: int,
    eval_benchmark: str,
):
    """Figure C: Per-problem solve rate CDF at convergence."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    has_data = False

    for method in METHODS:
        data = load_eval_results(
            results_dir, model, train_dataset, method, G, eval_benchmark
        )
        if not data or "raw" not in data:
            continue

        # Collect per-problem solve rates (c/n) across all seeds
        solve_rates = []
        for result in data["raw"]:
            for p in result.get("per_problem", []):
                if p["n"] > 0:
                    solve_rates.append(p["c"] / p["n"])

        if not solve_rates:
            continue

        has_data = True
        sorted_rates = np.sort(solve_rates)
        cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
        ax.plot(
            sorted_rates, cdf,
            color=COLORS[method], label=LABELS[method], linewidth=2,
        )

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Per-problem solve rate (c / n)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    title = f"Solve Rate Distribution: {model.upper()} (G={G})"
    title += f"\nTrained: {train_dataset.upper()} | Eval: {eval_benchmark.upper()}"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = (
        Path(output_dir)
        / f"fig_c_dist_{model}_{train_dataset}_g{G}_on_{eval_benchmark}.pdf"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def figure_d_norm_vs_unnorm(
    results_dir: str,
    output_dir: str,
    model: str,
    train_dataset: str,
    G: int,
):
    """Figure D: MaxRL normalized vs unnormalized comparison across benchmarks."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    valid_benchmarks = []
    unnorm_vals = []
    norm_vals = []
    unnorm_errs = []
    norm_errs = []
    k = 10  # Compare on pass@10

    for bench in EVAL_BENCHMARKS:
        unnorm = load_eval_results(
            results_dir, model, train_dataset, "maxrl_unnorm", G, bench
        )
        norm = load_eval_results(
            results_dir, model, train_dataset, "maxrl_norm", G, bench
        )
        if (
            unnorm and unnorm["avg"] and k in unnorm["avg"]
            and norm and norm["avg"] and k in norm["avg"]
        ):
            valid_benchmarks.append(bench)
            unnorm_vals.append(unnorm["avg"][k] * 100)
            norm_vals.append(norm["avg"][k] * 100)
            unnorm_errs.append(unnorm["std"].get(k, 0) * 100)
            norm_errs.append(norm["std"].get(k, 0) * 100)

    if not valid_benchmarks:
        plt.close()
        return

    x = np.arange(len(valid_benchmarks))
    width = 0.35
    ax.bar(
        x - width / 2, unnorm_vals, width, yerr=unnorm_errs,
        label="MaxRL (unnorm)", color=COLORS["maxrl_unnorm"], capsize=4,
    )
    ax.bar(
        x + width / 2, norm_vals, width, yerr=norm_errs,
        label="MaxRL (norm)", color=COLORS["maxrl_norm"], capsize=4,
    )

    ax.set_ylabel("pass@10 (%)", fontsize=12)
    ax.set_title(
        f"MaxRL: Normalized vs Unnormalized\n"
        f"{model.upper()} trained on {train_dataset.upper()} (G={G})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in valid_benchmarks])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = (
        Path(output_dir)
        / f"fig_d_norm_vs_unnorm_{model}_{train_dataset}_g{G}.pdf"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def figure_e_scaling_with_g(
    results_dir: str,
    output_dir: str,
    model: str,
    train_dataset: str,
    eval_benchmark: str,
):
    """Figure E: pass@10 vs G showing MaxRL improves while GRPO saturates."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    methods = ["grpo", "maxrl_unnorm", "maxrl_norm"]
    group_sizes = [4, 16, 32]
    has_data = False

    for method in methods:
        pass10_means = []
        pass10_stds = []
        valid_gs = []

        for G in group_sizes:
            data = load_eval_results(
                results_dir, model, train_dataset, method, G, eval_benchmark
            )
            if data and data["avg"] and 10 in data["avg"]:
                pass10_means.append(data["avg"][10] * 100)
                pass10_stds.append(data["std"][10] * 100)
                valid_gs.append(G)

        if valid_gs:
            has_data = True
            ax.errorbar(
                valid_gs, pass10_means, yerr=pass10_stds,
                color=COLORS[method],
                label=LABELS[method],
                marker="o",
                linewidth=2,
                capsize=4,
            )

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Group size (G)", fontsize=12)
    ax.set_ylabel("pass@10 (%)", fontsize=12)
    title = f"pass@10 vs Group Size: {model.upper()}"
    title += f"\nTrained: {train_dataset.upper()} | Eval: {eval_benchmark.upper()}"
    ax.set_title(title, fontsize=13)
    ax.set_xticks(group_sizes)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = (
        Path(output_dir)
        / f"fig_e_scaling_g_{model}_{train_dataset}_on_{eval_benchmark}.pdf"
    )
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
        for train_dataset in TRAIN_DATASETS:
            print(f"Model: {model}, Dataset: {train_dataset}")

            for G in [4, 16, 32]:
                # Figure A: pass@k (one per eval benchmark)
                for bench in EVAL_BENCHMARKS:
                    figure_a_pass_at_k(
                        args.results_dir, args.output_dir,
                        model, train_dataset, G, bench,
                    )

                # Figure B: training curves (from wandb)
                figure_b_training_curves(
                    args.results_dir, args.output_dir,
                    model, train_dataset, G,
                )

                # Figure C: reward distributions (one per eval benchmark)
                for bench in EVAL_BENCHMARKS:
                    figure_c_reward_distribution(
                        args.results_dir, args.output_dir,
                        model, train_dataset, G, bench,
                    )

                # Figure D: norm vs unnorm comparison
                figure_d_norm_vs_unnorm(
                    args.results_dir, args.output_dir,
                    model, train_dataset, G,
                )

            # Figure E: scaling with G (one per eval benchmark)
            for bench in EVAL_BENCHMARKS:
                figure_e_scaling_with_g(
                    args.results_dir, args.output_dir,
                    model, train_dataset, bench,
                )

            print()

    print("Done!")


if __name__ == "__main__":
    main()
