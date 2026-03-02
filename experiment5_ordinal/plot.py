"""Plot generation for ordinal regression experiment (Figures 1-5).

Can generate from local JSON metrics or W&B export.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Method display names and colors
METHOD_STYLES = {
    "ordinal_ce": {"label": "Ordinal CE (supervised)", "color": "#1f77b4", "ls": "--", "lw": 2.5},
    "plain_ce": {"label": "Plain CE (supervised)", "color": "#aec7e8", "ls": "--", "lw": 2.0},
    "cont_maxrl_N16": {"label": "Cont. MaxRL (N=16)", "color": "#2ca02c", "ls": "-", "lw": 1.5},
    "cont_maxrl_N64": {"label": "Cont. MaxRL (N=64)", "color": "#98df8a", "ls": "-", "lw": 1.8},
    "cont_maxrl_N256": {"label": "Cont. MaxRL (N=256)", "color": "#d62728", "ls": "-", "lw": 2.0},
    "cont_maxrl_N1024": {"label": "Cont. MaxRL (N=1024)", "color": "#ff7f0e", "ls": "-", "lw": 2.2},
    "grpo_N64": {"label": "GRPO (N=64)", "color": "#9467bd", "ls": "-.", "lw": 2.0},
    "binary_maxrl_N64": {"label": "Binary MaxRL (N=64)", "color": "#8c564b", "ls": ":", "lw": 2.0},
    "mse": {"label": "MSE Regression", "color": "#7f7f7f", "ls": "--", "lw": 1.5},
    "reinforce_N64": {"label": "REINFORCE (N=64)", "color": "#e377c2", "ls": "-.", "lw": 1.5},
}


def load_metrics(results_dir: str, method: str, N: int, L: int, seeds: list[int]) -> dict:
    """Load metrics from local JSON files across seeds.

    Returns dict with keys like 'val/mae' -> (epochs, seeds) arrays.
    """
    all_seed_data = []
    for seed in seeds:
        run_name = f"{method}_L{L}_N{N}_seed{seed}"
        path = Path(results_dir) / run_name / "metrics.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        all_seed_data.append(data)

    if not all_seed_data:
        return {}

    # Stack across seeds
    n_epochs = len(all_seed_data[0])
    result = {}
    for key in all_seed_data[0][0]:
        if key == "epoch":
            result["epoch"] = [d["epoch"] for d in all_seed_data[0]]
            continue
        try:
            values = np.array([[d[key] for d in seed_data] for seed_data in all_seed_data])
            result[key] = values  # (n_seeds, n_epochs)
        except (KeyError, TypeError):
            pass

    return result


def plot_figure1(results_dir: str, output_dir: str, L: int = 100, seeds: list[int] = None):
    """Figure 1: Training Dynamics (1x3 panels).

    Panel 1: MAE, Panel 2: Best-of-64 Reward, Panel 3: Ordinal CE Objective
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Methods to include: ordinal CE, plain CE, cont MaxRL at multiple N, GRPO at N=64
    methods_to_plot = [
        ("ordinal_ce", 0, "ordinal_ce"),
        ("plain_ce", 0, "plain_ce"),
        ("cont_maxrl", 16, "cont_maxrl_N16"),
        ("cont_maxrl", 64, "cont_maxrl_N64"),
        ("cont_maxrl", 256, "cont_maxrl_N256"),
        ("cont_maxrl", 1024, "cont_maxrl_N1024"),
        ("grpo", 64, "grpo_N64"),
    ]

    metric_panels = [
        ("val/mae", "Val MAE (years) ↓"),
        ("val/best_of_64_reward", "Best-of-64 Reward ↑"),
        ("val/ordinal_ce_objective", "Ordinal CE Objective ↑"),
    ]

    for panel_idx, (metric_key, ylabel) in enumerate(metric_panels):
        ax = axes[panel_idx]

        for method, N, style_key in methods_to_plot:
            data = load_metrics(results_dir, method, N, L, seeds)
            if not data or metric_key not in data:
                continue

            style = METHOD_STYLES.get(style_key, {})
            values = data[metric_key]  # (n_seeds, n_epochs)
            epochs = data["epoch"]

            mean = values.mean(axis=0)
            std = values.std(axis=0)

            ax.plot(epochs, mean, label=style.get("label", style_key),
                    color=style.get("color", "black"),
                    linestyle=style.get("ls", "-"),
                    linewidth=style.get("lw", 1.5))
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=style.get("color", "black"))

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Dynamics — Continuous MaxRL → Ordinal CE", fontsize=16, y=1.02)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure1_training_dynamics.pdf")
    fig.savefig(out / "figure1_training_dynamics.png")
    plt.close(fig)
    print(f"Figure 1 saved to {out}")


def plot_figure2(results_dir: str, output_dir: str, epoch: int = 25):
    """Figure 2: Gradient Geometry (1x4 panels).

    Scatter: x = P(exact match), y = gradient L2 norm.
    Panels: Ordinal CE, Cont MaxRL, GRPO, REINFORCE.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    methods = ["ordinal_ce", "cont_maxrl", "grpo", "reinforce"]
    titles = ["Ordinal CE", "Continuous MaxRL", "GRPO", "REINFORCE"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2"]

    for idx, (method, title, color) in enumerate(zip(methods, titles, colors)):
        ax = axes[idx]

        # Try to load gradient analysis results
        grad_dir = Path(results_dir) / f"{method}_L100_N64_seed42" / "gradient_analysis"
        grad_file = grad_dir / f"gradient_per_image_epoch{epoch}.json"

        if not grad_file.exists():
            # Try alternative path
            grad_file = Path(results_dir) / "gradient_analysis" / method / f"gradient_per_image_epoch{epoch}.json"

        if grad_file.exists():
            with open(grad_file) as f:
                data = json.load(f)

            x = [d["p_exact_match"] for d in data]
            y = [d["gradient_l2_norm"] for d in data]

            ax.scatter(x, y, alpha=0.3, s=10, c=color)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("P(exact match)")
        ax.set_ylabel("‖∇ loss‖₂")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Gradient Norm vs P(exact match) — Epoch {epoch}", fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure2_gradient_geometry.pdf")
    fig.savefig(out / "figure2_gradient_geometry.png")
    plt.close(fig)
    print(f"Figure 2 saved to {out}")


def plot_figure3(results_dir: str, output_dir: str, L: int = 100, seeds: list[int] = None):
    """Figure 3: Convergence to Ordinal CE as N→∞ (1x2 panels).

    Panel 1: Final J_ML vs N. Panel 2: Final MAE vs N.
    """
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    N_values = [4, 16, 64, 256, 1024]

    # Get supervised ordinal CE final values
    oc_data = load_metrics(results_dir, "ordinal_ce", 0, L, seeds)

    for panel_idx, (metric_key, ylabel, label) in enumerate([
        ("val/ordinal_ce_objective", "Final J_ML ↑", "Ordinal CE Objective"),
        ("val/mae", "Final MAE (years) ↓", "MAE"),
    ]):
        ax = axes[panel_idx]

        # Supervised baseline
        if oc_data and metric_key in oc_data:
            final_vals = oc_data[metric_key][:, -1]  # last epoch per seed
            oc_mean = final_vals.mean()
            ax.axhline(oc_mean, color="#1f77b4", ls="--", lw=2,
                       label=f"Ordinal CE (supervised): {oc_mean:.3f}")

        # MaxRL at each N
        means, stds, ns = [], [], []
        for N in N_values:
            data = load_metrics(results_dir, "cont_maxrl", N, L, seeds)
            if not data or metric_key not in data:
                continue
            final_vals = data[metric_key][:, -1]
            means.append(final_vals.mean())
            stds.append(final_vals.std())
            ns.append(N)

            # Plot individual seeds
            for val in final_vals:
                ax.scatter(N, val, c="#2ca02c", alpha=0.4, s=30, zorder=3)

        if means:
            ax.plot(ns, means, "o-", color="#2ca02c", lw=2, label="Cont. MaxRL", zorder=4)

        ax.set_xscale("log")
        ax.set_xlabel("N (rollouts per image)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_values)
        ax.set_xticklabels([str(n) for n in N_values])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence to Ordinal CE as N → ∞", fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure3_convergence.pdf")
    fig.savefig(out / "figure3_convergence.png")
    plt.close(fig)
    print(f"Figure 3 saved to {out}")


def plot_figure4(results_dir: str, output_dir: str, seeds: list[int] = None):
    """Figure 4: Binary Recovery Sanity Check (L=2, 1x2 panels)."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Binary MaxRL vs Plain CE
    ax = axes[0]
    for method, N, style_key in [
        ("plain_ce", 0, "plain_ce"),
        ("binary_maxrl", 64, "binary_maxrl_N64"),
        ("binary_maxrl", 256, None),
        ("binary_maxrl", 1024, None),
    ]:
        data = load_metrics(results_dir, method, N, 2, seeds)
        if not data or "val/acc_exact" not in data:
            continue

        values = data["val/acc_exact"]
        epochs = data["epoch"]
        mean = values.mean(axis=0)

        label = f"Binary MaxRL (N={N})" if method == "binary_maxrl" else "Plain CE"
        ax.plot(epochs, mean, label=label, lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Binary MaxRL vs Plain CE (L=2)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Continuous MaxRL vs Ordinal CE
    ax = axes[1]
    for method, N, label in [
        ("ordinal_ce", 0, "Ordinal CE"),
        ("cont_maxrl", 64, "Cont. MaxRL (N=64)"),
        ("cont_maxrl", 256, "Cont. MaxRL (N=256)"),
        ("cont_maxrl", 1024, "Cont. MaxRL (N=1024)"),
    ]:
        data = load_metrics(results_dir, method, N, 2, seeds)
        if not data or "val/acc_exact" not in data:
            continue

        values = data["val/acc_exact"]
        epochs = data["epoch"]
        mean = values.mean(axis=0)

        ax.plot(epochs, mean, label=label, lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cont. MaxRL vs Ordinal CE (L=2, should collapse)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Binary Recovery Sanity Check (L=2)", fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure4_binary_recovery.pdf")
    fig.savefig(out / "figure4_binary_recovery.png")
    plt.close(fig)
    print(f"Figure 4 saved to {out}")


def plot_figure5(results_dir: str, output_dir: str, epoch: int = 50):
    """Figure 5: Per-Threshold Survival and Gradient (2x3 panels, appendix)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Try to load per-threshold data
    grad_dir = Path(results_dir) / "gradient_analysis" / "ordinal_ce"
    for ep in [1, epoch]:
        grad_file = grad_dir / f"gradient_per_threshold_epoch{ep}.json"
        if not grad_file.exists():
            # Try alternative paths
            for run_pattern in ["ordinal_ce_L100_N0_seed42", "ordinal_ce_L100_N64_seed42"]:
                alt = Path(results_dir) / run_pattern / "gradient_analysis" / f"gradient_per_threshold_epoch{ep}.json"
                if alt.exists():
                    grad_file = alt
                    break

    # If no data, create placeholder
    for row_idx in range(2):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.text(0.5, 0.5, "Data not yet available\nRun gradient analysis first",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("Distance d" if row_idx == 1 else "Distance d")
            if row_idx == 0:
                ax.set_ylabel("W_d(x)")
                ax.set_title(f"{'Easy' if col_idx == 0 else 'Medium' if col_idx == 1 else 'Hard'} Image")
            else:
                ax.set_ylabel("‖∇ [-log W_d]‖₂")

    fig.suptitle("Per-Threshold Survival and Gradient Decomposition", fontsize=14, y=1.02)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure5_per_threshold.pdf")
    fig.savefig(out / "figure5_per_threshold.png")
    plt.close(fig)
    print(f"Figure 5 saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment figures")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--figures", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="Which figures to generate")
    args = parser.parse_args()

    if 1 in args.figures:
        plot_figure1(args.results_dir, args.output_dir, seeds=args.seeds)
    if 2 in args.figures:
        plot_figure2(args.results_dir, args.output_dir)
    if 3 in args.figures:
        plot_figure3(args.results_dir, args.output_dir, seeds=args.seeds)
    if 4 in args.figures:
        plot_figure4(args.results_dir, args.output_dir, seeds=args.seeds)
    if 5 in args.figures:
        plot_figure5(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
