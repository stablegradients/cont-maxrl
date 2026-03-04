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

# NeurIPS-style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Method display names and colors
METHOD_STYLES = {
    "ordinal_ce": {"label": "Ordinal CE (supervised)", "color": "#1f77b4", "ls": "--", "lw": 3.0, "zorder": 10},
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

    # Methods to include — ordinal CE plotted last so it draws on top
    methods_to_plot = [
        ("plain_ce", 0, "plain_ce"),
        ("cont_maxrl", 16, "cont_maxrl_N16"),
        ("cont_maxrl", 64, "cont_maxrl_N64"),
        ("cont_maxrl", 256, "cont_maxrl_N256"),
        ("cont_maxrl", 1024, "cont_maxrl_N1024"),
        ("grpo", 64, "grpo_N64"),
        ("ordinal_ce", 0, "ordinal_ce"),
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

            zorder = style.get("zorder", 2)
            ax.plot(epochs, mean, label=style.get("label", style_key),
                    color=style.get("color", "black"),
                    linestyle=style.get("ls", "-"),
                    linewidth=style.get("lw", 1.5),
                    zorder=zorder)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=style.get("color", "black"),
                            zorder=zorder - 1)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

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
        N_str = "N0" if method in ("ordinal_ce", "plain_ce", "mse") else "N64"
        grad_dir = Path(results_dir) / f"{method}_L100_{N_str}_seed42" / "gradient_analysis"
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

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure4_binary_recovery.pdf")
    fig.savefig(out / "figure4_binary_recovery.png")
    plt.close(fig)
    print(f"Figure 4 saved to {out}")


def plot_figure5(results_dir: str, output_dir: str, epoch: int = 50):
    """Figure 5: Per-Threshold Survival and Gradient (2x3 panels).

    Top row: survival probability S_d(x) vs threshold d for 3 representative images.
    Bottom row: gradient norm ‖∇[-log S_d]‖₂ vs threshold d for same images.
    Columns: easy / medium / hard image (by survival at d=0).
    """
    # 1.5x font scaling for figure 5
    f5_title = 16    # 11 * 1.5 ≈ 16
    f5_label = 15    # 10 * 1.5 = 15
    f5_tick = 14     #  9 * 1.5 ≈ 14
    f5_legend = 12   #  8 * 1.5 = 12

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Load per-threshold data for ordinal_ce
    grad_file = Path(results_dir) / "ordinal_ce_L100_N0_seed42" / "gradient_analysis" / f"gradient_per_threshold_epoch{epoch}.json"

    if not grad_file.exists():
        for row_idx in range(2):
            for col_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, "No data",
                    ha="center", va="center", transform=axes[row_idx, col_idx].transAxes)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "figure5_per_threshold.pdf")
        fig.savefig(out / "figure5_per_threshold.png")
        plt.close(fig)
        print(f"Figure 5 saved to {out} (no data)")
        return

    with open(grad_file) as f:
        data = json.load(f)

    # Group by image_idx
    from collections import defaultdict
    by_image = defaultdict(list)
    for d in data:
        by_image[d["image_idx"]].append(d)

    # Rank images by survival at d=0 (higher = easier)
    surv_at_0 = {}
    for img_idx, records in by_image.items():
        for r in records:
            if r["threshold_d"] == 0:
                surv_at_0[img_idx] = r["survival_prob"]
                break
    sorted_imgs = sorted(surv_at_0.items(), key=lambda x: x[1])
    n = len(sorted_imgs)

    # Pick easy (high survival), medium, hard (low survival)
    hard_img = sorted_imgs[0][0]
    med_img = sorted_imgs[n // 2][0]
    easy_img = sorted_imgs[-1][0]

    col_labels = [
        ("Easy", easy_img),
        ("Medium", med_img),
        ("Hard", hard_img),
    ]

    # Also load cont_maxrl and grpo for comparison
    methods_to_compare = [
        ("ordinal_ce", "ordinal_ce_L100_N0_seed42", "#1f77b4", "Ordinal CE"),
        ("cont_maxrl", "cont_maxrl_L100_N64_seed42", "#2ca02c", "Cont. MaxRL (N=64)"),
        ("grpo", "grpo_L100_N64_seed42", "#9467bd", "GRPO (N=64)"),
    ]

    for col_idx, (label, img_idx) in enumerate(col_labels):
        ax_surv = axes[0, col_idx]
        ax_grad = axes[1, col_idx]

        for method, run_name, color, disp_name in methods_to_compare:
            gf = Path(results_dir) / run_name / "gradient_analysis" / f"gradient_per_threshold_epoch{epoch}.json"
            if not gf.exists():
                continue
            with open(gf) as f:
                mdata = json.load(f)

            # Filter to this image
            img_records = [r for r in mdata if r["image_idx"] == img_idx]
            if not img_records:
                continue
            img_records.sort(key=lambda r: r["threshold_d"])
            ds = [r["threshold_d"] for r in img_records]
            survs = [r["survival_prob"] for r in img_records]
            gnorms = [r["gradient_norm_at_threshold"] for r in img_records]

            ax_surv.plot(ds, survs, color=color, lw=2, label=disp_name)
            ax_grad.plot(ds, gnorms, color=color, lw=2, label=disp_name)

        ax_surv.set_title(f"{label} Image (idx={img_idx})", fontsize=f5_title)
        ax_surv.set_ylabel("S_d(x)", fontsize=f5_label)
        ax_surv.set_xlabel("Threshold d", fontsize=f5_label)
        ax_surv.legend(loc="best", fontsize=f5_legend)
        ax_surv.tick_params(labelsize=f5_tick)
        ax_surv.grid(True, alpha=0.3)

        ax_grad.set_ylabel("‖∇ [-log S_d]‖₂", fontsize=f5_label)
        ax_grad.set_xlabel("Threshold d", fontsize=f5_label)
        ax_grad.legend(loc="best", fontsize=f5_legend)
        ax_grad.tick_params(labelsize=f5_tick)
        ax_grad.grid(True, alpha=0.3)

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure5_per_threshold.pdf")
    fig.savefig(out / "figure5_per_threshold.png")
    plt.close(fig)
    print(f"Figure 5 saved to {out}")


def _plot_cosine_panel(ax, cosine_file, methods_styles, ylabel):
    """Helper: plot cosine-vs-G lines on a single axis."""
    with open(cosine_file) as f:
        data = json.load(f)

    results = data["results"]
    for method, style in methods_styles.items():
        if method not in results:
            continue
        G_data = results[method]
        Gs = sorted([int(g) for g in G_data.keys()])
        means = np.array([G_data[str(g)]["mean"] for g in Gs])
        stds = np.array([G_data[str(g)]["std"] for g in Gs])

        ax.plot(Gs, means, label=style["label"], color=style["color"],
                linestyle=style["ls"], marker=style["marker"],
                linewidth=2, markersize=5)
        ax.fill_between(Gs, means - stds, np.minimum(means + stds, 1.0),
                        alpha=0.15, color=style["color"])

    ax.set_xscale("log")
    ax.set_xlabel("$G$ (rollouts per image)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return data


def plot_figure6(results_dir: str, output_dir: str, epoch: int = 25):
    """Figure 6: G vs Cosine Similarity (two panels).

    Left: Cont. MaxRL variants, GRPO, REINFORCE vs ordinal CE gradient.
    Right: Binary MaxRL vs plain CE gradient (its correct target).
    """
    # Data files
    ord_file = Path(results_dir) / "ordinal_ce_L100_N0_seed42" / "gradient_analysis" / f"cosine_vs_G_epoch{epoch}.json"
    bin_file = Path(results_dir) / "plain_ce_L100_N0_seed42" / "gradient_analysis" / f"cosine_vs_G_epoch{epoch}_ref_plain_ce.json"

    if not ord_file.exists():
        print(f"Figure 6: no ordinal CE cosine data ({ord_file})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # --- Left panel: vs ordinal CE ---
    left_styles = {
        "cont_maxrl": {"label": "Cont. MaxRL", "color": "#2ca02c", "ls": "-", "marker": "o"},
        "cont_maxrl_no_bl": {"label": "Cont. MaxRL (no baseline)", "color": "#98df8a", "ls": "--", "marker": "s"},
        "grpo": {"label": "GRPO", "color": "#9467bd", "ls": "-.", "marker": "D"},
        "reinforce": {"label": "REINFORCE", "color": "#e377c2", "ls": "-.", "marker": "v"},
    }
    _plot_cosine_panel(
        axes[0], ord_file, left_styles,
        r"Cosine sim. with $\nabla \mathcal{J}_{\mathrm{OrdCE}}$",
    )
    axes[0].set_title("vs. Ordinal CE gradient")

    # --- Right panel: binary_maxrl vs plain CE ---
    if bin_file.exists():
        right_styles = {
            "binary_maxrl": {"label": "Binary MaxRL", "color": "#8c564b", "ls": "-", "marker": "^"},
        }
        _plot_cosine_panel(
            axes[1], bin_file, right_styles,
            r"Cosine sim. with $\nabla \mathcal{J}_{\mathrm{CE}}$",
        )
    else:
        axes[1].text(0.5, 0.5, "Run cosine analysis with\n--ref_method plain_ce",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
    axes[1].set_title("vs. Plain CE gradient")

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure6_cosine_vs_G.pdf")
    fig.savefig(out / "figure6_cosine_vs_G.png")
    plt.close(fig)
    print(f"Figure 6 saved to {out}")


def plot_figure7(results_dir: str, output_dir: str):
    """Figure 7: Cosine similarity and magnitude ratio grids.

    For each reference (ordinal_ce, plain_ce, mse):
      Two heatmaps — cosine similarity and magnitude ratio.
      Rows: RL methods. Columns: epochs (1, 10, 25, 50).
      Each cell shows the value at G=64.
    Also produces per-reference detailed grids with G on y-axis.
    """
    refs = [
        ("ordinal_ce", r"Ordinal CE ($\mathcal{J}_{\mathrm{OrdCE}}$)"),
        ("plain_ce", r"Plain CE ($\mathcal{J}_{\mathrm{CE}}$)"),
        ("mse", r"MSE"),
    ]
    rl_methods = ["cont_maxrl", "cont_maxrl_no_bl", "binary_maxrl", "grpo", "reinforce"]
    rl_labels = ["Cont. MaxRL", "No Baseline", "Binary MaxRL", "GRPO", "REINFORCE"]
    epochs = [1, 10, 25, 50]
    G_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Per-reference detailed grid: G × epoch for each RL method ---
    for ref_key, ref_label in refs:
        # Load all data for this reference
        # MSE uses ordinal_ce model (MSE model has 1-output regression head)
        model_dir = "ordinal_ce" if ref_key == "mse" else ref_key
        all_data = {}  # (epoch) -> loaded json
        for epoch in epochs:
            suffix = f"_ref_{ref_key}" if ref_key != "ordinal_ce" else ""
            f = Path(results_dir) / f"{model_dir}_L100_N0_seed42" / "gradient_analysis" / f"cosine_vs_G_epoch{epoch}{suffix}.json"
            if f.exists():
                with open(f) as fh:
                    all_data[epoch] = json.load(fh)

        if not all_data:
            print(f"Figure 7 ({ref_key}): no data found, skipping")
            continue

        # Two rows (cosine, magnitude) × 5 columns (RL methods)
        fig, axes = plt.subplots(2, len(rl_methods), figsize=(4 * len(rl_methods), 8))
        metric_keys = [
            ("mean", "Cosine similarity"),
            ("mag_ratio_mean", "Magnitude ratio"),
        ]

        for col, (method, method_label) in enumerate(zip(rl_methods, rl_labels)):
            for row, (mkey, mlabel) in enumerate(metric_keys):
                ax = axes[row, col]

                # Build G × epoch matrix
                avail_epochs = sorted(all_data.keys())
                grid = np.full((len(G_values), len(avail_epochs)), np.nan)
                for ei, ep in enumerate(avail_epochs):
                    results = all_data[ep].get("results", {})
                    if method not in results:
                        continue
                    for gi, G in enumerate(G_values):
                        entry = results[method].get(str(G), {})
                        val = entry.get(mkey, np.nan)
                        grid[gi, ei] = val

                if mkey == "mean":
                    vmin, vmax = -0.2, 1.0
                    cmap = "RdYlGn"
                else:
                    vmin, vmax = 0, max(3.0, np.nanmax(grid) if not np.all(np.isnan(grid)) else 3.0)
                    cmap = "YlOrRd"

                im = ax.imshow(grid, aspect="auto", origin="lower",
                               vmin=vmin, vmax=vmax, cmap=cmap)

                # Annotate cells
                for gi in range(len(G_values)):
                    for ei in range(len(avail_epochs)):
                        v = grid[gi, ei]
                        if not np.isnan(v):
                            color = "white" if (mkey == "mean" and v < 0.3) or (mkey != "mean" and v > vmax * 0.7) else "black"
                            ax.text(ei, gi, f"{v:.2f}", ha="center", va="center",
                                    fontsize=7, color=color)

                ax.set_xticks(range(len(avail_epochs)))
                ax.set_xticklabels([str(e) for e in avail_epochs])
                ax.set_yticks(range(len(G_values)))
                ax.set_yticklabels([str(g) for g in G_values])

                if row == 0:
                    ax.set_title(method_label, fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{mlabel}\nG", fontsize=10)
                if row == 1:
                    ax.set_xlabel("Epoch", fontsize=10)

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"vs. {ref_label} gradient", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(out / f"figure7_grid_{ref_key}.pdf")
        fig.savefig(out / f"figure7_grid_{ref_key}.png")
        plt.close(fig)
        print(f"Figure 7 ({ref_key}) saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment figures")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--figures", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
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
    if 6 in args.figures:
        plot_figure6(args.results_dir, args.output_dir)
    if 7 in args.figures:
        plot_figure7(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
