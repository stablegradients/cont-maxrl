"""Plotting module for experiment7_depth. 12 figures + 3 summary tables."""

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLORS = {
    "cont_maxrl":   "#1D9E75",  # GOS green
    "grpo":         "#7F77DD",  # purple
    "binary_maxrl": "#D85A30",  # coral-red
    "reinforce":    "#888888",  # gray
    "ordinal_ce":   "#0D5E45",  # dark green (supervised oracle)
    "plain_ce":     "#4A3F9E",  # dark purple (supervised CE baseline)
    "mse":          "#000000",  # black (regression reference)
}

LINESTYLES = {
    "supervised": "--",
    "rl": "-",
    "reference": ":",
}

SUPERVISED_METHODS = ["ordinal_ce", "plain_ce", "mse"]
RL_METHODS = ["cont_maxrl", "grpo", "reinforce", "binary_maxrl"]
ALL_METHODS = SUPERVISED_METHODS + RL_METHODS

METHOD_LABELS = {
    "cont_maxrl":   "GOS (ours)",
    "grpo":         "GRPO",
    "binary_maxrl": "Binary MaxRL",
    "reinforce":    "REINFORCE",
    "ordinal_ce":   "Ordinal CE (sup.)",
    "plain_ce":     "Plain CE (sup.)",
    "mse":          "MSE (sup.)",
}

K_VALUES = [10, 100, 1000]
G_VALUES = [4, 16, 64, 256]

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------

def setup_matplotlib():
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox_inches": "tight",
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_manifest(path):
    """Load manifest CSV into list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def load_run_metrics(results_dir, job_id):
    """Load metrics.json for a single run. Returns list of per-epoch dicts or None."""
    path = Path(results_dir) / job_id / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_metrics(results_dir, manifest):
    """Load all run metrics, keyed by job_id."""
    all_metrics = {}
    for row in manifest:
        jid = row["job_id"]
        metrics = load_run_metrics(results_dir, jid)
        if metrics is not None:
            row_copy = dict(row)
            row_copy["metrics"] = metrics
            all_metrics[jid] = row_copy
    return all_metrics


def get_best_epoch_metric(metrics_list, metric_key, minimize=True):
    """Get the value of *metric_key* at the epoch with best val/mae_meters."""
    if not metrics_list:
        return float("nan")
    best_epoch_idx = 0
    best_val = float("inf") if minimize else float("-inf")
    for i, m in enumerate(metrics_list):
        v = m.get("val/mae_meters", float("inf"))
        if minimize and v < best_val:
            best_val = v
            best_epoch_idx = i
        elif not minimize and v > best_val:
            best_val = v
            best_epoch_idx = i
    return metrics_list[best_epoch_idx].get(metric_key, float("nan"))


def get_epoch_series(metrics_list, metric_key):
    """Return (epochs, values) arrays for a metric over training."""
    if not metrics_list:
        return np.array([]), np.array([])
    epochs = np.array([m.get("epoch", i) for i, m in enumerate(metrics_list)])
    values = np.array([m.get(metric_key, float("nan")) for m in metrics_list])
    return epochs, values


def group_by(all_metrics, method=None, K=None, G=None):
    """Filter and group runs by criteria. Returns list of run dicts."""
    result = []
    for _jid, run in all_metrics.items():
        if method is not None and run["method"] != method:
            continue
        if K is not None and int(run["K"]) != K:
            continue
        if G is not None and int(run.get("G", 0)) != G:
            continue
        result.append(run)
    return result


def seed_aggregate(runs, metric_key, minimize=True):
    """Aggregate metric across seeds. Returns (mean, std, values)."""
    values = []
    for run in runs:
        v = get_best_epoch_metric(run["metrics"], metric_key, minimize)
        if not np.isnan(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan"), []
    return float(np.mean(values)), float(np.std(values)), values


def seed_aggregate_series(runs, metric_key):
    """Aggregate epoch-level metric across seeds.

    Returns (epochs, mean_array, std_array) where mean/std are over seeds.
    """
    all_series = []
    ref_epochs = None
    for run in runs:
        epochs, vals = get_epoch_series(run["metrics"], metric_key)
        if len(vals) == 0:
            continue
        if ref_epochs is None:
            ref_epochs = epochs
        all_series.append(vals)
    if not all_series:
        return np.array([]), np.array([]), np.array([])
    # Truncate to shortest common length
    min_len = min(len(s) for s in all_series)
    stacked = np.array([s[:min_len] for s in all_series])
    return ref_epochs[:min_len], stacked.mean(axis=0), stacked.std(axis=0)


def _get_linestyle(method):
    """Return linestyle based on whether method is supervised or RL."""
    if method in SUPERVISED_METHODS:
        return LINESTYLES["supervised"]
    return LINESTYLES["rl"]


def save_figure(fig, output_dir, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 01: K-sweep headline MAE
# ---------------------------------------------------------------------------

def plot_figure_01(all_metrics, output_dir):
    """K-sweep headline: X=K (log), Y=val/mae_meters at best epoch.

    Lines: supervised ordinal/plain/mse + RL at G=64. Shaded +/-1 std.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    methods_to_plot = SUPERVISED_METHODS + RL_METHODS

    for method in methods_to_plot:
        ks, means, stds = [], [], []
        for K in K_VALUES:
            if method in SUPERVISED_METHODS:
                runs = group_by(all_metrics, method=method, K=K)
            else:
                runs = group_by(all_metrics, method=method, K=K, G=64)
            if not runs:
                continue
            m, s, _ = seed_aggregate(runs, "val/mae_meters", minimize=True)
            if np.isnan(m):
                continue
            ks.append(K)
            means.append(m)
            stds.append(s)
        if not ks:
            continue
        ks, means, stds = np.array(ks), np.array(means), np.array(stds)
        ax.plot(ks, means, marker="o", label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method, "black"), linestyle=_get_linestyle(method), linewidth=2)
        ax.fill_between(ks, means - stds, means + stds,
                        alpha=0.15, color=COLORS.get(method, "black"))

    ax.set_xscale("log")
    ax.set_xlabel("K (number of bins)")
    ax.set_ylabel("Val MAE (meters) at best epoch")
    ax.set_title("K-sweep: MAE")
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig01_kscan_mae_meters")


# ---------------------------------------------------------------------------
# Figure 02: K-sweep ordinal CE objective
# ---------------------------------------------------------------------------

def plot_figure_02(all_metrics, output_dir):
    """K-sweep: Y=val/ordinal_ce_objective (higher=better)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    methods_to_plot = SUPERVISED_METHODS + RL_METHODS

    for method in methods_to_plot:
        ks, means, stds = [], [], []
        for K in K_VALUES:
            if method in SUPERVISED_METHODS:
                runs = group_by(all_metrics, method=method, K=K)
            else:
                runs = group_by(all_metrics, method=method, K=K, G=64)
            if not runs:
                continue
            # ordinal_ce_objective: higher is better, but best epoch still selected by mae
            m, s, _ = seed_aggregate(runs, "val/ordinal_ce_objective", minimize=True)
            if np.isnan(m):
                continue
            ks.append(K)
            means.append(m)
            stds.append(s)
        if not ks:
            continue
        ks, means, stds = np.array(ks), np.array(means), np.array(stds)
        ax.plot(ks, means, marker="o", label=METHOD_LABELS.get(method, method),
                color=COLORS.get(method, "black"), linestyle=_get_linestyle(method), linewidth=2)
        ax.fill_between(ks, means - stds, means + stds,
                        alpha=0.15, color=COLORS.get(method, "black"))

    ax.set_xscale("log")
    ax.set_xlabel("K (number of bins)")
    ax.set_ylabel("Val Ordinal CE Objective (higher=better)")
    ax.set_title("K-sweep: Ordinal CE Objective")
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig02_kscan_ordinal_ce")


# ---------------------------------------------------------------------------
# Figure 03: K-sweep depth metrics (3 panels)
# ---------------------------------------------------------------------------

def plot_figure_03(all_metrics, output_dir):
    """3-panel: abs_rel, rmse, delta_1.25 across K. Same methods as fig01."""
    panels = [
        ("val/abs_rel", "Abs Rel Error", True),
        ("val/rmse_meters", "RMSE (meters)", True),
        ("val/delta_1.25", r"$\delta < 1.25$ accuracy", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods_to_plot = SUPERVISED_METHODS + RL_METHODS

    for panel_idx, (metric_key, ylabel, minimize) in enumerate(panels):
        ax = axes[panel_idx]
        for method in methods_to_plot:
            ks, means, stds = [], [], []
            for K in K_VALUES:
                if method in SUPERVISED_METHODS:
                    runs = group_by(all_metrics, method=method, K=K)
                else:
                    runs = group_by(all_metrics, method=method, K=K, G=64)
                if not runs:
                    continue
                m, s, _ = seed_aggregate(runs, metric_key, minimize=True)
                if np.isnan(m):
                    continue
                ks.append(K)
                means.append(m)
                stds.append(s)
            if not ks:
                continue
            ks_a, means_a, stds_a = np.array(ks), np.array(means), np.array(stds)
            ax.plot(ks_a, means_a, marker="o",
                    label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method, "black"),
                    linestyle=_get_linestyle(method), linewidth=2)
            ax.fill_between(ks_a, means_a - stds_a, means_a + stds_a,
                            alpha=0.15, color=COLORS.get(method, "black"))
        ax.set_xscale("log")
        ax.set_xlabel("K (number of bins)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig03_kscan_depth_metrics")


# ---------------------------------------------------------------------------
# Figure 04: G-scan per K (3 panels)
# ---------------------------------------------------------------------------

def plot_figure_04(all_metrics, output_dir):
    """3 panels (K=10,100,1000). X=G, Y=mae_meters.

    4 RL methods + horizontal supervised references.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for panel_idx, K in enumerate(K_VALUES):
        ax = axes[panel_idx]
        # Supervised horizontal references
        for method in SUPERVISED_METHODS:
            runs = group_by(all_metrics, method=method, K=K)
            if not runs:
                continue
            m, s, _ = seed_aggregate(runs, "val/mae_meters", minimize=True)
            if np.isnan(m):
                continue
            ax.axhline(m, color=COLORS.get(method, "black"),
                       linestyle=LINESTYLES["supervised"], linewidth=1.5,
                       label=METHOD_LABELS.get(method, method))

        # RL curves over G
        for method in RL_METHODS:
            gs, means, stds = [], [], []
            for G in G_VALUES:
                runs = group_by(all_metrics, method=method, K=K, G=G)
                if not runs:
                    continue
                m, s, _ = seed_aggregate(runs, "val/mae_meters", minimize=True)
                if np.isnan(m):
                    continue
                gs.append(G)
                means.append(m)
                stds.append(s)
            if not gs:
                continue
            gs_a, means_a, stds_a = np.array(gs), np.array(means), np.array(stds)
            ax.plot(gs_a, means_a, marker="o",
                    label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method, "black"),
                    linestyle=LINESTYLES["rl"], linewidth=2)
            ax.fill_between(gs_a, means_a - stds_a, means_a + stds_a,
                            alpha=0.15, color=COLORS.get(method, "black"))

        ax.set_xscale("log")
        ax.set_xlabel("G (rollouts per image)")
        ax.set_ylabel("Val MAE (meters)")
        ax.set_title(f"K = {K}")
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig04_gscan_per_k")


# ---------------------------------------------------------------------------
# Figure 05: Binary collapse
# ---------------------------------------------------------------------------

def plot_figure_05(all_metrics, output_dir):
    """2 panels. Left: X=K, Y=exact_match_rate for binary_maxrl G=64 + 1/K ref.
    Right: X=K, Y=mae_meters for binary_maxrl.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: exact match rate
    ax = axes[0]
    ks, means, stds = [], [], []
    for K in K_VALUES:
        runs = group_by(all_metrics, method="binary_maxrl", K=K, G=64)
        if not runs:
            continue
        m, s, _ = seed_aggregate(runs, "val/exact_match_rate", minimize=True)
        if np.isnan(m):
            continue
        ks.append(K)
        means.append(m)
        stds.append(s)
    if ks:
        ks_a, means_a, stds_a = np.array(ks), np.array(means), np.array(stds)
        ax.plot(ks_a, means_a, marker="o", color=COLORS["binary_maxrl"],
                linewidth=2, label="Binary MaxRL (G=64)")
        ax.fill_between(ks_a, means_a - stds_a, means_a + stds_a,
                        alpha=0.15, color=COLORS["binary_maxrl"])
    # 1/K reference line
    ks_ref = np.array(K_VALUES, dtype=float)
    ax.plot(ks_ref, 1.0 / ks_ref, linestyle=":", color="gray",
            linewidth=1.5, label="Random (1/K)")
    ax.set_xscale("log")
    ax.set_xlabel("K (number of bins)")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Binary Collapse: Exact Match")
    ax.legend(loc="best")

    # Right panel: mae_meters
    ax = axes[1]
    ks, means, stds = [], [], []
    for K in K_VALUES:
        runs = group_by(all_metrics, method="binary_maxrl", K=K, G=64)
        if not runs:
            continue
        m, s, _ = seed_aggregate(runs, "val/mae_meters", minimize=True)
        if np.isnan(m):
            continue
        ks.append(K)
        means.append(m)
        stds.append(s)
    if ks:
        ks_a, means_a, stds_a = np.array(ks), np.array(means), np.array(stds)
        ax.plot(ks_a, means_a, marker="o", color=COLORS["binary_maxrl"],
                linewidth=2, label="Binary MaxRL (G=64)")
        ax.fill_between(ks_a, means_a - stds_a, means_a + stds_a,
                        alpha=0.15, color=COLORS["binary_maxrl"])
    ax.set_xscale("log")
    ax.set_xlabel("K (number of bins)")
    ax.set_ylabel("Val MAE (meters)")
    ax.set_title("Binary Collapse: MAE")
    ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig05_binary_collapse")


# ---------------------------------------------------------------------------
# Figure 06: Training dynamics
# ---------------------------------------------------------------------------

def plot_figure_06(all_metrics, output_dir):
    """3 panels (K=10,100,1000). X=epoch, Y=mae_meters. All methods at G=64."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods_to_plot = SUPERVISED_METHODS + RL_METHODS

    for panel_idx, K in enumerate(K_VALUES):
        ax = axes[panel_idx]
        for method in methods_to_plot:
            if method in SUPERVISED_METHODS:
                runs = group_by(all_metrics, method=method, K=K)
            else:
                runs = group_by(all_metrics, method=method, K=K, G=64)
            if not runs:
                continue
            epochs, mean, std = seed_aggregate_series(runs, "val/mae_meters")
            if len(epochs) == 0:
                continue
            ax.plot(epochs, mean, label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method, "black"),
                    linestyle=_get_linestyle(method), linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=COLORS.get(method, "black"))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MAE (meters)")
        ax.set_title(f"K = {K}")
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig06_training_dynamics")


# ---------------------------------------------------------------------------
# Figure 07: Gradient cosine similarity
# ---------------------------------------------------------------------------

def plot_figure_07(all_metrics, output_dir):
    """3 panels (K=10,100,1000). X=epoch, Y=cosine similarity.

    Loaded from gradient_analysis output (val/gradient_cosine_sim).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for panel_idx, K in enumerate(K_VALUES):
        ax = axes[panel_idx]
        for method in RL_METHODS:
            runs = group_by(all_metrics, method=method, K=K, G=64)
            if not runs:
                continue
            epochs, mean, std = seed_aggregate_series(runs, "val/gradient_cosine_sim")
            if len(epochs) == 0:
                continue
            ax.plot(epochs, mean, label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method, "black"),
                    linestyle="-", linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=COLORS.get(method, "black"))

        ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine similarity with oracle gradient")
        ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"K = {K}")
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig07_gradient_cosine")


# ---------------------------------------------------------------------------
# Figure 08: Reward histograms
# ---------------------------------------------------------------------------

def plot_figure_08(all_metrics, output_dir):
    """2x2 panels. K=100/1000 x binary_maxrl/cont_maxrl.

    Histogram of per-pixel rewards from val/reward_histogram if available.
    """
    ks_plot = [100, 1000]
    methods_plot = ["binary_maxrl", "cont_maxrl"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row_idx, K in enumerate(ks_plot):
        for col_idx, method in enumerate(methods_plot):
            ax = axes[row_idx, col_idx]
            runs = group_by(all_metrics, method=method, K=K, G=64)
            if not runs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{METHOD_LABELS.get(method, method)}, K={K}")
                continue

            # Try to load reward histogram from the best epoch of first seed
            run = runs[0]
            best_idx = 0
            best_mae = float("inf")
            for i, m in enumerate(run["metrics"]):
                v = m.get("val/mae_meters", float("inf"))
                if v < best_mae:
                    best_mae = v
                    best_idx = i
            epoch_data = run["metrics"][best_idx]
            reward_hist = epoch_data.get("val/reward_histogram", None)
            if reward_hist is not None and isinstance(reward_hist, list):
                ax.hist(reward_hist, bins=50, color=COLORS.get(method, "black"),
                        alpha=0.7, edgecolor="white")
            else:
                ax.text(0.5, 0.5, "No histogram data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_title(f"{METHOD_LABELS.get(method, method)}, K={K}")
            ax.set_xlabel("Per-pixel reward")
            ax.set_ylabel("Count")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig08_reward_histograms")


# ---------------------------------------------------------------------------
# Figure 09: Cumulative accuracy at K=1000
# ---------------------------------------------------------------------------

def plot_figure_09(all_metrics, output_dir):
    """X=threshold d, Y=acc_within_d at K=1000. Multiple methods."""
    fig, ax = plt.subplots(figsize=(6, 4))
    K = 1000
    methods_to_plot = SUPERVISED_METHODS + RL_METHODS

    for method in methods_to_plot:
        if method in SUPERVISED_METHODS:
            runs = group_by(all_metrics, method=method, K=K)
        else:
            runs = group_by(all_metrics, method=method, K=K, G=64)
        if not runs:
            continue
        # Look for cumulative accuracy data in the best epoch
        run = runs[0]
        best_idx = 0
        best_mae = float("inf")
        for i, m in enumerate(run["metrics"]):
            v = m.get("val/mae_meters", float("inf"))
            if v < best_mae:
                best_mae = v
                best_idx = i
        epoch_data = run["metrics"][best_idx]
        cum_acc = epoch_data.get("val/cumulative_accuracy", None)
        if cum_acc is not None and isinstance(cum_acc, dict):
            thresholds = sorted([float(t) for t in cum_acc.keys()])
            accuracies = [cum_acc[str(t)] if str(t) in cum_acc else cum_acc[t]
                          for t in thresholds]
            ax.plot(thresholds, accuracies,
                    label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method, "black"),
                    linestyle=_get_linestyle(method), linewidth=2)
        elif cum_acc is not None and isinstance(cum_acc, list):
            # Assume evenly-spaced thresholds
            thresholds = np.arange(len(cum_acc))
            ax.plot(thresholds, cum_acc,
                    label=METHOD_LABELS.get(method, method),
                    color=COLORS.get(method, "black"),
                    linestyle=_get_linestyle(method), linewidth=2)
        else:
            # Skip this method
            continue

    ax.set_xlabel("Threshold d (bins)")
    ax.set_ylabel("Accuracy within d")
    ax.set_title("Cumulative Accuracy (K=1000)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig09_cumulative_acc_K1000")


# ---------------------------------------------------------------------------
# Figure 10: Rankings heatmap
# ---------------------------------------------------------------------------

def plot_figure_10(all_metrics, output_dir):
    """7 methods x 3 K. Cell=mae_meters, color viridis."""
    methods_to_plot = ALL_METHODS
    n_methods = len(methods_to_plot)
    n_ks = len(K_VALUES)

    grid = np.full((n_methods, n_ks), np.nan)
    for mi, method in enumerate(methods_to_plot):
        for ki, K in enumerate(K_VALUES):
            if method in SUPERVISED_METHODS:
                runs = group_by(all_metrics, method=method, K=K)
            else:
                runs = group_by(all_metrics, method=method, K=K, G=64)
            if not runs:
                continue
            m, _, _ = seed_aggregate(runs, "val/mae_meters", minimize=True)
            grid[mi, ki] = m

    fig, ax = plt.subplots(figsize=(6, 5))
    # Mask NaN for display
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="viridis_r", aspect="auto")

    # Annotate cells
    for mi in range(n_methods):
        for ki in range(n_ks):
            v = grid[mi, ki]
            if not np.isnan(v):
                text_color = "white" if v > np.nanmedian(grid) else "black"
                ax.text(ki, mi, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color)

    ax.set_xticks(range(n_ks))
    ax.set_xticklabels([f"K={K}" for K in K_VALUES])
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods_to_plot])
    ax.set_title("MAE (meters) Rankings")
    fig.colorbar(im, ax=ax, label="MAE (meters)")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig10_rankings_heatmap")


# ---------------------------------------------------------------------------
# Figure 11: Exact match evolution for binary_maxrl
# ---------------------------------------------------------------------------

def plot_figure_11(all_metrics, output_dir):
    """3 panels (K). X=epoch, Y=exact_match_rate. binary_maxrl at G=4,16,64,256."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    g_colors = {4: "#FFA07A", 16: "#D85A30", 64: "#A0522D", 256: "#5A1A00"}

    for panel_idx, K in enumerate(K_VALUES):
        ax = axes[panel_idx]
        for G in G_VALUES:
            runs = group_by(all_metrics, method="binary_maxrl", K=K, G=G)
            if not runs:
                continue
            epochs, mean, std = seed_aggregate_series(runs, "val/exact_match_rate")
            if len(epochs) == 0:
                continue
            ax.plot(epochs, mean, label=f"G={G}",
                    color=g_colors.get(G, "black"), linewidth=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=g_colors.get(G, "black"))

        # 1/K reference
        ax.axhline(1.0 / K, color="gray", ls=":", lw=1, alpha=0.7,
                    label="Random (1/K)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Exact Match Rate")
        ax.set_title(f"Binary MaxRL, K={K}")
        ax.legend(loc="best")

    fig.tight_layout()
    save_figure(fig, output_dir, "fig11_exact_match_evolution")


# ---------------------------------------------------------------------------
# Figure 12: Advantage distributions (boxplots)
# ---------------------------------------------------------------------------

def plot_figure_12(all_metrics, output_dir):
    """Boxplots of advantage_abs_mean across RL methods at K=1000 G=64."""
    fig, ax = plt.subplots(figsize=(6, 4))
    K = 1000
    box_data = []
    labels = []

    for method in RL_METHODS:
        runs = group_by(all_metrics, method=method, K=K, G=64)
        if not runs:
            continue
        values = []
        for run in runs:
            v = get_best_epoch_metric(run["metrics"], "val/advantage_abs_mean",
                                      minimize=True)
            if not np.isnan(v):
                values.append(v)
        if values:
            box_data.append(values)
            labels.append(METHOD_LABELS.get(method, method))

    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        method_list = [m for m in RL_METHODS
                       if METHOD_LABELS.get(m, m) in labels]
        for patch, method in zip(bp["boxes"], method_list):
            patch.set_facecolor(COLORS.get(method, "gray"))
            patch.set_alpha(0.6)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_ylabel("|Advantage| (mean)")
    ax.set_title("Advantage Distributions (K=1000, G=64)")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig12_advantage_distributions")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

# Metrics to include in summary tables
SUMMARY_METRICS = [
    "val/mae_meters",
    "val/rmse_meters",
    "val/abs_rel",
    "val/delta_1.25",
    "val/ordinal_ce_objective",
    "val/exact_match_rate",
]


def write_summary_tables(all_metrics, output_dir):
    """Write 3 summary CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # --- summary_full.csv: one row per (method, K, G, seed) ---
    full_rows = []
    for _jid, run in all_metrics.items():
        row = {
            "job_id": run["job_id"],
            "method": run["method"],
            "K": run["K"],
            "G": run.get("G", ""),
            "seed": run.get("seed", ""),
        }
        for mk in SUMMARY_METRICS:
            row[mk] = get_best_epoch_metric(run["metrics"], mk, minimize=True)
        full_rows.append(row)

    full_path = os.path.join(output_dir, "summary_full.csv")
    if full_rows:
        keys = list(full_rows[0].keys())
        with open(full_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(full_rows)
        print(f"  Wrote {full_path} ({len(full_rows)} rows)")
    else:
        print(f"  No data for summary_full.csv")

    # --- summary_seed_averaged.csv ---
    # Group by (method, K, G)
    groups = defaultdict(list)
    for row in full_rows:
        key = (row["method"], row["K"], row["G"])
        groups[key].append(row)

    avg_rows = []
    for (method, K, G), rows in sorted(groups.items()):
        avg_row = {"method": method, "K": K, "G": G, "n_seeds": len(rows)}
        for mk in SUMMARY_METRICS:
            vals = [r[mk] for r in rows if not np.isnan(r[mk])]
            if vals:
                avg_row[f"{mk}_mean"] = float(np.mean(vals))
                avg_row[f"{mk}_std"] = float(np.std(vals))
            else:
                avg_row[f"{mk}_mean"] = float("nan")
                avg_row[f"{mk}_std"] = float("nan")
        avg_rows.append(avg_row)

    avg_path = os.path.join(output_dir, "summary_seed_averaged.csv")
    if avg_rows:
        keys = list(avg_rows[0].keys())
        with open(avg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(avg_rows)
        print(f"  Wrote {avg_path} ({len(avg_rows)} rows)")
    else:
        print(f"  No data for summary_seed_averaged.csv")

    # --- paper_table_k_sweep.csv ---
    # Formatted: method, G, K10, K100, K1000 as "mean +/- std"
    paper_rows = []
    methods_for_paper = ALL_METHODS
    for method in methods_for_paper:
        # Determine G to use: supervised methods have no G
        if method in SUPERVISED_METHODS:
            g_val = ""
        else:
            g_val = "64"

        paper_row = {"method": METHOD_LABELS.get(method, method), "G": g_val}
        for K in K_VALUES:
            key = (method, str(K), str(g_val))
            matching = groups.get(key, [])
            if not matching:
                paper_row[f"K={K}"] = "-"
                continue
            # Use the averaged row
            avg_match = [r for r in avg_rows
                         if r["method"] == method
                         and str(r["K"]) == str(K)
                         and str(r["G"]) == str(g_val)]
            if avg_match:
                ar = avg_match[0]
                m = ar.get("val/mae_meters_mean", float("nan"))
                s = ar.get("val/mae_meters_std", float("nan"))
                if not np.isnan(m):
                    paper_row[f"K={K}"] = f"{m:.3f} +/- {s:.3f}"
                else:
                    paper_row[f"K={K}"] = "-"
            else:
                paper_row[f"K={K}"] = "-"
        paper_rows.append(paper_row)

    paper_path = os.path.join(output_dir, "paper_table_k_sweep.csv")
    if paper_rows:
        keys = ["method", "G"] + [f"K={K}" for K in K_VALUES]
        with open(paper_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(paper_rows)
        print(f"  Wrote {paper_path} ({len(paper_rows)} rows)")
    else:
        print(f"  No data for paper_table_k_sweep.csv")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 12 figures + 3 summary tables for experiment7_depth")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing <job_id>/metrics.json files")
    parser.add_argument("--output_dir", default="./experiment7_depth/figures",
                        help="Output directory for figures and tables")
    parser.add_argument("--manifest",
                        default="./experiment7_depth/scripts/runs/manifest.csv",
                        help="Path to manifest CSV")
    parser.add_argument("--figures", nargs="+", default=["all"],
                        help="Which figures to generate (1-12 or 'all')")
    args = parser.parse_args()

    setup_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = load_manifest(args.manifest)
    all_metrics = load_all_metrics(args.results_dir, manifest)
    print(f"Loaded {len(all_metrics)} runs from {len(manifest)} manifest entries")

    if args.figures == ["all"]:
        figs = list(range(1, 13))
    else:
        figs = [int(f) for f in args.figures]

    for n in figs:
        fn = globals().get(f"plot_figure_{n:02d}")
        if fn is None:
            print(f"  Figure {n:02d}: no function found, skipping")
            continue
        try:
            fn(all_metrics, args.output_dir)
            print(f"  Figure {n:02d} written")
        except Exception as e:
            print(f"  Figure {n:02d} FAILED: {e}")

    write_summary_tables(all_metrics, args.output_dir)
    print(f"All done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
