#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "grpo": {"label": "GRPO", "color": "tab:blue", "order": 0},
    "maxrl_unnorm": {"label": "MaxRL-unnorm", "color": "tab:red", "order": 1},
    "maxrl_norm": {"label": "MaxRL-norm", "color": "orange", "order": 2},
    "reinforce_baseline": {"label": "REINFORCE+bl", "color": "black", "order": 3},
}


def normalize_method_key(method: str) -> str:
    method = str(method)
    if method == "reinforce":
        return "reinforce_baseline"
    return method


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_problem_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_rows(path)


def _normalize_group_size(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _group_size_tag(group_size: int | None) -> str:
    return f"g{group_size}" if group_size is not None else "gNA"


def _method_order_key(method: str) -> tuple[int, str]:
    style = METHOD_STYLES.get(method)
    if style is None:
        return (100, method)
    return (int(style["order"]), method)


def aggregate_by_model_group_method(
    runs: list[dict[str, Any]],
) -> dict[str, dict[int | None, dict[str, dict[str, Any]]]]:
    grouped: dict[str, dict[int | None, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for row in runs:
        model = str(row.get("model", "unknown"))
        method = normalize_method_key(str(row.get("method", "unknown")))
        group_size = _normalize_group_size(row.get("group_size"))
        grouped[model][group_size][method].append(row)

    out: dict[str, dict[int | None, dict[str, dict[str, Any]]]] = {}
    for model, by_group in grouped.items():
        out[model] = {}
        for group_size, by_method in by_group.items():
            out[model][group_size] = {}
            for method, rows in by_method.items():
                k_keys = sorted(
                    {
                        str(k)
                        for row in rows
                        for k in dict(row.get("pass_at_k", {})).keys()
                    },
                    key=lambda x: int(x),
                )

                means: dict[str, float] = {}
                stds: dict[str, float] = {}
                for k in k_keys:
                    vals = np.array(
                        [
                            float(row["pass_at_k"][k])
                            for row in rows
                            if k in row.get("pass_at_k", {})
                        ],
                        dtype=float,
                    )
                    if vals.size == 0:
                        continue
                    means[k] = float(vals.mean())
                    stds[k] = float(vals.std(ddof=0))

                expected_reward_vals = np.array(
                    [
                        float(row["expected_reward"])
                        for row in rows
                        if row.get("expected_reward") is not None
                    ],
                    dtype=float,
                )
                expected_reward_mean = (
                    float(expected_reward_vals.mean()) if expected_reward_vals.size else None
                )

                best_of_n_vals = np.array(
                    [float(row["best_of_n"]) for row in rows if row.get("best_of_n") is not None],
                    dtype=float,
                )
                best_of_n_mean = float(best_of_n_vals.mean()) if best_of_n_vals.size else None

                out[model][group_size][method] = {
                    "num_seeds": len(rows),
                    "pass_at_k_mean": means,
                    "pass_at_k_std": stds,
                    "expected_reward_mean": expected_reward_mean,
                    "best_of_n_mean": best_of_n_mean,
                }
    return out


def find_crossover_k(grpo_curve: dict[str, float], other_curve: dict[str, float]) -> int | None:
    for k_str in sorted(grpo_curve.keys(), key=lambda x: int(x)):
        if k_str not in other_curve:
            continue
        if other_curve[k_str] > grpo_curve[k_str]:
            return int(k_str)
    return None


def _plot_pass_at_k_core(
    model: str,
    group_size: int | None,
    method_data: dict[str, dict[str, Any]],
    output_dir: Path,
    figure_prefix: str,
    focus_methods: set[str] | None = None,
) -> None:
    plt.figure(figsize=(7.2, 4.8))

    methods = sorted(method_data.keys(), key=_method_order_key)
    if focus_methods is not None:
        methods = [m for m in methods if m in focus_methods]
    if not methods:
        plt.close()
        return

    grpo_curve = method_data.get("grpo", {}).get("pass_at_k_mean", {})

    for method in methods:
        stats = method_data[method]
        means = stats["pass_at_k_mean"]
        stds = stats["pass_at_k_std"]
        k_vals = np.array(sorted([int(k) for k in means.keys()]), dtype=int)
        if k_vals.size == 0:
            continue

        y = np.array([means[str(k)] for k in k_vals], dtype=float)
        y_std = np.array([stds[str(k)] for k in k_vals], dtype=float)

        style = METHOD_STYLES.get(method, {"label": method, "color": None})
        label = str(style["label"])
        color = style["color"]

        plt.plot(k_vals, y * 100.0, marker="o", linewidth=2, label=label, color=color)
        if stats["num_seeds"] > 1:
            plt.fill_between(
                k_vals,
                np.clip((y - y_std) * 100.0, 0.0, 100.0),
                np.clip((y + y_std) * 100.0, 0.0, 100.0),
                alpha=0.15,
                color=color,
            )

        if method in {"maxrl_unnorm", "maxrl_norm"} and grpo_curve:
            crossover_k = find_crossover_k(grpo_curve, means)
            if crossover_k is not None and str(crossover_k) in means:
                cy = means[str(crossover_k)] * 100.0
                plt.scatter([crossover_k], [cy], marker="*", s=120, color=color, zorder=5)

    xticks = [1, 2, 5, 10, 20, 50, 100]
    plt.xscale("log")
    plt.xticks(xticks, [str(x) for x in xticks])
    plt.xlabel("k")
    plt.ylabel("pass@k (%)")

    title_group = f", G={group_size}" if group_size is not None else ""
    plt.title(f"{figure_prefix}: pass@k ({model}{title_group})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"{figure_prefix}_pass_at_k_{model.replace('/', '_')}_{_group_size_tag(group_size)}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pass_at_k(
    model: str,
    group_size: int | None,
    method_data: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    _plot_pass_at_k_core(
        model=model,
        group_size=group_size,
        method_data=method_data,
        output_dir=output_dir,
        figure_prefix="figA",
    )


def plot_maxrl_comparison(
    model: str,
    group_size: int | None,
    method_data: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    focus = {"grpo", "maxrl_unnorm", "maxrl_norm"}
    available = set(method_data.keys())
    if not focus.issubset(available):
        return
    _plot_pass_at_k_core(
        model=model,
        group_size=group_size,
        method_data=method_data,
        output_dir=output_dir,
        figure_prefix="figD",
        focus_methods=focus,
    )


def plot_reward_histograms(
    problem_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    grouped: dict[tuple[str, int | None], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in problem_rows:
        model = str(row.get("model", "unknown"))
        group_size = _normalize_group_size(row.get("group_size"))
        method = normalize_method_key(str(row.get("method", "unknown")))
        rewards = row.get("continuous_rewards")
        if rewards is None:
            continue

        try:
            values = [float(x) for x in rewards]
        except Exception:
            continue

        grouped[(model, group_size)][method].extend(values)

    bins = np.linspace(0.0, 1.0, 21)
    for (model, group_size), method_rewards in grouped.items():
        plt.figure(figsize=(7.2, 4.8))
        plotted = False
        for method in sorted(method_rewards.keys(), key=_method_order_key):
            values = np.array(method_rewards[method], dtype=float)
            if values.size == 0:
                continue
            style = METHOD_STYLES.get(method, {"label": method, "color": None})
            plt.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=str(style["label"]),
                color=style["color"],
            )
            plotted = True

        if not plotted:
            plt.close()
            continue

        title_group = f", G={group_size}" if group_size is not None else ""
        plt.title(f"figC: reward distribution ({model}{title_group})")
        plt.xlabel("Continuous reward")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        output_path = (
            output_dir
            / f"figC_reward_hist_{model.replace('/', '_')}_{_group_size_tag(group_size)}.png"
        )
        plt.savefig(output_path, dpi=200)
        plt.close()


def _normalize_pass_metric(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None

    if not np.isfinite(out):
        return None
    if out > 1.0 and out <= 100.0:
        out = out / 100.0
    return float(np.clip(out, 0.0, 1.0))


def _extract_pass_metric_from_mapping(mapping: dict[str, Any], k: int) -> float | None:
    k_str = str(int(k))
    direct_keys = (
        f"pass_at_{k_str}",
        f"pass@{k_str}",
        f"pass_{k_str}",
        f"pass{k_str}",
    )

    for key in direct_keys:
        if key in mapping:
            val = _normalize_pass_metric(mapping[key])
            if val is not None:
                return val

    pass_map = mapping.get("pass_at_k")
    if isinstance(pass_map, dict):
        if k_str in pass_map:
            val = _normalize_pass_metric(pass_map[k_str])
            if val is not None:
                return val
        if int(k_str) in pass_map:
            val = _normalize_pass_metric(pass_map[int(k_str)])
            if val is not None:
                return val

    return None


def _extract_pass_metric(row: dict[str, Any], k: int) -> float | None:
    direct = _extract_pass_metric_from_mapping(row, k)
    if direct is not None:
        return direct

    for key in ("metrics", "eval", "evaluation", "validation", "val", "test"):
        nested = row.get(key)
        if isinstance(nested, dict):
            val = _extract_pass_metric_from_mapping(nested, k)
            if val is not None:
                return val
    return None


def _extract_step(row: dict[str, Any]) -> int | None:
    for key in ("step", "global_step", "train_step", "iteration", "iter"):
        if key not in row:
            continue
        try:
            return int(row[key])
        except Exception:
            continue
    return None


def plot_training_curves(
    training_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    grouped: dict[
        tuple[str, int | None],
        dict[str, dict[int, dict[str, list[float]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pass1": [], "pass10": []})))

    for row in training_rows:
        step = _extract_step(row)
        if step is None:
            continue

        pass1 = _extract_pass_metric(row, 1)
        pass10 = _extract_pass_metric(row, 10)
        if pass1 is None and pass10 is None:
            continue

        model = str(row.get("model", "unknown"))
        method = normalize_method_key(str(row.get("method", "unknown")))
        group_size = _normalize_group_size(row.get("group_size"))
        bucket = grouped[(model, group_size)][method][step]
        if pass1 is not None:
            bucket["pass1"].append(pass1)
        if pass10 is not None:
            bucket["pass10"].append(pass10)

    for (model, group_size), by_method in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=False, sharey=False)
        plotted_any = False

        for method in sorted(by_method.keys(), key=_method_order_key):
            style = METHOD_STYLES.get(method, {"label": method, "color": None})
            label = str(style["label"])
            color = style["color"]

            step_data = by_method[method]
            steps_sorted = sorted(step_data.keys())

            xs1 = []
            y1 = []
            y1_std = []
            xs10 = []
            y10 = []
            y10_std = []

            for step in steps_sorted:
                vals1 = np.array(step_data[step]["pass1"], dtype=float)
                if vals1.size > 0:
                    xs1.append(step)
                    y1.append(float(vals1.mean()) * 100.0)
                    y1_std.append(float(vals1.std(ddof=0)) * 100.0)

                vals10 = np.array(step_data[step]["pass10"], dtype=float)
                if vals10.size > 0:
                    xs10.append(step)
                    y10.append(float(vals10.mean()) * 100.0)
                    y10_std.append(float(vals10.std(ddof=0)) * 100.0)

            if xs1:
                x = np.array(xs1, dtype=int)
                y = np.array(y1, dtype=float)
                y_std = np.array(y1_std, dtype=float)
                axes[0].plot(x, y, marker="o", linewidth=2, label=label, color=color)
                axes[0].fill_between(
                    x,
                    np.clip(y - y_std, 0.0, 100.0),
                    np.clip(y + y_std, 0.0, 100.0),
                    alpha=0.15,
                    color=color,
                )
                plotted_any = True

            if xs10:
                x = np.array(xs10, dtype=int)
                y = np.array(y10, dtype=float)
                y_std = np.array(y10_std, dtype=float)
                axes[1].plot(x, y, marker="o", linewidth=2, label=label, color=color)
                axes[1].fill_between(
                    x,
                    np.clip(y - y_std, 0.0, 100.0),
                    np.clip(y + y_std, 0.0, 100.0),
                    alpha=0.15,
                    color=color,
                )
                plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        title_group = f", G={group_size}" if group_size is not None else ""
        fig.suptitle(f"figB: training curves ({model}{title_group})")

        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("pass@1 (%)")
        axes[0].grid(True, alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("pass@10 (%)")
        axes[1].grid(True, alpha=0.25)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels)

        fig.tight_layout()
        output_path = (
            output_dir
            / f"figB_training_curves_{model.replace('/', '_')}_{_group_size_tag(group_size)}.png"
        )
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_scaling_pass10(
    agg: dict[str, dict[int | None, dict[str, dict[str, Any]]]],
    output_dir: Path,
) -> None:
    for model, by_group in agg.items():
        numeric_groups = sorted([g for g in by_group.keys() if g is not None])
        if len(numeric_groups) < 2:
            continue

        plt.figure(figsize=(7.2, 4.8))
        plotted = False

        all_methods = {
            method
            for group_size in numeric_groups
            for method in by_group[group_size].keys()
        }
        for method in sorted(all_methods, key=_method_order_key):
            xs = []
            ys = []
            for group_size in numeric_groups:
                stats = by_group[group_size].get(method)
                if stats is None:
                    continue
                pass10 = stats["pass_at_k_mean"].get("10")
                if pass10 is None:
                    continue
                xs.append(group_size)
                ys.append(float(pass10) * 100.0)

            if len(xs) < 2:
                continue

            style = METHOD_STYLES.get(method, {"label": method, "color": None})
            plt.plot(xs, ys, marker="o", linewidth=2, label=str(style["label"]), color=style["color"])
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xticks(numeric_groups, [str(x) for x in numeric_groups])
        plt.xlabel("Group size G")
        plt.ylabel("pass@10 (%)")
        plt.title(f"figE: scaling with G ({model})")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        output_path = output_dir / f"figE_scaling_pass10_{model.replace('/', '_')}.png"
        plt.savefig(output_path, dpi=200)
        plt.close()


def write_aggregate_table(
    agg: dict[str, dict[int | None, dict[str, dict[str, Any]]]],
    output_dir: Path,
) -> None:
    rows = []
    for model, by_group in agg.items():
        for group_size, method_data in by_group.items():
            grpo_curve = method_data.get("grpo", {}).get("pass_at_k_mean", {})

            for method, stats in method_data.items():
                crossover_k = None
                if method in {"maxrl_unnorm", "maxrl_norm"} and grpo_curve:
                    crossover_k = find_crossover_k(grpo_curve, stats["pass_at_k_mean"])

                rows.append(
                    {
                        "model": model,
                        "group_size": group_size,
                        "method": method,
                        "num_seeds": stats["num_seeds"],
                        "expected_reward_mean": stats["expected_reward_mean"],
                        "best_of_n_mean": stats["best_of_n_mean"],
                        "pass_at_k_mean": stats["pass_at_k_mean"],
                        "pass_at_k_std": stats["pass_at_k_std"],
                        "crossover_k_vs_grpo": crossover_k,
                    }
                )

    rows.sort(
        key=lambda row: (
            row["model"],
            row["group_size"] if row["group_size"] is not None else -1,
            _method_order_key(str(row["method"])),
        )
    )

    output_path = output_dir / "metrics_aggregated.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Experiment 4 figures (A/B/C/D/E) from evaluation outputs."
    )
    parser.add_argument("--metrics-summary", type=Path, required=True)
    parser.add_argument(
        "--problem-results-jsonl",
        type=Path,
        default=None,
        help="Optional problem-level JSONL containing continuous_rewards for reward histograms",
    )
    parser.add_argument(
        "--training-metrics-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL with per-step metrics. Supported fields include step/global_step "
            "and pass@1/pass@10 (or pass_at_k with keys 1 and 10)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.metrics_summary)
    runs = summary.get("runs", [])
    if not runs:
        raise ValueError("No runs found in metrics summary")

    agg = aggregate_by_model_group_method(runs)

    for model, by_group in agg.items():
        for group_size, method_data in by_group.items():
            plot_pass_at_k(
                model=model,
                group_size=group_size,
                method_data=method_data,
                output_dir=args.output_dir,
            )
            plot_maxrl_comparison(
                model=model,
                group_size=group_size,
                method_data=method_data,
                output_dir=args.output_dir,
            )

    if args.problem_results_jsonl is not None and args.problem_results_jsonl.exists():
        problem_rows = load_problem_rows(args.problem_results_jsonl)
        if problem_rows:
            plot_reward_histograms(problem_rows, output_dir=args.output_dir)

    if args.training_metrics_jsonl is not None and args.training_metrics_jsonl.exists():
        training_rows = load_jsonl_rows(args.training_metrics_jsonl)
        if training_rows:
            plot_training_curves(training_rows=training_rows, output_dir=args.output_dir)

    plot_scaling_pass10(agg, output_dir=args.output_dir)
    write_aggregate_table(agg, output_dir=args.output_dir)
    print(f"Generated figures and aggregate metrics in {args.output_dir}")


if __name__ == "__main__":
    main()
