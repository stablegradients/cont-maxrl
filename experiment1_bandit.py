#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import betainc


@dataclass
class ExperimentConfig:
    num_prompts: int = 100
    num_arms: int = 10
    rollouts_per_prompt: int = 8
    training_steps: int = 2000
    log_every: int = 50
    learning_rate: float = 1e-2
    eval_best_of_n_trials: int = 100
    integration_points: int = 200
    threshold_binary: float = 0.5
    env_seed: int = 42
    eps: float = 1e-8


METHODS = [
    {"key": "reinforce", "label": "REINFORCE", "color": "gray"},
    {"key": "reinforce_baseline", "label": "REINFORCE+baseline", "color": "black"},
    {"key": "grpo", "label": "GRPO", "color": "tab:blue"},
    {"key": "continuous_maxrl", "label": "Continuous MaxRL", "color": "tab:red"},
    {"key": "continuous_maxrl_normalized", "label": "Continuous MaxRL (normalized)", "color": "tab:cyan"},
    {"key": "binary_maxrl_05", "label": "Binary MaxRL@0.5", "color": "orange"},
]


def tier_indices(num_prompts: int) -> Dict[str, np.ndarray]:
    return {
        "easy": np.arange(0, 33),
        "medium": np.arange(33, 66),
        "hard": np.arange(66, num_prompts),
    }


def build_environment(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.env_seed)
    alpha = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    beta = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    best_arms = np.zeros(cfg.num_prompts, dtype=np.int64)

    for i in range(cfg.num_prompts):
        if i < 33:
            best_base = (8.0, 2.0)
            other_base = (2.0, 5.0)
        elif i < 66:
            best_base = (4.0, 4.0)
            other_base = (2.0, 6.0)
        else:
            best_base = (1.0, 9.0)
            other_base = (1.0, 19.0)

        best_arm = int(rng.integers(0, cfg.num_arms))
        best_arms[i] = best_arm
        for a in range(cfg.num_arms):
            base_a, base_b = best_base if a == best_arm else other_base
            a_perturb = rng.uniform(-0.2, 0.2)
            b_perturb = rng.uniform(-0.2, 0.2)
            alpha[i, a] = max(0.05, base_a + a_perturb)
            beta[i, a] = max(0.05, base_b + b_perturb)

    mean_rewards = alpha / (alpha + beta)
    tau_grid = np.linspace(0.0, 1.0 - 1e-6, cfg.integration_points, dtype=np.float64)
    survival_grid = 1.0 - betainc(alpha[:, :, None], beta[:, :, None], tau_grid[None, None, :])
    survival_at_05 = 1.0 - betainc(alpha, beta, 0.5)

    return {
        "alpha": alpha,
        "beta": beta,
        "mean_rewards": mean_rewards,
        "best_arms": best_arms,
        "tau_grid": tau_grid,
        "survival_grid": survival_grid,
        "survival_at_05": survival_at_05,
    }


def summarize_metric(values: np.ndarray, tiers: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
    overall = float(values.mean())
    tier_means = {name: float(values[idx].mean()) for name, idx in tiers.items()}
    return overall, tier_means


def estimate_best_of_n_mc(
    probs: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_rollouts: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_prompts, num_arms = probs.shape
    estimates = np.zeros(num_prompts, dtype=np.float64)
    for i in range(num_prompts):
        actions = rng.choice(num_arms, size=(trials, n_rollouts), p=probs[i])
        alpha_sel = alpha[i, actions]
        beta_sel = beta[i, actions]
        rewards = rng.beta(alpha_sel, beta_sel)
        estimates[i] = rewards.max(axis=1).mean()
    return estimates


def evaluate_policy(
    theta: torch.Tensor,
    env: Dict[str, np.ndarray],
    tiers: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    rng_eval: np.random.Generator,
) -> Dict[str, object]:
    probs = torch.softmax(theta.detach(), dim=-1).cpu().numpy()

    expected_reward_prompt = (probs * env["mean_rewards"]).sum(axis=1)
    best_of_n_prompt = estimate_best_of_n_mc(
        probs=probs,
        alpha=env["alpha"],
        beta=env["beta"],
        n_rollouts=cfg.rollouts_per_prompt,
        trials=cfg.eval_best_of_n_trials,
        rng=rng_eval,
    )

    p_tau = np.einsum("pk,pkt->pt", probs, env["survival_grid"])
    p_tau = np.clip(p_tau, 1e-12, 1.0)
    integrated_log_survival_prompt = np.trapezoid(np.log(p_tau), env["tau_grid"], axis=1)

    pass_at_05_prompt = (probs * env["survival_at_05"]).sum(axis=1)
    entropy_prompt = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)

    out = {}
    for key, values in [
        ("expected_reward", expected_reward_prompt),
        ("best_of_8", best_of_n_prompt),
        ("integrated_log_survival", integrated_log_survival_prompt),
        ("pass_at_05", pass_at_05_prompt),
        ("entropy", entropy_prompt),
    ]:
        overall, by_tier = summarize_metric(values, tiers)
        out[key] = overall
        out[f"{key}_tiers"] = by_tier
    return out


def advantages_reinforce(rewards: torch.Tensor) -> torch.Tensor:
    return rewards


def advantages_reinforce_baseline(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - rewards.mean(dim=1, keepdim=True)


def advantages_grpo(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    adv = (rewards - mean) / (std + eps)
    zero_std_mask = std < eps
    if zero_std_mask.any():
        adv = torch.where(zero_std_mask, torch.zeros_like(adv), adv)
    return adv


def advantages_binary_maxrl_threshold(rewards: torch.Tensor, threshold: float) -> torch.Tensor:
    n = rewards.size(1)
    b = (rewards > threshold).to(rewards.dtype)
    k = b.sum(dim=1, keepdim=True)

    fail_adv = -torch.ones_like(rewards)
    success_adv = (n - k) / (k + 1e-12)
    adv = torch.where(b > 0, success_adv, fail_adv)

    no_success = k.squeeze(1) == 0
    if no_success.any():
        adv[no_success] = 0.0
    return adv


def advantages_continuous_maxrl(
    rewards: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_prompts, n = rewards.shape
    sorted_rewards, sort_idx = torch.sort(rewards, dim=1)

    prev = torch.cat(
        [torch.zeros((num_prompts, 1), dtype=rewards.dtype, device=rewards.device), sorted_rewards[:, :-1]],
        dim=1,
    )
    gaps = sorted_rewards - prev
    survivors = torch.arange(n, 0, -1, dtype=rewards.dtype, device=rewards.device).view(1, n)
    raw_sorted_weights = torch.cumsum(gaps / survivors, dim=1)

    # Scale by n so the 1/n in the policy gradient loss doesn't shrink the gradient
    raw_sorted_weights = raw_sorted_weights * n

    centered_sorted_weights = raw_sorted_weights - raw_sorted_weights.mean(dim=1, keepdim=True)
    if normalize:
        std = centered_sorted_weights.std(dim=1, keepdim=True, unbiased=False)
        centered_sorted_weights = centered_sorted_weights / (std + eps)
        zero_std_mask = std < eps
        if zero_std_mask.any():
            centered_sorted_weights = torch.where(zero_std_mask, torch.zeros_like(centered_sorted_weights), centered_sorted_weights)

    advantages = torch.zeros_like(rewards)
    advantages.scatter_(1, sort_idx, centered_sorted_weights)

    raw_sum = raw_sorted_weights.sum(dim=1)
    raw_sq_sum = (raw_sorted_weights**2).sum(dim=1)
    ess = torch.where(raw_sq_sum > 0.0, raw_sum**2 / raw_sq_sum, torch.zeros_like(raw_sum))
    top_weight_frac = torch.where(raw_sum > 0.0, raw_sorted_weights[:, -1] / raw_sum, torch.zeros_like(raw_sum))

    return advantages, ess, top_weight_frac


def training_step(
    method_key: str,
    theta: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    rng_train: np.random.Generator,
    tiers: Dict[str, np.ndarray],
) -> Dict[str, object]:
    probs = torch.softmax(theta, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample((cfg.rollouts_per_prompt,)).transpose(0, 1)

    log_probs = torch.log_softmax(theta, dim=-1).gather(1, actions)

    actions_np = actions.detach().cpu().numpy()
    alpha_sel = np.take_along_axis(env["alpha"], actions_np, axis=1)
    beta_sel = np.take_along_axis(env["beta"], actions_np, axis=1)
    rewards_np = rng_train.beta(alpha_sel, beta_sel)
    rewards = torch.from_numpy(rewards_np).to(dtype=theta.dtype, device=theta.device)

    diag = {
        "ess": None,
        "top_weight_frac": None,
        "ess_tiers": None,
        "top_weight_frac_tiers": None,
    }

    if method_key == "reinforce":
        advantages = advantages_reinforce(rewards)
    elif method_key == "reinforce_baseline":
        advantages = advantages_reinforce_baseline(rewards)
    elif method_key == "grpo":
        advantages = advantages_grpo(rewards, eps=cfg.eps)
    elif method_key == "binary_maxrl_05":
        advantages = advantages_binary_maxrl_threshold(rewards, threshold=cfg.threshold_binary)
    elif method_key in {"continuous_maxrl", "continuous_maxrl_normalized"}:
        advantages, ess, top_weight_frac = advantages_continuous_maxrl(
            rewards,
            normalize=(method_key == "continuous_maxrl_normalized"),
            eps=cfg.eps,
        )
        ess_np = ess.detach().cpu().numpy()
        top_np = top_weight_frac.detach().cpu().numpy()
        diag["ess"] = float(ess_np.mean())
        diag["top_weight_frac"] = float(top_np.mean())
        diag["ess_tiers"] = {name: float(ess_np[idx].mean()) for name, idx in tiers.items()}
        diag["top_weight_frac_tiers"] = {name: float(top_np[idx].mean()) for name, idx in tiers.items()}
    else:
        raise ValueError(f"Unknown method: {method_key}")

    loss = -(advantages * log_probs).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(theta.grad.norm().item())
    optimizer.step()

    return {"gradient_norm": grad_norm, "diagnostics": diag}


def run_method_seed(
    method_key: str,
    seed: int,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    tiers: Dict[str, np.ndarray],
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 10000)

    theta = torch.zeros((cfg.num_prompts, cfg.num_arms), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=cfg.learning_rate)

    history: List[Dict[str, object]] = []

    initial_eval = evaluate_policy(theta, env, tiers, cfg, rng_eval)
    initial_eval["step"] = 0
    initial_eval["gradient_norm"] = 0.0
    initial_eval["ess"] = None
    initial_eval["top_weight_frac"] = None
    initial_eval["ess_tiers"] = None
    initial_eval["top_weight_frac_tiers"] = None
    history.append(initial_eval)

    for step in range(1, cfg.training_steps + 1):
        step_out = training_step(
            method_key=method_key,
            theta=theta,
            optimizer=optimizer,
            env=env,
            cfg=cfg,
            rng_train=rng_train,
            tiers=tiers,
        )

        if step % cfg.log_every == 0:
            metrics = evaluate_policy(theta, env, tiers, cfg, rng_eval)
            metrics["step"] = step
            metrics["gradient_norm"] = step_out["gradient_norm"]
            metrics["ess"] = step_out["diagnostics"]["ess"]
            metrics["top_weight_frac"] = step_out["diagnostics"]["top_weight_frac"]
            metrics["ess_tiers"] = step_out["diagnostics"]["ess_tiers"]
            metrics["top_weight_frac_tiers"] = step_out["diagnostics"]["top_weight_frac_tiers"]
            history.append(metrics)

    final_policy = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    return {"history": history, "final_policy": final_policy}


def stack_series(
    method_runs: List[Dict[str, object]],
    key: str,
    tier: str = None,
    tier_key: str = None,
) -> np.ndarray:
    out = []
    for run in method_runs:
        row = []
        for entry in run["history"]:
            if tier is None:
                value = entry[key]
            else:
                tier_values = entry[tier_key]
                value = None if tier_values is None else tier_values[tier]
            if value is None:
                value = np.nan
            row.append(value)
        out.append(row)
    return np.asarray(out, dtype=np.float64)


def plot_with_shade(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: str,
) -> None:
    masked = np.ma.masked_invalid(y)
    mean = np.ma.mean(masked, axis=0).filled(np.nan)
    std = np.ma.std(masked, axis=0).filled(np.nan)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    lower = mean - std
    upper = mean + std
    ax.fill_between(x, lower, upper, where=~np.isnan(mean), color=color, alpha=0.2)


def make_figure_1(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metric_specs = [
        ("expected_reward", "Expected Reward"),
        ("best_of_8", "Best-of-8 Reward"),
        ("integrated_log_survival", "Integrated Log-Survival"),
        ("pass_at_05", "Pass@1 (r > 0.5)"),
    ]

    for ax, (metric_key, title) in zip(axes.flatten(), metric_specs):
        for method in METHODS:
            runs = all_results[method["key"]]
            steps = np.asarray([entry["step"] for entry in runs[0]["history"]], dtype=np.int64)
            series = stack_series(runs, metric_key)
            plot_with_shade(ax, steps, series, method["label"], method["color"])
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_2(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    tiers = ["easy", "medium", "hard"]
    row_specs = [
        ("expected_reward", "Expected Reward"),
        ("best_of_8", "Best-of-8 Reward"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for row, (metric_key, row_label) in enumerate(row_specs):
        for col, tier in enumerate(tiers):
            ax = axes[row, col]
            for method in METHODS:
                runs = all_results[method["key"]]
                steps = np.asarray([entry["step"] for entry in runs[0]["history"]], dtype=np.int64)
                series = stack_series(runs, key=None, tier=tier, tier_key=f"{metric_key}_tiers")
                plot_with_shade(ax, steps, series, method["label"], method["color"])
            ax.set_title(f"{tier.capitalize()} - {row_label}")
            ax.set_xlabel("Training step")
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_3(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    ax = axes[0]
    for method in METHODS:
        runs = all_results[method["key"]]
        steps = np.asarray([entry["step"] for entry in runs[0]["history"]], dtype=np.int64)
        series = stack_series(runs, "gradient_norm")
        plot_with_shade(ax, steps, series, method["label"], method["color"])
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Training step")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    cont_runs = all_results["continuous_maxrl"]
    tiers = ["easy", "medium", "hard"]
    tier_colors = {"easy": "tab:green", "medium": "tab:purple", "hard": "tab:brown"}

    ax = axes[1]
    for tier in tiers:
        steps = np.asarray([entry["step"] for entry in cont_runs[0]["history"]], dtype=np.int64)
        series = stack_series(cont_runs, key=None, tier=tier, tier_key="ess_tiers")
        plot_with_shade(ax, steps, series, f"{tier.capitalize()}", tier_colors[tier])
    ax.set_title("Continuous MaxRL ESS")
    ax.set_xlabel("Training step")
    ax.set_ylabel("ESS")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    for tier in tiers:
        steps = np.asarray([entry["step"] for entry in cont_runs[0]["history"]], dtype=np.int64)
        series = stack_series(cont_runs, key=None, tier=tier, tier_key="top_weight_frac_tiers")
        plot_with_shade(ax, steps, series, f"{tier.capitalize()}", tier_colors[tier])
    ax.set_title("Top Sample Weight Fraction")
    ax.set_xlabel("Training step")
    ax.set_ylabel("w_(N) / sum(w)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_4(
    all_results: Dict[str, List[Dict[str, object]]],
    env: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    prompts = [0, 40, 80]
    prompt_labels = ["Easy prompt #1", "Medium prompt #41", "Hard prompt #81"]
    arms = np.arange(env["alpha"].shape[1])

    # Use seed 42 final policy for each method.
    fig, axes = plt.subplots(3, len(METHODS), figsize=(4 * len(METHODS), 10), sharey=True)
    if len(METHODS) == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, method in enumerate(METHODS):
        final_policy = all_results[method["key"]][0]["final_policy"]
        for row, (prompt_idx, label) in enumerate(zip(prompts, prompt_labels)):
            ax = axes[row, col]
            probs = final_policy[prompt_idx]
            best_arm = int(env["best_arms"][prompt_idx])
            colors = ["tab:gray"] * len(arms)
            colors[best_arm] = "tab:red"
            ax.bar(arms, probs, color=colors, alpha=0.85)
            if row == 0:
                ax.set_title(method["label"])
            if col == 0:
                ax.set_ylabel(label)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks(arms)
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)
            ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Synthetic Bandit for Continuous MaxRL")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-trials", type=int, default=100)
    parser.add_argument("--integration-points", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-fig4", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        training_steps=args.steps,
        log_every=args.log_every,
        eval_best_of_n_trials=args.eval_trials,
        integration_points=args.integration_points,
        learning_rate=args.lr,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiers = tier_indices(cfg.num_prompts)
    env = build_environment(cfg)

    all_results: Dict[str, List[Dict[str, object]]] = {m["key"]: [] for m in METHODS}

    for method in METHODS:
        method_key = method["key"]
        print(f"\n=== Running method: {method['label']} ===")
        for seed in args.seeds:
            print(f"Seed {seed}...")
            run_out = run_method_seed(
                method_key=method_key,
                seed=seed,
                env=env,
                cfg=cfg,
                tiers=tiers,
            )
            all_results[method_key].append(run_out)

    metrics_payload = {
        "config": {
            "num_prompts": cfg.num_prompts,
            "num_arms": cfg.num_arms,
            "rollouts_per_prompt": cfg.rollouts_per_prompt,
            "training_steps": cfg.training_steps,
            "log_every": cfg.log_every,
            "learning_rate": cfg.learning_rate,
            "eval_best_of_n_trials": cfg.eval_best_of_n_trials,
            "integration_points": cfg.integration_points,
            "threshold_binary": cfg.threshold_binary,
            "seeds": args.seeds,
            "env_seed": cfg.env_seed,
        },
        "tiers": {k: v.tolist() for k, v in tiers.items()},
        "environment": {
            "alpha": env["alpha"],
            "beta": env["beta"],
            "best_arms": env["best_arms"],
        },
        "methods": all_results,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(metrics_payload), f, indent=2)

    make_figure_1(all_results, out_dir / "fig1_learning_curves.png")
    make_figure_2(all_results, out_dir / "fig2_per_tier.png")
    make_figure_3(all_results, out_dir / "fig3_diagnostics.png")
    if not args.skip_fig4:
        make_figure_4(all_results, env, out_dir / "fig4_final_policies.png")

    print("\nDone.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
