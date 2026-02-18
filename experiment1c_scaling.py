#!/usr/bin/env python3
import argparse
import os
import json
import math
import re
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import betainc


TYPE_C = 0
TYPE_A = 1
TYPE_B = 2


@dataclass
class ExperimentConfig:
    num_prompts: int = 100
    num_arms: int = 50
    num_type_a: int = 5
    num_type_b: int = 5
    env_seed: int = 42
    perturbation: float = 0.3
    min_alpha_beta: float = 0.1

    # Same hard environment used in Experiment 1b (selected tradeoff candidate).
    a_alpha: float = 20.0
    a_beta: float = 5.0
    b_low_alpha: float = 1.0
    b_low_beta: float = 5.0
    b_high_alpha: float = 8.0
    b_high_beta: float = 1.0
    b_mix_prob_high: float = 0.30
    c_alpha: float = 2.0
    c_beta: float = 8.0

    n_values: Tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    training_steps: int = 1000
    log_every: int = 25
    learning_rate: float = 3e-3

    eval_best_trials: int = 200
    eval_best16_n: int = 16
    integration_points: int = 200
    pass_threshold: float = 0.8
    binary_threshold: float = 0.5
    eps: float = 1e-8

    # Optional Mode B support.
    mode: str = "fixed_steps"
    total_rollouts_budget: int = 128000


METHODS = [
    {"key": "grpo", "label": "GRPO", "color": "tab:blue"},
    {"key": "continuous_maxrl_normalized", "label": "Continuous MaxRL (normalized)", "color": "tab:red"},
    {"key": "continuous_maxrl_unnormalized", "label": "Cont. MaxRL (unnorm)", "color": "orange"},
    {"key": "binary_maxrl_05", "label": "Binary MaxRL@0.5", "color": "tab:purple"},
    {"key": "reinforce_baseline", "label": "REINFORCE+baseline", "color": "black"},
]


def beta_mean_var(alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = alpha / (alpha + beta)
    var = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    return mean, var


def beta_survival(alpha: np.ndarray, beta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return 1.0 - betainc(alpha, beta, tau)


def build_environment(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.env_seed)
    alpha1 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    beta1 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    alpha2 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    beta2 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    mix_prob = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    dist_kind = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.int64)  # 0=beta, 1=mixture
    type_ids = np.full((cfg.num_prompts, cfg.num_arms), TYPE_C, dtype=np.int64)

    for i in range(cfg.num_prompts):
        perm = rng.permutation(cfg.num_arms)
        a_idx = perm[: cfg.num_type_a]
        b_idx = perm[cfg.num_type_a : cfg.num_type_a + cfg.num_type_b]
        c_idx = perm[cfg.num_type_a + cfg.num_type_b :]
        type_ids[i, a_idx] = TYPE_A
        type_ids[i, b_idx] = TYPE_B
        type_ids[i, c_idx] = TYPE_C

        for a in range(cfg.num_arms):
            t = type_ids[i, a]
            if t == TYPE_A:
                alpha1[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.a_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                beta1[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.a_beta + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                alpha2[i, a] = alpha1[i, a]
                beta2[i, a] = beta1[i, a]
                mix_prob[i, a] = 0.0
                dist_kind[i, a] = 0
            elif t == TYPE_B:
                alpha1[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.b_low_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                beta1[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.b_low_beta + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                alpha2[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.b_high_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                beta2[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.b_high_beta + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                mix_prob[i, a] = cfg.b_mix_prob_high
                dist_kind[i, a] = 1
            else:
                alpha1[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.c_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                beta1[i, a] = max(
                    cfg.min_alpha_beta,
                    cfg.c_beta + rng.uniform(-cfg.perturbation, cfg.perturbation),
                )
                alpha2[i, a] = alpha1[i, a]
                beta2[i, a] = beta1[i, a]
                mix_prob[i, a] = 0.0
                dist_kind[i, a] = 0

    tau_grid = np.linspace(0.0, 1.0 - 1e-6, cfg.integration_points, dtype=np.float64)
    surv1 = beta_survival(alpha1[:, :, None], beta1[:, :, None], tau_grid[None, None, :])
    surv2 = beta_survival(alpha2[:, :, None], beta2[:, :, None], tau_grid[None, None, :])
    survival_grid = np.where(
        dist_kind[:, :, None] == 1,
        (1.0 - mix_prob[:, :, None]) * surv1 + mix_prob[:, :, None] * surv2,
        surv1,
    )
    surv1_pass = beta_survival(alpha1, beta1, cfg.pass_threshold)
    surv2_pass = beta_survival(alpha2, beta2, cfg.pass_threshold)
    survival_pass = np.where(
        dist_kind == 1, (1.0 - mix_prob) * surv1_pass + mix_prob * surv2_pass, surv1_pass
    )
    mean1, _ = beta_mean_var(alpha1, beta1)
    mean2, _ = beta_mean_var(alpha2, beta2)
    mean_rewards = np.where(dist_kind == 1, (1.0 - mix_prob) * mean1 + mix_prob * mean2, mean1)

    mask_a = (type_ids == TYPE_A).astype(np.float64)
    mask_b = (type_ids == TYPE_B).astype(np.float64)
    mask_c = (type_ids == TYPE_C).astype(np.float64)

    return {
        "alpha1": alpha1,
        "beta1": beta1,
        "alpha2": alpha2,
        "beta2": beta2,
        "mix_prob": mix_prob,
        "dist_kind": dist_kind,
        "type_ids": type_ids,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "mask_c": mask_c,
        "tau_grid": tau_grid,
        "survival_grid": survival_grid,
        "survival_pass": survival_pass,
        "mean_rewards": mean_rewards,
    }


def sample_rewards_for_actions(
    env: Dict[str, np.ndarray],
    prompt_idx: int,
    actions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    kind = env["dist_kind"][prompt_idx, actions]
    a1 = env["alpha1"][prompt_idx, actions]
    b1 = env["beta1"][prompt_idx, actions]
    rewards1 = rng.beta(a1, b1)
    if not np.any(kind == 1):
        return rewards1

    a2 = env["alpha2"][prompt_idx, actions]
    b2 = env["beta2"][prompt_idx, actions]
    p = env["mix_prob"][prompt_idx, actions]
    high_component = rng.random(size=actions.shape) < p
    rewards2 = rng.beta(a2, b2)
    mix_rewards = np.where(high_component, rewards2, rewards1)
    return np.where(kind == 1, mix_rewards, rewards1)


def estimate_best_of_k_mc(
    probs: np.ndarray,
    env: Dict[str, np.ndarray],
    k: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_prompts, num_arms = probs.shape
    out = np.zeros(num_prompts, dtype=np.float64)
    for i in range(num_prompts):
        actions = rng.choice(num_arms, size=(trials, k), p=probs[i])
        rewards = sample_rewards_for_actions(env, i, actions, rng)
        out[i] = rewards.max(axis=1).mean()
    return out


def evaluate_policy(
    theta: torch.Tensor,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    train_n: int,
    rng_eval: np.random.Generator,
) -> Dict[str, float]:
    probs = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    p_tau = np.einsum("pk,pkt->pt", probs, env["survival_grid"])
    p_tau = np.clip(p_tau, 1e-12, 1.0)

    expected_reward_prompt = (probs * env["mean_rewards"]).sum(axis=1)
    best16_prompt = estimate_best_of_k_mc(
        probs=probs,
        env=env,
        k=cfg.eval_best16_n,
        trials=cfg.eval_best_trials,
        rng=rng_eval,
    )
    bestn_prompt = estimate_best_of_k_mc(
        probs=probs,
        env=env,
        k=train_n,
        trials=cfg.eval_best_trials,
        rng=rng_eval,
    )
    ils_prompt = np.trapezoid(np.log(p_tau), env["tau_grid"], axis=1)
    pass_prompt = (probs * env["survival_pass"]).sum(axis=1)
    pi_a_prompt = (probs * env["mask_a"]).sum(axis=1)
    pi_b_prompt = (probs * env["mask_b"]).sum(axis=1)
    pi_c_prompt = (probs * env["mask_c"]).sum(axis=1)

    return {
        "expected_reward": float(expected_reward_prompt.mean()),
        "best_of_16": float(best16_prompt.mean()),
        "best_of_n": float(bestn_prompt.mean()),
        "integrated_log_survival": float(ils_prompt.mean()),
        "pass_at_08": float(pass_prompt.mean()),
        "pi_type_a": float(pi_a_prompt.mean()),
        "pi_type_b": float(pi_b_prompt.mean()),
        "pi_type_c": float(pi_c_prompt.mean()),
    }


def advantages_reinforce_baseline(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - rewards.mean(dim=1, keepdim=True)


def advantages_grpo(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    out = (rewards - mean) / (std + eps)
    zero_std = std < eps
    if zero_std.any():
        out = torch.where(zero_std, torch.zeros_like(out), out)
    return out


def advantages_binary_maxrl(rewards: torch.Tensor, threshold: float) -> torch.Tensor:
    n = rewards.size(1)
    b = (rewards > threshold).to(rewards.dtype)
    k = b.sum(dim=1, keepdim=True)
    fail_adv = -torch.ones_like(rewards) / float(n)
    success_adv = (n - k) / (k * float(n) + 1e-12)
    out = torch.where(b > 0, success_adv, fail_adv)
    no_success = k.squeeze(1) == 0
    if no_success.any():
        out[no_success] = 0.0
    return out


def advantages_continuous_maxrl_normalized(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    num_prompts, n = rewards.shape
    sorted_rewards, sort_idx = torch.sort(rewards, dim=1)
    prev = torch.cat(
        [torch.zeros((num_prompts, 1), dtype=rewards.dtype, device=rewards.device), sorted_rewards[:, :-1]],
        dim=1,
    )
    gaps = sorted_rewards - prev
    survivors = torch.arange(n, 0, -1, dtype=rewards.dtype, device=rewards.device).view(1, n)
    sorted_weights = torch.cumsum(gaps / survivors, dim=1)
    centered = sorted_weights - sorted_weights.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, keepdim=True, unbiased=False)
    normalized = centered / (std + eps)
    zero_std = std < eps
    if zero_std.any():
        normalized = torch.where(zero_std, torch.zeros_like(normalized), normalized)

    adv = torch.zeros_like(rewards)
    adv.scatter_(1, sort_idx, normalized)
    return adv


def advantages_continuous_maxrl_unnormalized(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    num_prompts, n = rewards.shape
    sorted_rewards, sort_idx = torch.sort(rewards, dim=1)
    prev = torch.cat(
        [torch.zeros((num_prompts, 1), dtype=rewards.dtype, device=rewards.device), sorted_rewards[:, :-1]],
        dim=1,
    )
    gaps = sorted_rewards - prev
    survivors = torch.arange(n, 0, -1, dtype=rewards.dtype, device=rewards.device).view(1, n)
    sorted_weights = torch.cumsum(gaps / survivors, dim=1)
    centered = sorted_weights - sorted_weights.mean(dim=1, keepdim=True)

    adv = torch.zeros_like(rewards)
    adv.scatter_(1, sort_idx, centered)
    return adv


def compute_advantages(method_key: str, rewards: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    if method_key == "reinforce_baseline":
        return advantages_reinforce_baseline(rewards)
    if method_key == "grpo":
        return advantages_grpo(rewards, eps=cfg.eps)
    if method_key == "continuous_maxrl_normalized":
        return advantages_continuous_maxrl_normalized(rewards, eps=cfg.eps)
    if method_key == "continuous_maxrl_unnormalized":
        return advantages_continuous_maxrl_unnormalized(rewards, eps=cfg.eps)
    if method_key == "binary_maxrl_05":
        return advantages_binary_maxrl(rewards, threshold=cfg.binary_threshold)
    raise ValueError(f"Unknown method: {method_key}")


def rollout_rewards(
    theta: torch.Tensor,
    env: Dict[str, np.ndarray],
    n_rollouts: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(theta, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample((n_rollouts,)).transpose(0, 1)

    actions_np = actions.detach().cpu().numpy()
    rewards_np = np.zeros((actions_np.shape[0], actions_np.shape[1]), dtype=np.float64)
    for i in range(actions_np.shape[0]):
        rewards_np[i] = sample_rewards_for_actions(env, i, actions_np[i], rng)
    rewards = torch.from_numpy(rewards_np).to(dtype=theta.dtype, device=theta.device)
    return actions, rewards


def training_step(
    method_key: str,
    theta: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    n_rollouts: int,
    rng: np.random.Generator,
) -> float:
    actions, rewards = rollout_rewards(theta, env, n_rollouts, rng)
    log_probs = torch.log_softmax(theta, dim=-1).gather(1, actions)
    advantages = compute_advantages(method_key, rewards, cfg)

    loss = -(advantages * log_probs).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(theta.grad.norm().item())
    optimizer.step()
    return grad_norm


def steps_for_n(cfg: ExperimentConfig, n_rollouts: int) -> int:
    if cfg.mode == "fixed_steps":
        return cfg.training_steps
    return int(cfg.total_rollouts_budget // n_rollouts)


def run_method_seed_n(
    method_key: str,
    seed: int,
    n_rollouts: int,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    device: str,
    gpu_id: int = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    if device == "cuda":
        if gpu_id is None:
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed_all(seed)
        device_name = f"cuda:{gpu_id}"
    else:
        device_name = "cpu"
    np.random.seed(seed)
    rng_train = np.random.default_rng(seed + 1000 * n_rollouts)
    rng_eval = np.random.default_rng(seed + 2000 * n_rollouts)

    theta = torch.zeros(
        (cfg.num_prompts, cfg.num_arms), dtype=torch.float32, device=device_name, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=cfg.learning_rate)
    total_steps = steps_for_n(cfg, n_rollouts)
    history: List[Dict[str, float]] = []

    initial = evaluate_policy(theta, env, cfg, train_n=n_rollouts, rng_eval=rng_eval)
    initial["step"] = 0
    initial["gradient_norm"] = 0.0
    history.append(initial)

    for step in range(1, total_steps + 1):
        grad_norm = training_step(
            method_key=method_key,
            theta=theta,
            optimizer=optimizer,
            env=env,
            cfg=cfg,
            n_rollouts=n_rollouts,
            rng=rng_train,
        )
        if step % cfg.log_every == 0 or step == total_steps:
            metrics = evaluate_policy(theta, env, cfg, train_n=n_rollouts, rng_eval=rng_eval)
            metrics["step"] = step
            metrics["gradient_norm"] = grad_norm
            history.append(metrics)

    final_metrics = evaluate_policy(theta, env, cfg, train_n=n_rollouts, rng_eval=rng_eval)
    final_policy = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    return {
        "history": history,
        "final_metrics": final_metrics,
        "final_policy": final_policy,
    }


def run_single_job(job: Dict[str, object]) -> Dict[str, object]:
    run = run_method_seed_n(
        method_key=job["method_key"],
        seed=job["seed"],
        n_rollouts=job["n_rollouts"],
        env=job["env"],
        cfg=job["cfg"],
        device=job["device"],
        gpu_id=job.get("gpu_id", None),
    )
    return {
        "method_key": job["method_key"],
        "n_rollouts": job["n_rollouts"],
        "seed": job["seed"],
        "gpu_id": job.get("gpu_id", None),
        "run": run,
    }


def series_from_runs(runs: List[Dict[str, object]], key: str) -> np.ndarray:
    out = []
    for run in runs:
        out.append([entry.get(key, np.nan) for entry in run["history"]])
    return np.asarray(out, dtype=np.float64)


def plot_with_shade(ax, x, y, label, color):
    masked = np.ma.masked_invalid(y)
    mean = np.ma.mean(masked, axis=0).filled(np.nan)
    std = np.ma.std(masked, axis=0).filled(np.nan)
    ax.plot(x, mean, label=label, color=color, linewidth=2.0)
    ax.fill_between(x, mean - std, mean + std, where=~np.isnan(mean), color=color, alpha=0.2)


def method_legend_handles() -> List[Line2D]:
    return [Line2D([0], [0], color=m["color"], lw=2, label=m["label"]) for m in METHODS]


def make_figure_1_scaling(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    cfg: ExperimentConfig,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [
        ("expected_reward", "Expected reward"),
        ("best_of_16", "Best-of-16 reward"),
        ("best_of_n", "Best-of-N reward"),
        ("integrated_log_survival", "Integrated log-survival"),
        ("pass_at_08", "Pass@1 @ 0.8"),
        ("pi_type_b", "pi(Type B)"),
    ]
    n_values = np.asarray(cfg.n_values, dtype=np.float64)

    for ax, (metric_key, title) in zip(axes.flatten(), specs):
        for method in METHODS:
            y_runs = []
            for n in cfg.n_values:
                vals = [run["final_metrics"][metric_key] for run in results[method["key"]][n]]
                y_runs.append(vals)
            y = np.asarray(y_runs, dtype=np.float64)  # [num_n, num_seeds]
            mean = y.mean(axis=1)
            std = y.std(axis=1)
            ax.plot(n_values, mean, color=method["color"], linewidth=2.0, label=method["label"], marker="o")
            ax.fill_between(n_values, mean - std, mean + std, color=method["color"], alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values, labels=[str(int(v)) for v in n_values])
        ax.set_title(title)
        ax.set_xlabel("Training rollouts N")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    fig.legend(
        handles=method_legend_handles(),
        loc="upper center",
        ncol=min(len(METHODS), 3),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_2_learning(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    cfg: ExperimentConfig,
    out_path: Path,
) -> None:
    selected_n = sorted(int(n) for n in cfg.n_values)
    num_cols = len(selected_n)
    fig_width = max(16.0, 4.0 * num_cols)
    fig, axes = plt.subplots(2, num_cols, figsize=(fig_width, 9), sharex=False)
    if num_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    row_specs = [("expected_reward", "Expected reward"), ("best_of_16", "Best-of-16 reward")]

    for row, (metric_key, row_title) in enumerate(row_specs):
        for col, n in enumerate(selected_n):
            ax = axes[row, col]
            for method in METHODS:
                runs = results[method["key"]][n]
                steps = np.asarray([entry["step"] for entry in runs[0]["history"]], dtype=np.int64)
                y = series_from_runs(runs, metric_key)
                plot_with_shade(ax, steps, y, method["label"], method["color"])
            ax.set_title(f"N={n} - {row_title}")
            ax.set_xlabel("Training step")
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    fig.legend(
        handles=method_legend_handles(),
        loc="upper center",
        ncol=min(len(METHODS), 5),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def collect_advantage_batches(
    env: Dict[str, np.ndarray],
    policy_probs: np.ndarray,
    n_rollouts: int,
    prompt_idx: int,
    num_batches: int,
    seed: int,
    cfg: ExperimentConfig,
) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num_batches):
        actions = rng.choice(cfg.num_arms, size=n_rollouts, p=policy_probs[prompt_idx])
        rewards_np = sample_rewards_for_actions(env, prompt_idx, actions, rng)
        rewards = torch.tensor(rewards_np, dtype=torch.float32).unsqueeze(0)
        grpo_adv = advantages_grpo(rewards, eps=cfg.eps).squeeze(0).cpu().numpy()
        cmax_adv = advantages_continuous_maxrl_normalized(rewards, eps=cfg.eps).squeeze(0).cpu().numpy()
        out.append({"rewards": rewards_np, "grpo": grpo_adv, "cmax": cmax_adv})
    return out


def make_figure_3_advantages(
    batches: List[Dict[str, np.ndarray]],
    n_ref: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for b in batches:
        ax.hist(b["grpo"], bins=30, alpha=0.15, color="tab:blue", density=True)
    ax.set_title(f"GRPO Advantages (N={n_ref}, 10 batches)")
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for b in batches:
        ax.hist(b["cmax"], bins=30, alpha=0.15, color="tab:red", density=True)
    ax.set_title(f"Continuous MaxRL Advantages (N={n_ref}, 10 batches)")
    ax.set_xlabel("Advantage")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for b in batches:
        sc = ax.scatter(b["grpo"], b["cmax"], c=b["rewards"], cmap="viridis", alpha=0.5, s=12)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Reward")
    ax.set_title("GRPO vs Continuous MaxRL Advantages")
    ax.set_xlabel("GRPO advantage")
    ax.set_ylabel("Continuous MaxRL advantage")
    ax.grid(alpha=0.3)

    fig.legend(
        handles=[
            Patch(facecolor="tab:blue", alpha=0.35, label="GRPO advantage histogram"),
            Patch(facecolor="tab:red", alpha=0.35, label="Continuous MaxRL histogram"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                color="tab:green",
                label="Scatter (color = reward)",
            ),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_4_weight_nonlinearity(batch: Dict[str, np.ndarray], n_ref: int, out_path: Path) -> None:
    rewards = batch["rewards"]
    order = np.argsort(rewards)
    grpo_sorted = batch["grpo"][order]
    cmax_sorted = batch["cmax"][order]
    ranks = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranks, grpo_sorted, color="tab:blue", linewidth=2.0, label="GRPO")
    ax.plot(ranks, cmax_sorted, color="tab:red", linewidth=2.0, label="Continuous MaxRL (normalized)")
    ax.set_title(f"Advantage Weight vs Reward Rank (N={n_ref}, one batch)")
    ax.set_xlabel("Reward rank (1=lowest, N=highest)")
    ax.set_ylabel("Normalized advantage")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def build_metrics_summary(payload: Dict[str, object]) -> Dict[str, object]:
    results_raw = payload["results"]

    first_method = next(iter(results_raw))
    first_n = next(iter(results_raw[first_method]))
    first_run = results_raw[first_method][first_n][0]
    metric_names = list(first_run["final_metrics"].keys())

    aggregated: Dict[str, Dict[str, Dict[str, object]]] = {}
    for method_key, by_n in results_raw.items():
        aggregated[method_key] = {}
        for n_key, runs in by_n.items():
            metric_stats = {}
            for metric in metric_names:
                vals = [float(run["final_metrics"][metric]) for run in runs]
                arr = np.asarray(vals, dtype=np.float64)
                metric_stats[metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            aggregated[method_key][str(n_key)] = {
                "num_runs": len(runs),
                "metrics": metric_stats,
            }

    all_n = sorted({int(n) for by_n in aggregated.values() for n in by_n.keys()})
    best_by_n: Dict[str, Dict[str, Dict[str, object]]] = {}
    for n in all_n:
        n_key = str(n)
        best_by_n[n_key] = {}
        for metric in metric_names:
            candidates = []
            for method_key, by_n in aggregated.items():
                if n_key in by_n:
                    candidates.append((method_key, by_n[n_key]["metrics"][metric]["mean"]))
            if candidates:
                winner = max(candidates, key=lambda x: x[1])
                best_by_n[n_key][metric] = {"method": winner[0], "value": float(winner[1])}

    method_average_across_n: Dict[str, Dict[str, float]] = {}
    for method_key, by_n in aggregated.items():
        method_average_across_n[method_key] = {}
        sorted_n_keys = sorted(by_n.keys(), key=lambda x: int(x))
        for metric in metric_names:
            vals = [by_n[n_key]["metrics"][metric]["mean"] for n_key in sorted_n_keys]
            method_average_across_n[method_key][metric] = float(np.mean(vals))

    config = payload.get("config", {})
    config_subset = {
        "n_values": config.get("n_values"),
        "seeds": config.get("seeds"),
        "methods": config.get("methods"),
        "training_steps": config.get("training_steps"),
        "mode": config.get("mode"),
        "device": config.get("device"),
    }

    return {
        "config_subset": config_subset,
        "aggregated_final_by_method_and_n": aggregated,
        "best_method_by_n_and_metric": best_by_n,
        "method_average_across_n": method_average_across_n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1c: Scaling with rollouts N")
    parser.add_argument("--output-dir", type=str, default="results_1c")
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-trials", type=int, default=200)
    parser.add_argument("--integration-points", type=int, default=200)
    parser.add_argument("--mode", type=str, choices=["fixed_steps", "fixed_rollouts"], default="fixed_steps")
    parser.add_argument("--total-rollouts-budget", type=int, default=128000)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--metrics-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        metrics_path = Path(args.metrics_path) if args.metrics_path else out_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics file not found: {metrics_path}")

        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        summary_payload = build_metrics_summary(payload)
        summary_path = out_dir / "metrics_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(sanitize(summary_payload), f, indent=2)

        config = payload["config"]
        cfg = ExperimentConfig(
            n_values=tuple(int(x) for x in config.get("n_values", args.n_values)),
            seeds=tuple(int(x) for x in config.get("seeds", args.seeds)),
            training_steps=int(config.get("training_steps", args.steps)),
            log_every=int(config.get("log_every", args.log_every)),
            learning_rate=float(config.get("learning_rate", args.lr)),
            eval_best_trials=int(config.get("eval_best_trials", args.eval_trials)),
            integration_points=int(config.get("integration_points", args.integration_points)),
            mode=str(config.get("mode", args.mode)),
            total_rollouts_budget=int(config.get("total_rollouts_budget", args.total_rollouts_budget)),
        )

        results_raw = payload["results"]
        results: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        for method_key, by_n in results_raw.items():
            results[method_key] = {int(n): runs for n, runs in by_n.items()}

        adv_key = next((k for k in payload.keys() if k.startswith("advantage_batches_n")), None)
        if adv_key is None:
            raise ValueError("No advantage batch key found in metrics payload.")
        m = re.match(r"advantage_batches_n(\d+)_prompt0", adv_key)
        n_ref = int(m.group(1)) if m else (128 if 128 in cfg.n_values else max(cfg.n_values))

        adv_batches_raw = payload[adv_key]
        adv_batches = [
            {
                "rewards": np.asarray(b["rewards"], dtype=np.float64),
                "grpo": np.asarray(b["grpo"], dtype=np.float64),
                "cmax": np.asarray(b["cmax"], dtype=np.float64),
            }
            for b in adv_batches_raw
        ]

        make_figure_1_scaling(results, cfg, out_dir / "fig1_scaling_curves.png")
        make_figure_2_learning(results, cfg, out_dir / "fig2_learning_curves_by_N.png")
        make_figure_3_advantages(adv_batches, n_ref, out_dir / "fig3_advantage_distributions.png")
        make_figure_4_weight_nonlinearity(adv_batches[0], n_ref, out_dir / "fig4_weight_nonlinearity.png")
        print("Plots regenerated from existing metrics.")
        print(f"Metrics source: {metrics_path}")
        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved figures to: {out_dir}")
        return

    cfg = ExperimentConfig(
        n_values=tuple(args.n_values),
        seeds=tuple(args.seeds),
        training_steps=args.steps,
        log_every=args.log_every,
        learning_rate=args.lr,
        eval_best_trials=args.eval_trials,
        integration_points=args.integration_points,
        mode=args.mode,
        total_rollouts_budget=args.total_rollouts_budget,
    )
    env = build_environment(cfg)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but no GPU was detected; falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    selected_gpu_ids: List[int] = []
    if device == "cuda":
        visible_gpu_count = torch.cuda.device_count()
        visible_gpu_ids = list(range(visible_gpu_count))
        if args.gpu_ids is None:
            selected_gpu_ids = visible_gpu_ids
        else:
            selected_gpu_ids = [int(g) for g in args.gpu_ids]
            invalid = [g for g in selected_gpu_ids if g < 0 or g >= visible_gpu_count]
            if invalid:
                raise ValueError(
                    f"Invalid --gpu-ids {invalid}. Visible GPU ids are 0..{visible_gpu_count - 1}."
                )
        if not selected_gpu_ids:
            raise ValueError("No GPUs selected. Provide at least one id via --gpu-ids.")

    if args.num_workers <= 0:
        if device == "cuda":
            num_workers = len(selected_gpu_ids)
        else:
            num_workers = min(len(cfg.seeds), os.cpu_count() or 1)
    else:
        num_workers = max(1, args.num_workers)

    print("Environment fixed to Experiment 1b hard setup:")
    print("  Type A = Beta(20,5)")
    print("  Type B = 0.7*Beta(1,5) + 0.3*Beta(8,1)")
    print("  Type C = Beta(2,8)")
    print(f"  prompts={cfg.num_prompts}, arms={cfg.num_arms}, env_seed={cfg.env_seed}")
    print(f"Mode: {cfg.mode}")
    print(f"Device: {device}")
    if device == "cuda":
        active_gpu_ids = selected_gpu_ids
        gpu_slots: List[int] = []
        base = num_workers // len(active_gpu_ids)
        rem = num_workers % len(active_gpu_ids)
        for i, g in enumerate(active_gpu_ids):
            slot_count = base + (1 if i < rem else 0)
            gpu_slots.extend([g] * slot_count)
        if not gpu_slots:
            gpu_slots = [active_gpu_ids[0]]
        print(f"Visible GPUs: {selected_gpu_ids}")
        print(f"Using GPUs: {active_gpu_ids}")
        print(
            "Worker slots per GPU: "
            + ", ".join(
                f"{g}:{gpu_slots.count(g)}" for g in active_gpu_ids
            )
        )
    else:
        active_gpu_ids = []
        gpu_slots = []
    print(f"Workers: {num_workers}")

    results_map: Dict[str, Dict[int, Dict[int, Dict[str, object]]]] = {
        m["key"]: {n: {} for n in cfg.n_values} for m in METHODS
    }

    representative_policies: Dict[str, np.ndarray] = {}
    n_ref = 128 if 128 in cfg.n_values else max(cfg.n_values)

    jobs: List[Dict[str, object]] = []
    gpu_rr = 0
    for n in cfg.n_values:
        steps_n = steps_for_n(cfg, n)
        print(f"N={n}: steps={steps_n}")
        for method in METHODS:
            for seed in cfg.seeds:
                gpu_id = None
                if device == "cuda":
                    gpu_id = gpu_slots[gpu_rr % len(gpu_slots)]
                    gpu_rr += 1
                jobs.append(
                    {
                        "method_key": method["key"],
                        "n_rollouts": n,
                        "seed": seed,
                        "env": env,
                        "cfg": cfg,
                        "device": device,
                        "gpu_id": gpu_id,
                    }
                )
    total_jobs = len(jobs)
    print(f"Total runs: {total_jobs}")

    if num_workers == 1:
        for idx, job in enumerate(jobs, start=1):
            print(
                f"[{idx}/{total_jobs}] method={job['method_key']} N={job['n_rollouts']} seed={job['seed']} gpu={job.get('gpu_id')}"
            )
            job_out = run_single_job(job)
            m = job_out["method_key"]
            n = job_out["n_rollouts"]
            s = job_out["seed"]
            run = job_out["run"]
            results_map[m][n][s] = run
            if n == n_ref and s == 42 and m in {"grpo", "continuous_maxrl_normalized"}:
                representative_policies[m] = run["final_policy"]
    else:
        spawn_ctx = mp.get_context("spawn") if device == "cuda" else None
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=spawn_ctx,
        ) as executor:
            futures = [executor.submit(run_single_job, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                job_out = future.result()
                m = job_out["method_key"]
                n = job_out["n_rollouts"]
                s = job_out["seed"]
                run = job_out["run"]
                results_map[m][n][s] = run
                if n == n_ref and s == 42 and m in {"grpo", "continuous_maxrl_normalized"}:
                    representative_policies[m] = run["final_policy"]
                if device == "cuda":
                    print(f"[{idx}/{total_jobs}] done method={m} N={n} seed={s} gpu={job_out.get('gpu_id')}")
                else:
                    print(f"[{idx}/{total_jobs}] done method={m} N={n} seed={s}")

    results: Dict[str, Dict[int, List[Dict[str, object]]]] = {
        m["key"]: {n: [results_map[m["key"]][n][s] for s in cfg.seeds] for n in cfg.n_values}
        for m in METHODS
    }

    if "continuous_maxrl_normalized" not in representative_policies:
        representative_policies["continuous_maxrl_normalized"] = results["continuous_maxrl_normalized"][n_ref][0][
            "final_policy"
        ]
    if "grpo" not in representative_policies:
        representative_policies["grpo"] = results["grpo"][n_ref][0]["final_policy"]

    # Use Continuous MaxRL normalized final policy (seed 42 if available) to sample batches.
    adv_batches = collect_advantage_batches(
        env=env,
        policy_probs=representative_policies["continuous_maxrl_normalized"],
        n_rollouts=n_ref,
        prompt_idx=0,
        num_batches=10,
        seed=2026,
        cfg=cfg,
    )

    metrics_payload = {
        "config": {
            "num_prompts": cfg.num_prompts,
            "num_arms": cfg.num_arms,
            "num_type_a": cfg.num_type_a,
            "num_type_b": cfg.num_type_b,
            "num_type_c": cfg.num_arms - cfg.num_type_a - cfg.num_type_b,
            "env_seed": cfg.env_seed,
            "n_values": list(cfg.n_values),
            "seeds": list(cfg.seeds),
            "methods": [m["key"] for m in METHODS],
            "mode": cfg.mode,
            "training_steps": cfg.training_steps,
            "log_every": cfg.log_every,
            "learning_rate": cfg.learning_rate,
            "eval_best_trials": cfg.eval_best_trials,
            "eval_best16_n": cfg.eval_best16_n,
            "integration_points": cfg.integration_points,
            "total_rollouts_budget": cfg.total_rollouts_budget,
            "device": device,
            "num_workers": num_workers,
            "gpu_ids": active_gpu_ids if device == "cuda" else [],
        },
        "environment": {
            "alpha1": env["alpha1"],
            "beta1": env["beta1"],
            "alpha2": env["alpha2"],
            "beta2": env["beta2"],
            "mix_prob": env["mix_prob"],
            "dist_kind": env["dist_kind"],
            "type_ids": env["type_ids"],
        },
        "results": results,
        f"advantage_batches_n{n_ref}_prompt0": adv_batches,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(metrics_payload), f, indent=2)
    summary_payload = build_metrics_summary(metrics_payload)
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(summary_payload), f, indent=2)

    make_figure_1_scaling(results, cfg, out_dir / "fig1_scaling_curves.png")
    make_figure_2_learning(results, cfg, out_dir / "fig2_learning_curves_by_N.png")
    make_figure_3_advantages(adv_batches, n_ref, out_dir / "fig3_advantage_distributions.png")
    make_figure_4_weight_nonlinearity(adv_batches[0], n_ref, out_dir / "fig4_weight_nonlinearity.png")

    print("\nDone.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
