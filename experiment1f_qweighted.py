#!/usr/bin/env python3
"""
Experiment 1f: Q-Weighted Survival MaxRL on the bandit benchmark.

Tests whether top-heavy threshold weighting q(τ) in the gap-over-survivors
estimator can match inference softmax's tail performance while maintaining
better mean reward.

Methods:
  Baselines (5): GRPO, survival unnorm/norm, inference softmax α=0.1/0.05
  Power-law (6): QW β=1,2,4,8 (unnorm), QW β=2,4 (norm)
  Exponential (3): QW-exp α=0.2, 0.1, 0.05

Same environment as experiment1d/1e (hard bandit v2).
"""
import argparse
import os
import json
import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.special import betainc


# ── Environment constants ─────────────────────────────────────────────
TYPE_C, TYPE_A, TYPE_B = 0, 1, 2


@dataclass
class ExperimentConfig:
    num_prompts: int = 100
    num_arms: int = 100
    num_type_a: int = 5
    num_type_b: int = 3
    env_seed: int = 42
    perturbation: float = 0.15
    min_alpha_beta: float = 0.1

    a_alpha: float = 5.0
    a_beta: float = 5.0
    b_low_alpha: float = 1.0
    b_low_beta: float = 8.0
    b_high_alpha: float = 15.0
    b_high_beta: float = 1.0
    b_mix_prob_high: float = 0.10
    c_alpha: float = 1.0
    c_beta: float = 8.0

    n_values: Tuple[int, ...] = (32,)
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    training_steps: int = 3000
    log_every: int = 25
    learning_rate: float = 3e-3

    eval_best_trials: int = 300
    eval_best16_n: int = 16
    eval_best32_n: int = 32
    integration_points: int = 200
    pass_threshold: float = 0.5
    binary_threshold: float = 0.3
    eps: float = 1e-8

    mode: str = "fixed_steps"
    total_rollouts_budget: int = 128000


# ── Method registry ───────────────────────────────────────────────────

METHODS = [
    # Baselines
    {"key": "grpo",              "label": "GRPO",              "color": "#1f77b4", "ls": "-"},
    {"key": "survival_unnorm",   "label": "Survival (unnorm)", "color": "#d62728", "ls": "-"},
    {"key": "survival_norm",     "label": "Survival (norm)",   "color": "#d62728", "ls": "--"},
    {"key": "inf_sm_a010",       "label": "Inf-SM α=0.1",     "color": "#ff7f0e", "ls": "-"},
    {"key": "inf_sm_a005",       "label": "Inf-SM α=0.05",    "color": "#ff7f0e", "ls": "--"},
    # Q-weighted power-law (unnormalized)
    {"key": "qw_beta1",          "label": "QW β=1",           "color": "#2ca02c", "ls": "-"},
    {"key": "qw_beta2",          "label": "QW β=2",           "color": "#2ca02c", "ls": "--"},
    {"key": "qw_beta4",          "label": "QW β=4",           "color": "#196f19", "ls": "-"},
    {"key": "qw_beta8",          "label": "QW β=8",           "color": "#196f19", "ls": "--"},
    # Q-weighted power-law (normalized)
    {"key": "qw_beta2_norm",     "label": "QW β=2 (n)",       "color": "#98df8a", "ls": "--"},
    {"key": "qw_beta4_norm",     "label": "QW β=4 (n)",       "color": "#98df8a", "ls": "-."},
    # Q-weighted exponential (unnormalized)
    {"key": "qw_exp_a020",       "label": "QW-exp α=0.2",     "color": "#9467bd", "ls": "-"},
    {"key": "qw_exp_a010",       "label": "QW-exp α=0.1",     "color": "#9467bd", "ls": "--"},
    {"key": "qw_exp_a005",       "label": "QW-exp α=0.05",    "color": "#7b4fae", "ls": "-."},
]


# ── Environment ───────────────────────────────────────────────────────

def beta_survival(a, b, tau):
    return 1.0 - betainc(a, b, tau)


def integrate_trapezoid(y, x, axis):
    return np.trapezoid(y, x, axis=axis) if hasattr(np, "trapezoid") else np.trapz(y, x, axis=axis)


def build_environment(cfg):
    rng = np.random.default_rng(cfg.env_seed)
    P, A = cfg.num_prompts, cfg.num_arms
    alpha1 = np.zeros((P, A), dtype=np.float64)
    beta1  = np.zeros((P, A), dtype=np.float64)
    alpha2 = np.zeros((P, A), dtype=np.float64)
    beta2  = np.zeros((P, A), dtype=np.float64)
    mix_prob  = np.zeros((P, A), dtype=np.float64)
    dist_kind = np.zeros((P, A), dtype=np.int64)
    type_ids  = np.full((P, A), TYPE_C, dtype=np.int64)

    for i in range(P):
        perm = rng.permutation(A)
        type_ids[i, perm[:cfg.num_type_a]] = TYPE_A
        type_ids[i, perm[cfg.num_type_a:cfg.num_type_a + cfg.num_type_b]] = TYPE_B
        for a in range(A):
            t = type_ids[i, a]
            pert = lambda v: max(cfg.min_alpha_beta, v + rng.uniform(-cfg.perturbation, cfg.perturbation))
            if t == TYPE_A:
                alpha1[i, a], beta1[i, a] = pert(cfg.a_alpha), pert(cfg.a_beta)
                alpha2[i, a], beta2[i, a] = alpha1[i, a], beta1[i, a]
            elif t == TYPE_B:
                alpha1[i, a], beta1[i, a] = pert(cfg.b_low_alpha), pert(cfg.b_low_beta)
                alpha2[i, a], beta2[i, a] = pert(cfg.b_high_alpha), pert(cfg.b_high_beta)
                mix_prob[i, a] = cfg.b_mix_prob_high
                dist_kind[i, a] = 1
            else:
                alpha1[i, a], beta1[i, a] = pert(cfg.c_alpha), pert(cfg.c_beta)
                alpha2[i, a], beta2[i, a] = alpha1[i, a], beta1[i, a]

    tau_grid = np.linspace(0.0, 1.0 - 1e-6, cfg.integration_points, dtype=np.float64)
    surv1 = beta_survival(alpha1[:, :, None], beta1[:, :, None], tau_grid[None, None, :])
    surv2 = beta_survival(alpha2[:, :, None], beta2[:, :, None], tau_grid[None, None, :])
    survival_grid = np.where(dist_kind[:, :, None] == 1,
                             (1.0 - mix_prob[:, :, None]) * surv1 + mix_prob[:, :, None] * surv2, surv1)
    surv1_pass = beta_survival(alpha1, beta1, cfg.pass_threshold)
    surv2_pass = beta_survival(alpha2, beta2, cfg.pass_threshold)
    survival_pass = np.where(dist_kind == 1, (1 - mix_prob) * surv1_pass + mix_prob * surv2_pass, surv1_pass)
    m1 = alpha1 / (alpha1 + beta1)
    m2 = alpha2 / (alpha2 + beta2)
    mean_rewards = np.where(dist_kind == 1, (1 - mix_prob) * m1 + mix_prob * m2, m1)

    return {
        "alpha1": alpha1, "beta1": beta1, "alpha2": alpha2, "beta2": beta2,
        "mix_prob": mix_prob, "dist_kind": dist_kind, "type_ids": type_ids,
        "mask_a": (type_ids == TYPE_A).astype(np.float64),
        "mask_b": (type_ids == TYPE_B).astype(np.float64),
        "mask_c": (type_ids == TYPE_C).astype(np.float64),
        "tau_grid": tau_grid, "survival_grid": survival_grid,
        "survival_pass": survival_pass, "mean_rewards": mean_rewards,
    }


def sample_rewards_for_actions(env, prompt_idx, actions, rng):
    kind = env["dist_kind"][prompt_idx, actions]
    a1, b1 = env["alpha1"][prompt_idx, actions], env["beta1"][prompt_idx, actions]
    rewards1 = rng.beta(a1, b1)
    if not np.any(kind == 1):
        return rewards1
    a2, b2 = env["alpha2"][prompt_idx, actions], env["beta2"][prompt_idx, actions]
    p = env["mix_prob"][prompt_idx, actions]
    rewards2 = rng.beta(a2, b2)
    mix = np.where(rng.random(size=actions.shape) < p, rewards2, rewards1)
    return np.where(kind == 1, mix, rewards1)


# ── Evaluation ────────────────────────────────────────────────────────

def estimate_best_of_k_mc(probs, env, k, trials, rng):
    P, A = probs.shape
    out = np.zeros(P, dtype=np.float64)
    for i in range(P):
        acts = rng.choice(A, size=(trials, k), p=probs[i])
        out[i] = sample_rewards_for_actions(env, i, acts, rng).max(axis=1).mean()
    return out


def compute_qw_ils(p_tau, tau_grid, q_transform, beta=1.0, alpha=0.1):
    """Q-weighted ILS: ∫ q(τ) log P(r>τ) dτ per prompt."""
    log_p = np.log(np.clip(p_tau, 1e-12, 1.0))
    if q_transform == "power":
        q = (beta + 1.0) * tau_grid ** beta
    else:  # exp
        c = 1.0 / (alpha * (np.exp(1.0 / alpha) - 1.0))
        q = c * np.exp(tau_grid / alpha)
    return integrate_trapezoid(q[None, :] * log_p, tau_grid, axis=1)


def evaluate_policy(theta, env, cfg, train_n, rng_eval):
    probs = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    p_tau = np.einsum("pk,pkt->pt", probs, env["survival_grid"])
    p_tau = np.clip(p_tau, 1e-12, 1.0)

    result = {
        "expected_reward": float((probs * env["mean_rewards"]).sum(axis=1).mean()),
        "best_of_16": float(estimate_best_of_k_mc(probs, env, cfg.eval_best16_n, cfg.eval_best_trials, rng_eval).mean()),
        "best_of_32": float(estimate_best_of_k_mc(probs, env, cfg.eval_best32_n, cfg.eval_best_trials, rng_eval).mean()),
        "best_of_n":  float(estimate_best_of_k_mc(probs, env, train_n, cfg.eval_best_trials, rng_eval).mean()),
        "integrated_log_survival": float(integrate_trapezoid(np.log(p_tau), env["tau_grid"], axis=1).mean()),
        "pass_at_05": float((probs * env["survival_pass"]).sum(axis=1).mean()),
        "pi_type_a":  float((probs * env["mask_a"]).sum(axis=1).mean()),
        "pi_type_b":  float((probs * env["mask_b"]).sum(axis=1).mean()),
        "pi_type_c":  float((probs * env["mask_c"]).sum(axis=1).mean()),
    }

    # Q-weighted ILS for key configurations
    tau_grid = env["tau_grid"]
    for b in [1, 2, 4, 8]:
        result[f"qw_ils_beta{b}"] = float(compute_qw_ils(p_tau, tau_grid, "power", beta=float(b)).mean())
    for a in [0.2, 0.1, 0.05]:
        a_str = f"{a:.3f}".replace(".", "")
        result[f"qw_ils_exp_a{a_str}"] = float(compute_qw_ils(p_tau, tau_grid, "exp", alpha=a).mean())

    return result


# ── Advantage functions ───────────────────────────────────────────────

def advantages_grpo(rewards, eps):
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    out = (rewards - mean) / (std + eps)
    if (std < eps).any():
        out = torch.where(std < eps, torch.zeros_like(out), out)
    return out


def advantages_survival(rewards, eps, normalize):
    P, N = rewards.shape
    dev, dt = rewards.device, rewards.dtype
    sr, si = torch.sort(rewards, dim=1)
    prev = torch.cat([torch.zeros((P, 1), device=dev, dtype=dt), sr[:, :-1]], dim=1)
    gaps = sr - prev
    survivors = torch.arange(N, 0, -1, device=dev, dtype=dt).view(1, N)
    w = torch.cumsum(gaps / survivors, dim=1)
    # Scale by N so the 1/N in the policy gradient loss doesn't shrink the gradient
    w = w * N
    w = w - w.mean(dim=1, keepdim=True)
    if normalize:
        std = w.std(dim=1, keepdim=True, unbiased=False)
        w = w / (std + eps)
        if (std < eps).any():
            w = torch.where(std < eps, torch.zeros_like(w), w)
    adv = torch.zeros_like(rewards)
    adv.scatter_(1, si, w)
    return adv


def advantages_inference_softmax(rewards, alpha):
    logits = rewards / alpha
    logits = logits - logits.max(dim=1, keepdim=True).values
    weights = torch.exp(logits)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights - 1.0 / rewards.size(1)


def advantages_qw_maxrl(rewards, q_transform, beta=1.0, alpha=0.1, normalize=False, eps=1e-8):
    """
    Gap-over-survivors with non-uniform threshold weighting q(τ).
    Only change from uniform: gaps = Q(sorted) - Q(prev) instead of sorted - prev.
    Batched: rewards shape (num_prompts, N).
    """
    P, N = rewards.shape
    dev, dt = rewards.device, rewards.dtype
    sr, si = torch.sort(rewards, dim=1)
    prev = torch.cat([torch.zeros((P, 1), device=dev, dtype=dt), sr[:, :-1]], dim=1)

    if q_transform == "power":
        exp = beta + 1.0
        Q_sr = sr ** exp
        Q_prev = prev ** exp
    elif q_transform == "exp":
        denom = torch.exp(torch.tensor(1.0 / alpha, device=dev, dtype=dt)) - 1.0
        Q_sr = (torch.exp(sr / alpha) - 1.0) / denom
        Q_prev = (torch.exp(prev / alpha) - 1.0) / denom
    else:
        raise ValueError(f"Unknown q_transform: {q_transform}")

    gaps = Q_sr - Q_prev
    survivors = torch.arange(N, 0, -1, device=dev, dtype=dt).view(1, N)
    w = torch.cumsum(gaps / survivors, dim=1)
    # Scale by N so the 1/N in the policy gradient loss doesn't shrink the gradient
    w = w * N
    w = w - w.mean(dim=1, keepdim=True)

    if normalize:
        std = w.std(dim=1, keepdim=True, unbiased=False)
        w = w / (std + eps)
        if (std < eps).any():
            w = torch.where(std < eps, torch.zeros_like(w), w)

    adv = torch.zeros_like(rewards)
    adv.scatter_(1, si, w)
    return adv


# ── Advantage dispatch ────────────────────────────────────────────────

def compute_advantages(key, rewards, cfg):
    e = cfg.eps
    if key == "grpo":                return advantages_grpo(rewards, e)
    if key == "survival_unnorm":     return advantages_survival(rewards, e, normalize=False)
    if key == "survival_norm":       return advantages_survival(rewards, e, normalize=True)
    if key == "inf_sm_a010":         return advantages_inference_softmax(rewards, alpha=0.1)
    if key == "inf_sm_a005":         return advantages_inference_softmax(rewards, alpha=0.05)
    if key == "qw_beta1":            return advantages_qw_maxrl(rewards, "power", beta=1.0)
    if key == "qw_beta2":            return advantages_qw_maxrl(rewards, "power", beta=2.0)
    if key == "qw_beta4":            return advantages_qw_maxrl(rewards, "power", beta=4.0)
    if key == "qw_beta8":            return advantages_qw_maxrl(rewards, "power", beta=8.0)
    if key == "qw_beta2_norm":       return advantages_qw_maxrl(rewards, "power", beta=2.0, normalize=True, eps=e)
    if key == "qw_beta4_norm":       return advantages_qw_maxrl(rewards, "power", beta=4.0, normalize=True, eps=e)
    if key == "qw_exp_a020":         return advantages_qw_maxrl(rewards, "exp", alpha=0.2)
    if key == "qw_exp_a010":         return advantages_qw_maxrl(rewards, "exp", alpha=0.1)
    if key == "qw_exp_a005":         return advantages_qw_maxrl(rewards, "exp", alpha=0.05)
    raise ValueError(f"Unknown method: {key}")


# ── Training ──────────────────────────────────────────────────────────

def rollout_rewards(theta, env, n_rollouts, rng):
    probs = torch.softmax(theta, dim=-1)
    actions = torch.distributions.Categorical(probs=probs).sample((n_rollouts,)).transpose(0, 1)
    actions_np = actions.detach().cpu().numpy()
    rewards_np = np.zeros_like(actions_np, dtype=np.float64)
    for i in range(actions_np.shape[0]):
        rewards_np[i] = sample_rewards_for_actions(env, i, actions_np[i], rng)
    return actions, torch.from_numpy(rewards_np).to(dtype=theta.dtype, device=theta.device)


def training_step(key, theta, optimizer, env, cfg, n_rollouts, rng):
    actions, rewards = rollout_rewards(theta, env, n_rollouts, rng)
    log_probs = torch.log_softmax(theta, dim=-1).gather(1, actions)
    advantages = compute_advantages(key, rewards, cfg)
    loss = -(advantages * log_probs).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(theta.grad.norm().item())
    optimizer.step()
    return grad_norm


def steps_for_n(cfg, n):
    return cfg.training_steps if cfg.mode == "fixed_steps" else int(cfg.total_rollouts_budget // n)


def run_method_seed_n(key, seed, n, env, cfg, device, gpu_id=None):
    torch.manual_seed(seed)
    if device == "cuda":
        gpu_id = gpu_id or 0
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed_all(seed)
        dev = f"cuda:{gpu_id}"
    else:
        dev = "cpu"
    np.random.seed(seed)
    rng_train = np.random.default_rng(seed + 1000 * n)
    rng_eval  = np.random.default_rng(seed + 2000 * n)

    theta = torch.zeros((cfg.num_prompts, cfg.num_arms), dtype=torch.float32, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=cfg.learning_rate)
    total_steps = steps_for_n(cfg, n)
    history = []

    init = evaluate_policy(theta, env, cfg, train_n=n, rng_eval=rng_eval)
    init["step"], init["gradient_norm"] = 0, 0.0
    history.append(init)

    for step in range(1, total_steps + 1):
        gn = training_step(key, theta, optimizer, env, cfg, n, rng_train)
        if step % cfg.log_every == 0 or step == total_steps:
            m = evaluate_policy(theta, env, cfg, train_n=n, rng_eval=rng_eval)
            m["step"], m["gradient_norm"] = step, gn
            history.append(m)

    final = evaluate_policy(theta, env, cfg, train_n=n, rng_eval=rng_eval)
    return {"history": history, "final_metrics": final,
            "final_policy": torch.softmax(theta.detach(), dim=-1).cpu().numpy()}


def run_single_job(job):
    run = run_method_seed_n(job["method_key"], job["seed"], job["n_rollouts"],
                            job["env"], job["cfg"], job["device"], job.get("gpu_id"))
    return {"method_key": job["method_key"], "n_rollouts": job["n_rollouts"],
            "seed": job["seed"], "gpu_id": job.get("gpu_id"), "run": run}


# ── Plotting utilities ────────────────────────────────────────────────

def series_from_runs(runs, key):
    return np.asarray([[e.get(key, np.nan) for e in r["history"]] for r in runs], dtype=np.float64)


def plot_with_shade(ax, x, y, label, color, ls="-", lw=2.0):
    m = np.ma.masked_invalid(y)
    mean = np.ma.mean(m, axis=0).filled(np.nan)
    std = np.ma.std(m, axis=0).filled(np.nan)
    ax.plot(x, mean, label=label, color=color, linewidth=lw, linestyle=ls)
    ax.fill_between(x, mean - std, mean + std, where=~np.isnan(mean), color=color, alpha=0.12)


# ── Comparison table ──────────────────────────────────────────────────

def print_comparison_table(results, methods, n_fixed, out_path=None):
    """Print and optionally save the main comparison table sorted by best_of_32."""
    rows = []
    for m in methods:
        if m["key"] not in results or n_fixed not in results[m["key"]]:
            continue
        runs = results[m["key"]][n_fixed]
        vals = {k: np.mean([r["final_metrics"][k] for r in runs]) for k in
                ["expected_reward", "best_of_16", "best_of_32", "integrated_log_survival", "pi_type_b"]}
        stds = {k: np.std([r["final_metrics"][k] for r in runs]) for k in vals}
        rows.append((m["label"], m["key"], vals, stds))
    rows.sort(key=lambda x: -x[2]["best_of_32"])

    hdr = f"{'Method':<22s} | {'ExpRew':>8s} | {'Best-16':>8s} | {'Best-32':>8s} | {'ILS':>8s} | {'π(B)':>8s}"
    sep = "-" * len(hdr)
    lines = [f"\nComparison Table (N={n_fixed}, sorted by Best-of-32)", sep, hdr, sep]
    for label, key, v, s in rows:
        lines.append(f"{label:<22s} | {v['expected_reward']:>7.3f}± | {v['best_of_16']:>7.3f}± | "
                     f"{v['best_of_32']:>7.3f}± | {v['integrated_log_survival']:>7.3f}± | {v['pi_type_b']:>7.3f}±")
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    if out_path:
        out_path.write_text(text + "\n", encoding="utf-8")


# ── Figures ───────────────────────────────────────────────────────────

def make_figure_1_pareto(results, methods, cfg, n_fixed, out_path):
    """Scatter: expected_reward (x) vs best_of_32 (y). One point per method."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for m in methods:
        if m["key"] not in results or n_fixed not in results[m["key"]]:
            continue
        runs = results[m["key"]][n_fixed]
        er = [r["final_metrics"]["expected_reward"] for r in runs]
        b32 = [r["final_metrics"]["best_of_32"] for r in runs]
        ax.errorbar(np.mean(er), np.mean(b32), xerr=np.std(er), yerr=np.std(b32),
                    fmt="o", color=m["color"], markersize=8, capsize=4, linewidth=1.5)
        ax.annotate(m["label"], (np.mean(er), np.mean(b32)),
                    textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel("Expected reward (mean)")
    ax.set_ylabel("Best-of-32 reward")
    ax.set_title(f"Pareto Frontier: Mean vs Tail (N={n_fixed})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_2_learning(results, methods, cfg, n_fixed, out_path):
    """2×1 grid: best_of_32 vs steps | expected_reward vs steps."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    specs = [("best_of_32", "Best-of-32 reward"), ("expected_reward", "Expected reward")]
    for ax, (key, title) in zip(axes, specs):
        for m in methods:
            if m["key"] not in results or n_fixed not in results[m["key"]]:
                continue
            runs = results[m["key"]][n_fixed]
            steps = np.asarray([e["step"] for e in runs[0]["history"]], dtype=np.int64)
            y = series_from_runs(runs, key)
            plot_with_shade(ax, steps, y, m["label"], m["color"], ls=m["ls"])
        ax.set_title(f"{title} (N={n_fixed})")
        ax.set_ylabel("Metric value")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Training step")
    handles = [Line2D([0], [0], color=m["color"], lw=2, ls=m["ls"], label=m["label"]) for m in methods
               if m["key"] in results]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_3_sensitivity(results, methods, cfg, n_fixed, out_path):
    """β/α sensitivity: final best_of_32 vs hyperparameter value."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Power-law β
    ax = axes[0]
    betas = [1, 2, 4, 8]
    keys_unnorm = [f"qw_beta{b}" for b in betas]
    keys_norm   = [f"qw_beta{b}_norm" for b in [2, 4]]
    for metric, ylabel, marker in [("best_of_32", "Best-of-32", "o"), ("expected_reward", "Expected reward", "s")]:
        vals = []
        for k in keys_unnorm:
            if k in results and n_fixed in results[k]:
                vals.append(np.mean([r["final_metrics"][metric] for r in results[k][n_fixed]]))
            else:
                vals.append(np.nan)
        ax.plot(betas, vals, marker=marker, linewidth=2, label=f"{ylabel} (unnorm)")
    # Horizontal baselines
    for bm in [m for m in methods if m["key"] in ("survival_unnorm", "inf_sm_a005", "grpo")]:
        if bm["key"] in results and n_fixed in results[bm["key"]]:
            v = np.mean([r["final_metrics"]["best_of_32"] for r in results[bm["key"]][n_fixed]])
            ax.axhline(v, color=bm["color"], ls=":", lw=1.5, label=f"{bm['label']} (B32)")
    ax.set_xlabel("β")
    ax.set_title("Power-law QW sensitivity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel 2: Exponential α
    ax = axes[1]
    exp_alphas = [0.2, 0.1, 0.05]
    exp_keys = [f"qw_exp_a{a:.3f}".replace(".", "") for a in exp_alphas]
    for metric, ylabel, marker in [("best_of_32", "Best-of-32", "o"), ("expected_reward", "Expected reward", "s")]:
        vals = []
        for k in exp_keys:
            if k in results and n_fixed in results[k]:
                vals.append(np.mean([r["final_metrics"][metric] for r in results[k][n_fixed]]))
            else:
                vals.append(np.nan)
        ax.plot(exp_alphas, vals, marker=marker, linewidth=2, label=f"{ylabel}")
    for bm in [m for m in methods if m["key"] in ("survival_unnorm", "inf_sm_a005", "grpo")]:
        if bm["key"] in results and n_fixed in results[bm["key"]]:
            v = np.mean([r["final_metrics"]["best_of_32"] for r in results[bm["key"]][n_fixed]])
            ax.axhline(v, color=bm["color"], ls=":", lw=1.5, label=f"{bm['label']} (B32)")
    ax.set_xlabel("α (smaller = more aggressive)")
    ax.set_xscale("log")
    ax.set_xlim(max(exp_alphas) * 1.5, min(exp_alphas) * 0.7)  # reversed
    ax.set_title("Exponential QW sensitivity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_4_scaling(results, methods, cfg, out_path):
    """Scaling: best_of_32 and expected_reward vs N (if multiple N)."""
    if len(cfg.n_values) < 2:
        return
    n_arr = np.asarray(cfg.n_values, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric, title) in zip(axes, [("best_of_32", "Best-of-32"), ("expected_reward", "Expected reward")]):
        for m in methods:
            if m["key"] not in results:
                continue
            y_runs = []
            for n in cfg.n_values:
                if n not in results[m["key"]]:
                    y_runs.append([np.nan])
                    continue
                y_runs.append([r["final_metrics"][metric] for r in results[m["key"]][n]])
            y = np.asarray(y_runs, dtype=np.float64)
            mean, std = y.mean(axis=1), y.std(axis=1)
            ax.plot(n_arr, mean, color=m["color"], ls=m["ls"], lw=2, marker="o", markersize=4, label=m["label"])
            ax.fill_between(n_arr, mean - std, mean + std, color=m["color"], alpha=0.1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_arr, labels=[str(int(v)) for v in n_arr])
        ax.set_title(title)
        ax.set_xlabel("N")
        ax.grid(alpha=0.3)
    handles = [Line2D([0], [0], color=m["color"], lw=2, ls=m["ls"], label=m["label"]) for m in methods
               if m["key"] in results]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ── Sanity checks ─────────────────────────────────────────────────────

def run_sanity_checks():
    print("Running sanity checks...")

    # Test 1: β=0 matches uniform survival
    print("  Test 1: β=0 matches uniform... ", end="")
    r = torch.rand(4, 16)
    a_uniform = advantages_survival(r, 1e-8, normalize=False)
    a_qw0 = advantages_qw_maxrl(r, "power", beta=0.0)
    assert torch.allclose(a_uniform, a_qw0, atol=1e-5), f"Max diff: {(a_uniform - a_qw0).abs().max():.2e}"
    print("OK")

    # Test 2: Monotone in reward
    print("  Test 2: Monotone... ", end="")
    r = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
    for beta in [1.0, 2.0, 4.0, 8.0]:
        adv = advantages_qw_maxrl(r, "power", beta=beta)
        for i in range(4):
            assert adv[0, i] < adv[0, i + 1], f"Not monotone at β={beta}, i={i}"
    for alpha in [0.2, 0.1, 0.05]:
        adv = advantages_qw_maxrl(r, "exp", alpha=alpha)
        for i in range(4):
            assert adv[0, i] < adv[0, i + 1], f"Not monotone at exp α={alpha}, i={i}"
    print("OK")

    # Test 3: Mean-centered
    print("  Test 3: Mean-centered... ", end="")
    r = torch.rand(4, 16)
    for beta in [0.0, 1.0, 4.0]:
        adv = advantages_qw_maxrl(r, "power", beta=beta)
        assert adv.sum(dim=1).abs().max() < 1e-5, f"Not mean-zero at β={beta}"
    for alpha in [0.2, 0.1, 0.05]:
        adv = advantages_qw_maxrl(r, "exp", alpha=alpha)
        assert adv.sum(dim=1).abs().max() < 1e-5, f"Not mean-zero at exp α={alpha}"
    print("OK")

    # Test 4: Numerical stability (exp with small α)
    print("  Test 4: Numerical stability... ", end="")
    r = torch.tensor([[0.01, 0.5, 0.99, 0.3]])
    adv = advantages_qw_maxrl(r, "exp", alpha=0.05)
    assert not torch.isnan(adv).any(), "NaN at exp α=0.05"
    assert not torch.isinf(adv).any(), "Inf at exp α=0.05"
    print("OK")

    # Test 5: Top-heaviness — higher β gives top sample more advantage
    print("  Test 5: Top-heaviness... ", end="")
    r = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
    a0 = advantages_qw_maxrl(r, "power", beta=0.0)
    a4 = advantages_qw_maxrl(r, "power", beta=4.0)
    # Top sample should get larger advantage at higher β
    assert a4[0, 4] > a0[0, 4], f"β=4 top adv {a4[0,4]:.3f} not > β=0 top adv {a0[0,4]:.3f}"
    # Gap between top and second should widen
    gap0 = a0[0, 4] - a0[0, 3]
    gap4 = a4[0, 4] - a4[0, 3]
    assert gap4 > gap0, f"β=4 gap {gap4:.3f} not > β=0 gap {gap0:.3f}"
    print("OK")

    # Test 6: Exp matches power-law qualitatively (sign pattern)
    print("  Test 6: Exp ~ power qualitatively... ", end="")
    r = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
    a_exp = advantages_qw_maxrl(r, "exp", alpha=0.1)
    a_pow = advantages_qw_maxrl(r, "power", beta=4.0)
    assert (a_exp > 0).tolist() == (a_pow > 0).tolist(), "Sign pattern mismatch"
    print("OK")

    # Test 7: Worked example from spec
    print("  Test 7: Worked example... ", end="")
    r = torch.tensor([[0.3, 0.9, 0.1, 0.6]])
    adv = advantages_qw_maxrl(r, "power", beta=4.0)
    # r=0.9 should get highest advantage (~0.404), r=0.6 should be negative (~-0.109)
    assert adv[0, 1] > 0.35, f"r=0.9 advantage too low: {adv[0,1]:.4f}"
    assert adv[0, 3] < 0.0, f"r=0.6 should be negative: {adv[0,3]:.4f}"
    assert adv[0, 2] < adv[0, 0] < adv[0, 3] < adv[0, 1], "Wrong ordering"
    print("OK")

    print("All sanity checks passed.\n")


# ── Serialization ─────────────────────────────────────────────────────

def sanitize(obj):
    if isinstance(obj, dict):   return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.floating):
        v = float(obj); return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj


def build_metrics_summary(payload):
    results_raw = payload["results"]
    first_method = next(iter(results_raw))
    first_n = next(iter(results_raw[first_method]))
    metric_names = list(results_raw[first_method][first_n][0]["final_metrics"].keys())
    agg = {}
    for mk, by_n in results_raw.items():
        agg[mk] = {}
        for nk, runs in by_n.items():
            stats = {}
            for metric in metric_names:
                arr = np.asarray([float(r["final_metrics"].get(metric, 0.0)) for r in runs])
                stats[metric] = {"mean": float(arr.mean()), "std": float(arr.std())}
            agg[mk][str(nk)] = {"num_runs": len(runs), "metrics": stats}
    return {"aggregated": agg}


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment 1f: Q-Weighted Survival MaxRL")
    p.add_argument("--output-dir", type=str, default="results_1f")
    p.add_argument("--n-values", type=int, nargs="+", default=[32])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-trials", type=int, default=300)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--metrics-path", type=str, default=None)
    p.add_argument("--sanity-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sanity_only:
        run_sanity_checks()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        n_values=tuple(args.n_values), seeds=tuple(args.seeds),
        training_steps=args.steps, log_every=args.log_every,
        learning_rate=args.lr, eval_best_trials=args.eval_trials,
    )

    if args.plots_only:
        mp_ = Path(args.metrics_path) if args.metrics_path else out_dir / "metrics.json"
        with mp_.open() as f:
            payload = json.load(f)
        config = payload["config"]
        cfg = ExperimentConfig(
            n_values=tuple(int(x) for x in config["n_values"]),
            seeds=tuple(int(x) for x in config["seeds"]),
            training_steps=int(config["training_steps"]),
            log_every=int(config["log_every"]),
            learning_rate=float(config["learning_rate"]),
            eval_best_trials=int(config["eval_best_trials"]),
        )
        results = {mk: {int(n): runs for n, runs in by_n.items()}
                   for mk, by_n in payload["results"].items()}
        n0 = 32 if 32 in cfg.n_values else cfg.n_values[0]
        print_comparison_table(results, METHODS, n0, out_dir / "comparison_table.txt")
        make_figure_1_pareto(results, METHODS, cfg, n0, out_dir / "fig1_pareto.png")
        make_figure_2_learning(results, METHODS, cfg, n0, out_dir / "fig2_learning.png")
        make_figure_3_sensitivity(results, METHODS, cfg, n0, out_dir / "fig3_sensitivity.png")
        make_figure_4_scaling(results, METHODS, cfg, out_dir / "fig4_scaling.png")
        print(f"Plots saved to: {out_dir}")
        return

    # ── Run experiment ────────────────────────────────────────────────
    run_sanity_checks()
    env = build_environment(cfg)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("No GPU detected; falling back to CPU.")
        device = "cpu"

    gpu_ids = []
    if device == "cuda":
        ngpu = torch.cuda.device_count()
        gpu_ids = list(range(ngpu)) if args.gpu_ids is None else [g for g in args.gpu_ids if 0 <= g < ngpu]

    num_workers = args.num_workers if args.num_workers > 0 else (
        len(gpu_ids) if device == "cuda" else min(len(cfg.seeds), os.cpu_count() or 1))

    if device == "cuda":
        gpu_slots = []
        base, rem = num_workers // len(gpu_ids), num_workers % len(gpu_ids)
        for i, g in enumerate(gpu_ids):
            gpu_slots.extend([g] * (base + (1 if i < rem else 0)))
    else:
        gpu_slots = []

    print(f"Experiment 1f: Q-Weighted Survival MaxRL")
    print(f"  Methods: {len(METHODS)}, N: {list(cfg.n_values)}, Seeds: {list(cfg.seeds)}")
    print(f"  Device: {device}, Workers: {num_workers}")

    results_map = {m["key"]: {n: {} for n in cfg.n_values} for m in METHODS}
    jobs, gpu_rr = [], 0
    for n in cfg.n_values:
        for m in METHODS:
            for seed in cfg.seeds:
                gid = gpu_slots[gpu_rr % len(gpu_slots)] if gpu_slots else None
                gpu_rr += 1
                jobs.append({"method_key": m["key"], "n_rollouts": n, "seed": seed,
                             "env": env, "cfg": cfg, "device": device, "gpu_id": gid})
    total = len(jobs)
    print(f"  Total runs: {total}\n")

    if num_workers == 1:
        for idx, job in enumerate(jobs, 1):
            print(f"[{idx}/{total}] {job['method_key']} N={job['n_rollouts']} seed={job['seed']}")
            out = run_single_job(job)
            results_map[out["method_key"]][out["n_rollouts"]][out["seed"]] = out["run"]
    else:
        ctx = mp.get_context("spawn") if device == "cuda" else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(run_single_job, j) for j in jobs]
            for idx, f in enumerate(as_completed(futs), 1):
                out = f.result()
                results_map[out["method_key"]][out["n_rollouts"]][out["seed"]] = out["run"]
                print(f"[{idx}/{total}] done {out['method_key']} N={out['n_rollouts']} seed={out['seed']}")

    results = {m["key"]: {n: [results_map[m["key"]][n][s] for s in cfg.seeds] for n in cfg.n_values}
               for m in METHODS}

    payload = {
        "config": {"num_prompts": cfg.num_prompts, "num_arms": cfg.num_arms,
                    "n_values": list(cfg.n_values), "seeds": list(cfg.seeds),
                    "training_steps": cfg.training_steps, "log_every": cfg.log_every,
                    "learning_rate": cfg.learning_rate, "eval_best_trials": cfg.eval_best_trials,
                    "device": device, "num_workers": num_workers},
        "results": results,
    }
    mp_ = out_dir / "metrics.json"
    with mp_.open("w") as f:
        json.dump(sanitize(payload), f, indent=2)
    sp = out_dir / "metrics_summary.json"
    with sp.open("w") as f:
        json.dump(sanitize(build_metrics_summary(payload)), f, indent=2)

    n0 = 32 if 32 in cfg.n_values else cfg.n_values[0]
    print_comparison_table(results, METHODS, n0, out_dir / "comparison_table.txt")
    make_figure_1_pareto(results, METHODS, cfg, n0, out_dir / "fig1_pareto.png")
    make_figure_2_learning(results, METHODS, cfg, n0, out_dir / "fig2_learning.png")
    make_figure_3_sensitivity(results, METHODS, cfg, n0, out_dir / "fig3_sensitivity.png")
    make_figure_4_scaling(results, METHODS, cfg, out_dir / "fig4_scaling.png")

    print(f"\nDone. Metrics: {mp_}, Figures: {out_dir}")


if __name__ == "__main__":
    main()
