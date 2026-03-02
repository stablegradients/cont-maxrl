#!/usr/bin/env python3
"""
Experiment 1e: Inference MaxRL (Boltzmann optimality) on the bandit benchmark.

Adds two inference-based MaxRL estimators to the existing bandit experiment:
  - Softmax (deterministic, biased): softmax-weighted advantages
  - Bernoulli (stochastic, unbiased): Bernoulli-sampled optimality indicators

Based on experiment1d_hard_bandit.py. Same environment, same training loop,
same existing methods; only adds new advantage functions, metrics, and plots.
"""
import argparse
import os
import json
import math
import re
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import betainc


TYPE_C = 0
TYPE_A = 1
TYPE_B = 2

ALPHA_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]


@dataclass
class ExperimentConfig:
    num_prompts: int = 100
    num_arms: int = 100
    num_type_a: int = 5
    num_type_b: int = 3
    env_seed: int = 42
    perturbation: float = 0.15
    min_alpha_beta: float = 0.1

    # Experiment 1d hard setup v2.
    a_alpha: float = 5.0
    a_beta: float = 5.0
    b_low_alpha: float = 1.0
    b_low_beta: float = 8.0
    b_high_alpha: float = 15.0
    b_high_beta: float = 1.0
    b_mix_prob_high: float = 0.10
    c_alpha: float = 1.0
    c_beta: float = 8.0

    n_values: Tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    training_steps: int = 3000
    log_every: int = 25
    learning_rate: float = 3e-3

    eval_best_trials: int = 300
    eval_best16_n: int = 16
    integration_points: int = 200
    pass_threshold: float = 0.5
    binary_threshold: float = 0.3
    eps: float = 1e-8

    # Optional Mode B support.
    mode: str = "fixed_steps"
    total_rollouts_budget: int = 128000

    # Inference MaxRL alpha values to sweep.
    inference_alphas: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _build_methods(cfg: ExperimentConfig) -> List[Dict]:
    """Build method list including inference MaxRL alpha variants."""
    base = [
        {"key": "grpo", "label": "GRPO", "color": "tab:blue", "group": "baseline"},
        {"key": "continuous_maxrl_normalized", "label": "Cont. MaxRL (norm)", "color": "tab:red", "group": "survival"},
        {"key": "continuous_maxrl_unnormalized", "label": "Cont. MaxRL (unnorm)", "color": "orange", "group": "survival"},
        {"key": "binary_maxrl_03", "label": "Binary MaxRL@0.3", "color": "tab:purple", "group": "survival"},
        {"key": "reinforce_baseline", "label": "REINFORCE+baseline", "color": "black", "group": "baseline"},
    ]

    alpha_colors_softmax = {
        0.05: "#1b9e77", 0.1: "#d95f02", 0.2: "#7570b3", 0.5: "#e7298a", 1.0: "#66a61e",
    }
    alpha_colors_bernoulli = {
        0.05: "#1b9e77", 0.1: "#d95f02", 0.2: "#7570b3", 0.5: "#e7298a", 1.0: "#66a61e",
    }

    for alpha in cfg.inference_alphas:
        a_str = f"{alpha:.2f}".replace(".", "")
        base.append({
            "key": f"inference_softmax_a{a_str}",
            "label": f"Inf-Softmax α={alpha}",
            "color": alpha_colors_softmax.get(alpha, "tab:green"),
            "group": "inference_softmax",
            "alpha": alpha,
        })
    for alpha in cfg.inference_alphas:
        a_str = f"{alpha:.2f}".replace(".", "")
        base.append({
            "key": f"inference_bernoulli_a{a_str}",
            "label": f"Inf-Bernoulli α={alpha}",
            "color": alpha_colors_bernoulli.get(alpha, "tab:olive"),
            "group": "inference_bernoulli",
            "alpha": alpha,
        })
    return base


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def beta_mean_var(alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = alpha / (alpha + beta)
    var = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    return mean, var


def beta_survival(alpha: np.ndarray, beta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return 1.0 - betainc(alpha, beta, tau)


def integrate_trapezoid(y: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def build_environment(cfg: ExperimentConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.env_seed)
    alpha1 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    beta1 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    alpha2 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    beta2 = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    mix_prob = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.float64)
    dist_kind = np.zeros((cfg.num_prompts, cfg.num_arms), dtype=np.int64)
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
                alpha1[i, a] = max(cfg.min_alpha_beta, cfg.a_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation))
                beta1[i, a] = max(cfg.min_alpha_beta, cfg.a_beta + rng.uniform(-cfg.perturbation, cfg.perturbation))
                alpha2[i, a] = alpha1[i, a]
                beta2[i, a] = beta1[i, a]
                mix_prob[i, a] = 0.0
                dist_kind[i, a] = 0
            elif t == TYPE_B:
                alpha1[i, a] = max(cfg.min_alpha_beta, cfg.b_low_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation))
                beta1[i, a] = max(cfg.min_alpha_beta, cfg.b_low_beta + rng.uniform(-cfg.perturbation, cfg.perturbation))
                alpha2[i, a] = max(cfg.min_alpha_beta, cfg.b_high_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation))
                beta2[i, a] = max(cfg.min_alpha_beta, cfg.b_high_beta + rng.uniform(-cfg.perturbation, cfg.perturbation))
                mix_prob[i, a] = cfg.b_mix_prob_high
                dist_kind[i, a] = 1
            else:
                alpha1[i, a] = max(cfg.min_alpha_beta, cfg.c_alpha + rng.uniform(-cfg.perturbation, cfg.perturbation))
                beta1[i, a] = max(cfg.min_alpha_beta, cfg.c_beta + rng.uniform(-cfg.perturbation, cfg.perturbation))
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
        "alpha1": alpha1, "beta1": beta1,
        "alpha2": alpha2, "beta2": beta2,
        "mix_prob": mix_prob, "dist_kind": dist_kind, "type_ids": type_ids,
        "mask_a": mask_a, "mask_b": mask_b, "mask_c": mask_c,
        "tau_grid": tau_grid, "survival_grid": survival_grid,
        "survival_pass": survival_pass, "mean_rewards": mean_rewards,
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


def print_environment_verification(cfg: ExperimentConfig, env: Dict[str, np.ndarray]) -> None:
    a_mean = cfg.a_alpha / (cfg.a_alpha + cfg.a_beta)
    b_low_mean = cfg.b_low_alpha / (cfg.b_low_alpha + cfg.b_low_beta)
    b_high_mean = cfg.b_high_alpha / (cfg.b_high_alpha + cfg.b_high_beta)
    b_mean = (1.0 - cfg.b_mix_prob_high) * b_low_mean + cfg.b_mix_prob_high * b_high_mean
    c_mean = cfg.c_alpha / (cfg.c_alpha + cfg.c_beta)

    mask_a = env["mask_a"] > 0
    mask_b = env["mask_b"] > 0
    mask_c = env["mask_c"] > 0
    mean_rewards = env["mean_rewards"]

    print("Environment verification (Experiment 1d hard setup v2):")
    print(f"  Type A: {cfg.num_type_a} arms, analytic mean={a_mean:.3f}, observed={float(mean_rewards[mask_a].mean()):.3f}")
    print(f"  Type B: {cfg.num_type_b} arms, analytic mean={b_mean:.3f}, observed={float(mean_rewards[mask_b].mean()):.3f}")
    print(f"  Type C: {cfg.num_arms - cfg.num_type_a - cfg.num_type_b} arms, analytic mean={c_mean:.3f}, observed={float(mean_rewards[mask_c].mean()):.3f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def estimate_best_of_k_mc(
    probs: np.ndarray, env: Dict[str, np.ndarray], k: int, trials: int, rng: np.random.Generator,
) -> np.ndarray:
    num_prompts, num_arms = probs.shape
    out = np.zeros(num_prompts, dtype=np.float64)
    for i in range(num_prompts):
        actions = rng.choice(num_arms, size=(trials, k), p=probs[i])
        rewards = sample_rewards_for_actions(env, i, actions, rng)
        out[i] = rewards.max(axis=1).mean()
    return out


def compute_log_soft_pass_rate_mc(
    policy_probs: np.ndarray,
    env: Dict[str, np.ndarray],
    alpha: float = 0.2,
    n_mc: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Monte Carlo estimation of log(p_alpha) = log E_pi[exp((r-1)/alpha)] per prompt.
    Much faster than numerical integration: O(num_prompts * n_mc).
    """
    if rng is None:
        rng = np.random.default_rng(9999)
    num_prompts, num_arms = policy_probs.shape
    log_soft_pass = np.zeros(num_prompts, dtype=np.float64)

    for i in range(num_prompts):
        arms = rng.choice(num_arms, size=n_mc, p=policy_probs[i])
        rewards = sample_rewards_for_actions(env, i, arms, rng)
        sigma = np.exp((rewards - 1.0) / alpha)
        p_alpha = sigma.mean()
        log_soft_pass[i] = np.log(max(p_alpha, 1e-30))
    return log_soft_pass


def evaluate_policy(
    theta: torch.Tensor,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    train_n: int,
    rng_eval: np.random.Generator,
    eval_log_soft_pass: bool = True,
) -> Dict[str, float]:
    probs = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    p_tau = np.einsum("pk,pkt->pt", probs, env["survival_grid"])
    p_tau = np.clip(p_tau, 1e-12, 1.0)

    expected_reward_prompt = (probs * env["mean_rewards"]).sum(axis=1)
    best16_prompt = estimate_best_of_k_mc(probs=probs, env=env, k=cfg.eval_best16_n, trials=cfg.eval_best_trials, rng=rng_eval)
    bestn_prompt = estimate_best_of_k_mc(probs=probs, env=env, k=train_n, trials=cfg.eval_best_trials, rng=rng_eval)
    ils_prompt = integrate_trapezoid(np.log(p_tau), env["tau_grid"], axis=1)
    pass_prompt = (probs * env["survival_pass"]).sum(axis=1)
    pi_a_prompt = (probs * env["mask_a"]).sum(axis=1)
    pi_b_prompt = (probs * env["mask_b"]).sum(axis=1)
    pi_c_prompt = (probs * env["mask_c"]).sum(axis=1)

    result = {
        "expected_reward": float(expected_reward_prompt.mean()),
        "best_of_16": float(best16_prompt.mean()),
        "best_of_n": float(bestn_prompt.mean()),
        "integrated_log_survival": float(ils_prompt.mean()),
        "pass_at_05": float(pass_prompt.mean()),
        "pi_type_a": float(pi_a_prompt.mean()),
        "pi_type_b": float(pi_b_prompt.mean()),
        "pi_type_c": float(pi_c_prompt.mean()),
    }

    if eval_log_soft_pass:
        rng_lsp = np.random.default_rng(9999)
        for alpha in cfg.inference_alphas:
            a_str = f"{alpha:.2f}".replace(".", "")
            lsp = compute_log_soft_pass_rate_mc(probs, env, alpha=alpha, n_mc=1000, rng=rng_lsp)
            result[f"log_soft_pass_a{a_str}"] = float(lsp.mean())

    return result


# ---------------------------------------------------------------------------
# Advantage functions — existing (from 1d)
# ---------------------------------------------------------------------------

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
    fail_adv = -torch.ones_like(rewards)
    success_adv = (n - k) / (k + 1e-12)
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
    # Scale by n so the 1/n in the policy gradient loss doesn't shrink the gradient
    sorted_weights = sorted_weights * n
    centered = sorted_weights - sorted_weights.mean(dim=1, keepdim=True)
    adv = torch.zeros_like(rewards)
    adv.scatter_(1, sort_idx, centered)
    return adv


# ---------------------------------------------------------------------------
# Advantage functions — inference MaxRL (NEW)
# ---------------------------------------------------------------------------

def advantages_inference_softmax(rewards: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Inference-based MaxRL using softmax (deterministic, biased) estimator.

    weight_i = exp(r_i / alpha) / sum_j exp(r_j / alpha)
    A_i = weight_i - 1/N  (mean-centered)

    Batched: rewards shape (num_prompts, N).
    """
    logits = rewards / alpha
    logits = logits - logits.max(dim=1, keepdim=True).values  # log-sum-exp stability
    exp_logits = torch.exp(logits)
    weights = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    N = rewards.size(1)
    return weights - 1.0 / N


def advantages_inference_softmax_normalized(rewards: torch.Tensor, alpha: float, eps: float = 1e-8) -> torch.Tensor:
    """Z-score normalized variant of the softmax inference estimator."""
    adv = advantages_inference_softmax(rewards, alpha=alpha)
    std = adv.std(dim=1, keepdim=True, unbiased=False)
    out = adv / (std + eps)
    zero_std = std < eps
    if zero_std.any():
        out = torch.where(zero_std, torch.zeros_like(out), out)
    return out


def advantages_inference_bernoulli(rewards: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Inference-based MaxRL using Bernoulli sampling (stochastic, unbiased) estimator.

    sigma_i = exp((r_i - 1) / alpha)
    o_i ~ Bernoulli(sigma_i)
    K = sum(o_i)
    if K > 0: A_i = o_i/K - 1/N
    else:     A_i = 0

    Batched: rewards shape (num_prompts, N).
    Returns advantages and diagnostic dict.
    """
    num_prompts, N = rewards.shape
    sigma = torch.exp((rewards - 1.0) / alpha)
    sigma = sigma.clamp(0.0, 1.0)  # numerical safety
    o = torch.bernoulli(sigma)
    K = o.sum(dim=1, keepdim=True)  # (num_prompts, 1)

    weights = o / K.clamp(min=1.0)  # avoid div-by-zero; K=0 handled below
    adv = weights - weights.mean(dim=1, keepdim=True)

    # Zero out advantages for prompts where K=0
    k_zero = (K.squeeze(1) == 0)
    if k_zero.any():
        adv[k_zero] = 0.0

    return adv


def _bernoulli_diagnostics(rewards: torch.Tensor, alpha: float) -> Dict[str, float]:
    """Compute Bernoulli K diagnostics (not part of gradient computation)."""
    sigma = torch.exp((rewards - 1.0) / alpha).clamp(0.0, 1.0)
    o = torch.bernoulli(sigma)
    K = o.sum(dim=1)
    return {
        "K_mean": float(K.mean().item()),
        "K_zero_frac": float((K == 0).float().mean().item()),
        "K_one_frac": float((K == 1).float().mean().item()),
    }


def _softmax_diagnostics(rewards: torch.Tensor, alpha: float) -> Dict[str, float]:
    """Compute softmax ESS diagnostics."""
    logits = rewards / alpha
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    weights = exp_logits / exp_logits.sum(dim=1, keepdim=True)
    ess = 1.0 / (weights ** 2).sum(dim=1)  # per prompt
    return {
        "ess_mean": float(ess.mean().item()),
        "ess_min": float(ess.min().item()),
    }


# ---------------------------------------------------------------------------
# Advantage dispatch
# ---------------------------------------------------------------------------

def compute_advantages(method_key: str, rewards: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    if method_key == "reinforce_baseline":
        return advantages_reinforce_baseline(rewards)
    if method_key == "grpo":
        return advantages_grpo(rewards, eps=cfg.eps)
    if method_key == "continuous_maxrl_normalized":
        return advantages_continuous_maxrl_normalized(rewards, eps=cfg.eps)
    if method_key == "continuous_maxrl_unnormalized":
        return advantages_continuous_maxrl_unnormalized(rewards, eps=cfg.eps)
    if method_key == "binary_maxrl_03":
        return advantages_binary_maxrl(rewards, threshold=cfg.binary_threshold)

    # Inference softmax variants
    m = re.match(r"inference_softmax_a(\d+)", method_key)
    if m:
        a_str = m.group(1)
        alpha = float(f"{a_str[:1]}.{a_str[1:]}")  # e.g. "005" -> 0.05, "020" -> 0.20
        return advantages_inference_softmax(rewards, alpha=alpha)

    # Inference softmax normalized variants
    m = re.match(r"inference_softmax_norm_a(\d+)", method_key)
    if m:
        a_str = m.group(1)
        alpha = float(f"{a_str[:1]}.{a_str[1:]}")
        return advantages_inference_softmax_normalized(rewards, alpha=alpha, eps=cfg.eps)

    # Inference Bernoulli variants
    m = re.match(r"inference_bernoulli_a(\d+)", method_key)
    if m:
        a_str = m.group(1)
        alpha = float(f"{a_str[:1]}.{a_str[1:]}")
        return advantages_inference_bernoulli(rewards, alpha=alpha)

    raise ValueError(f"Unknown method: {method_key}")


def _parse_alpha_from_key(method_key: str) -> Optional[float]:
    """Extract alpha value from method key like 'inference_softmax_a020'."""
    m = re.match(r"inference_(?:softmax|bernoulli|softmax_norm)_a(\d+)", method_key)
    if m:
        a_str = m.group(1)
        return float(f"{a_str[:1]}.{a_str[1:]}")
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def rollout_rewards(
    theta: torch.Tensor, env: Dict[str, np.ndarray], n_rollouts: int, rng: np.random.Generator,
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
    method_key: str, theta: torch.Tensor, optimizer: torch.optim.Optimizer,
    env: Dict[str, np.ndarray], cfg: ExperimentConfig, n_rollouts: int, rng: np.random.Generator,
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
    method_key: str, seed: int, n_rollouts: int,
    env: Dict[str, np.ndarray], cfg: ExperimentConfig,
    device: str, gpu_id: int = None,
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

    # Only compute expensive log_soft_pass for a subset of eval steps
    eval_lsp = method_key.startswith("inference_") or method_key in {"grpo", "continuous_maxrl_normalized"}

    theta = torch.zeros(
        (cfg.num_prompts, cfg.num_arms), dtype=torch.float32, device=device_name, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=cfg.learning_rate)
    total_steps = steps_for_n(cfg, n_rollouts)
    history: List[Dict[str, float]] = []

    initial = evaluate_policy(theta, env, cfg, train_n=n_rollouts, rng_eval=rng_eval, eval_log_soft_pass=eval_lsp)
    initial["step"] = 0
    initial["gradient_norm"] = 0.0
    history.append(initial)

    for step in range(1, total_steps + 1):
        grad_norm = training_step(
            method_key=method_key, theta=theta, optimizer=optimizer,
            env=env, cfg=cfg, n_rollouts=n_rollouts, rng=rng_train,
        )
        if step % cfg.log_every == 0 or step == total_steps:
            metrics = evaluate_policy(theta, env, cfg, train_n=n_rollouts, rng_eval=rng_eval, eval_log_soft_pass=eval_lsp)
            metrics["step"] = step
            metrics["gradient_norm"] = grad_norm
            history.append(metrics)

    final_metrics = evaluate_policy(theta, env, cfg, train_n=n_rollouts, rng_eval=rng_eval, eval_log_soft_pass=eval_lsp)
    final_policy = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    return {
        "history": history,
        "final_metrics": final_metrics,
        "final_policy": final_policy,
    }


def run_single_job(job: Dict[str, object]) -> Dict[str, object]:
    run = run_method_seed_n(
        method_key=job["method_key"], seed=job["seed"], n_rollouts=job["n_rollouts"],
        env=job["env"], cfg=job["cfg"], device=job["device"], gpu_id=job.get("gpu_id", None),
    )
    return {
        "method_key": job["method_key"],
        "n_rollouts": job["n_rollouts"],
        "seed": job["seed"],
        "gpu_id": job.get("gpu_id", None),
        "run": run,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def series_from_runs(runs: List[Dict[str, object]], key: str) -> np.ndarray:
    out = []
    for run in runs:
        out.append([entry.get(key, np.nan) for entry in run["history"]])
    return np.asarray(out, dtype=np.float64)


def plot_with_shade(ax, x, y, label, color, linestyle="-", linewidth=2.0):
    masked = np.ma.masked_invalid(y)
    mean = np.ma.mean(masked, axis=0).filled(np.nan)
    std = np.ma.std(masked, axis=0).filled(np.nan)
    ax.plot(x, mean, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
    ax.fill_between(x, mean - std, mean + std, where=~np.isnan(mean), color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _method_list_for_main_plots(methods: List[Dict]) -> List[Dict]:
    """Select methods for main (non-alpha-sweep) figures: baselines + best inference variants."""
    return [m for m in methods if m["group"] in ("baseline", "survival")]


def make_figure_1_scaling(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    cfg: ExperimentConfig,
    out_path: Path,
) -> None:
    plot_methods = [m for m in methods if m["key"] in results]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [
        ("expected_reward", "Expected reward"),
        ("best_of_16", "Best-of-16 reward"),
        ("best_of_n", "Best-of-N reward"),
        ("integrated_log_survival", "Integrated log-survival"),
        ("pass_at_05", "Pass@1 @ 0.5"),
        ("pi_type_b", "pi(Type B)"),
    ]
    n_values = np.asarray(cfg.n_values, dtype=np.float64)

    for ax, (metric_key, title) in zip(axes.flatten(), specs):
        for method in plot_methods:
            y_runs = []
            for n in cfg.n_values:
                vals = [run["final_metrics"][metric_key] for run in results[method["key"]][n]]
                y_runs.append(vals)
            y = np.asarray(y_runs, dtype=np.float64)
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
    handles = [Line2D([0], [0], color=m["color"], lw=2, label=m["label"]) for m in plot_methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(plot_methods), 4), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_2_learning(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    cfg: ExperimentConfig,
    out_path: Path,
) -> None:
    plot_methods = [m for m in methods if m["key"] in results]
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
            for method in plot_methods:
                runs = results[method["key"]][n]
                steps = np.asarray([entry["step"] for entry in runs[0]["history"]], dtype=np.int64)
                y = series_from_runs(runs, metric_key)
                plot_with_shade(ax, steps, y, method["label"], method["color"])
            ax.set_title(f"N={n} - {row_title}")
            ax.set_xlabel("Training step")
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    handles = [Line2D([0], [0], color=m["color"], lw=2, label=m["label"]) for m in plot_methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(plot_methods), 5), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_3_advantages(
    batches: List[Dict[str, np.ndarray]], n_ref: int, out_path: Path,
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
            Line2D([0], [0], marker="o", linestyle="", markersize=6, color="tab:green", label="Scatter (color = reward)"),
        ],
        loc="upper center", ncol=3, frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_4_weight_nonlinearity(
    batch: Dict[str, np.ndarray], n_ref: int, cfg: ExperimentConfig, out_path: Path,
) -> None:
    """Advantage weight vs reward rank for GRPO, survival MaxRL, and inference softmax."""
    rewards = batch["rewards"]
    order = np.argsort(rewards)
    grpo_sorted = batch["grpo"][order]
    cmax_sorted = batch["cmax"][order]
    ranks = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ranks, grpo_sorted, color="tab:blue", linewidth=2.0, label="GRPO")
    ax.plot(ranks, cmax_sorted, color="tab:red", linewidth=2.0, label="Survival MaxRL (norm)")

    # Add inference softmax curves for each alpha
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
    colors_sm = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    for idx, alpha in enumerate(cfg.inference_alphas):
        sm_adv = advantages_inference_softmax(rewards_t, alpha=alpha).squeeze(0).numpy()
        ax.plot(ranks, sm_adv[order], color=colors_sm[idx % len(colors_sm)],
                linewidth=1.5, linestyle="--", label=f"Inf-Softmax α={alpha}")

    ax.set_title(f"Advantage Weight vs Reward Rank (N={n_ref}, one batch)")
    ax.set_xlabel("Reward rank (1=lowest, N=highest)")
    ax.set_ylabel("Advantage")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_5_alpha_sensitivity(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    cfg: ExperimentConfig,
    n_fixed: int,
    out_path: Path,
) -> None:
    """
    Figure 5: Alpha sensitivity — 1x3 grid.
    Panel 1: Final expected reward vs alpha
    Panel 2: Final best-of-N reward vs alpha
    Panel 3: Final ILS vs alpha
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_specs = [
        ("expected_reward", "Expected reward"),
        ("best_of_n", "Best-of-N reward"),
        ("integrated_log_survival", "Integrated log-survival"),
    ]

    # Gather alpha-independent baseline values
    baseline_methods = [m for m in methods if m["group"] in ("baseline", "survival") and m["key"] in results]

    softmax_methods = sorted(
        [m for m in methods if m["group"] == "inference_softmax" and m["key"] in results],
        key=lambda m: m["alpha"],
    )
    bernoulli_methods = sorted(
        [m for m in methods if m["group"] == "inference_bernoulli" and m["key"] in results],
        key=lambda m: m["alpha"],
    )

    for ax, (metric_key, title) in zip(axes, panel_specs):
        # Horizontal lines for baselines
        for bm in baseline_methods:
            if n_fixed not in results[bm["key"]]:
                continue
            vals = [run["final_metrics"][metric_key] for run in results[bm["key"]][n_fixed]]
            mean_val = np.mean(vals)
            ax.axhline(mean_val, color=bm["color"], linestyle=":", linewidth=1.5, label=bm["label"])

        # Softmax line
        if softmax_methods:
            alphas_sm = [m["alpha"] for m in softmax_methods]
            means_sm, stds_sm = [], []
            for m in softmax_methods:
                if n_fixed not in results[m["key"]]:
                    means_sm.append(np.nan)
                    stds_sm.append(np.nan)
                    continue
                vals = [run["final_metrics"][metric_key] for run in results[m["key"]][n_fixed]]
                means_sm.append(np.mean(vals))
                stds_sm.append(np.std(vals))
            means_sm = np.array(means_sm)
            stds_sm = np.array(stds_sm)
            ax.plot(alphas_sm, means_sm, color="tab:green", linewidth=2.0, marker="o", label="Inf-Softmax")
            ax.fill_between(alphas_sm, means_sm - stds_sm, means_sm + stds_sm, color="tab:green", alpha=0.15)

        # Bernoulli line
        if bernoulli_methods:
            alphas_bn = [m["alpha"] for m in bernoulli_methods]
            means_bn, stds_bn = [], []
            for m in bernoulli_methods:
                if n_fixed not in results[m["key"]]:
                    means_bn.append(np.nan)
                    stds_bn.append(np.nan)
                    continue
                vals = [run["final_metrics"][metric_key] for run in results[m["key"]][n_fixed]]
                means_bn.append(np.mean(vals))
                stds_bn.append(np.std(vals))
            means_bn = np.array(means_bn)
            stds_bn = np.array(stds_bn)
            ax.plot(alphas_bn, means_bn, color="tab:olive", linewidth=2.0, marker="s", linestyle="--", label="Inf-Bernoulli")
            ax.fill_between(alphas_bn, means_bn - stds_bn, means_bn + stds_bn, color="tab:olive", alpha=0.15)

        ax.set_xscale("log")
        ax.set_title(f"{title} (N={n_fixed})")
        ax.set_xlabel("α (temperature)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_6_learning_at_best_alpha(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    cfg: ExperimentConfig,
    best_softmax_key: str,
    best_bernoulli_key: str,
    n_fixed: int,
    out_path: Path,
) -> None:
    """
    Figure 6: Learning curves at best alpha — 2x2 grid.
    Compares baselines + survival + best inference variants at a single N.
    """
    plot_keys = set()
    for m in methods:
        if m["group"] in ("baseline", "survival"):
            plot_keys.add(m["key"])
    plot_keys.add(best_softmax_key)
    plot_keys.add(best_bernoulli_key)
    plot_methods = [m for m in methods if m["key"] in plot_keys and m["key"] in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_specs = [
        ("expected_reward", "Expected reward"),
        ("best_of_n", "Best-of-N reward"),
        ("integrated_log_survival", "Integrated log-survival"),
        ("pass_at_05", "Pass@1 @ 0.5"),
    ]

    for ax, (metric_key, title) in zip(axes.flatten(), panel_specs):
        for method in plot_methods:
            if n_fixed not in results[method["key"]]:
                continue
            runs = results[method["key"]][n_fixed]
            steps = np.asarray([entry["step"] for entry in runs[0]["history"]], dtype=np.int64)
            y = series_from_runs(runs, metric_key)
            ls = "--" if method["group"].startswith("inference") else "-"
            plot_with_shade(ax, steps, y, method["label"], method["color"], linestyle=ls)
        ax.set_title(f"N={n_fixed} - {title}")
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")

    handles = [Line2D([0], [0], color=m["color"], lw=2, label=m["label"],
                       linestyle="--" if m["group"].startswith("inference") else "-")
               for m in plot_methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(plot_methods), 4), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_7_n_scaling_comparison(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    cfg: ExperimentConfig,
    best_softmax_key: str,
    best_bernoulli_key: str,
    out_path: Path,
) -> None:
    """
    Figure 7: N scaling comparison — 1x2 grid.
    Panel 1: Best-of-N at convergence vs N
    Panel 2: ILS at convergence vs N
    """
    plot_keys = {"grpo", "continuous_maxrl_normalized", best_softmax_key, best_bernoulli_key}
    plot_methods = [m for m in methods if m["key"] in plot_keys and m["key"] in results]
    n_values = np.asarray(cfg.n_values, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric_key, title) in zip(axes, [("best_of_n", "Best-of-N reward"), ("integrated_log_survival", "Integrated log-survival")]):
        for method in plot_methods:
            y_runs = []
            for n in cfg.n_values:
                if n not in results[method["key"]]:
                    y_runs.append([np.nan])
                    continue
                vals = [run["final_metrics"][metric_key] for run in results[method["key"]][n]]
                y_runs.append(vals)
            y = np.asarray(y_runs, dtype=np.float64)
            mean = y.mean(axis=1)
            std = y.std(axis=1)
            ls = "--" if method["group"].startswith("inference") else "-"
            ax.plot(n_values, mean, color=method["color"], linewidth=2.0, label=method["label"], marker="o", linestyle=ls)
            ax.fill_between(n_values, mean - std, mean + std, color=method["color"], alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values, labels=[str(int(v)) for v in n_values])
        ax.set_title(title)
        ax.set_xlabel("Training rollouts N")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Metric value")

    handles = [Line2D([0], [0], color=m["color"], lw=2, label=m["label"],
                       linestyle="--" if m["group"].startswith("inference") else "-")
               for m in plot_methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(plot_methods), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_8_weight_comparison(
    batch: Dict[str, np.ndarray], n_ref: int, cfg: ExperimentConfig, out_path: Path,
) -> None:
    """
    Figure 8: Weight distribution comparison — 1x4 grid.
    Panels: raw rewards, GRPO adv, survival adv, inference softmax adv (best alpha).
    """
    rewards = batch["rewards"]
    order = np.argsort(rewards)
    ranks = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    ax = axes[0]
    ax.bar(ranks, rewards[order], color="gray", alpha=0.7)
    ax.set_title("Sorted rewards")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Reward")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(ranks, batch["grpo"][order], color="tab:blue", alpha=0.7)
    ax.set_title("GRPO advantages")
    ax.set_xlabel("Rank")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(ranks, batch["cmax"][order], color="tab:red", alpha=0.7)
    ax.set_title("Survival MaxRL advantages")
    ax.set_xlabel("Rank")
    ax.grid(alpha=0.3)

    ax = axes[3]
    # Use alpha=0.2 as default for this visualization
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
    sm_adv = advantages_inference_softmax(rewards_t, alpha=0.2).squeeze(0).numpy()
    ax.bar(ranks, sm_adv[order], color="tab:green", alpha=0.7)
    ax.set_title("Inf-Softmax α=0.2 advantages")
    ax.set_xlabel("Rank")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Advantage comparison (N={n_ref}, one batch)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figure_9_bernoulli_diagnostics(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    cfg: ExperimentConfig,
    n_fixed: int,
    out_path: Path,
) -> None:
    """
    Figure 9: Bernoulli K diagnostics — placeholder.
    This requires per-step K tracking which we collect via the training loop diagnostics.
    For now, we compute K statistics from final policies as a proxy.
    """
    bernoulli_methods = sorted(
        [m for m in methods if m["group"] == "inference_bernoulli" and m["key"] in results],
        key=lambda m: m["alpha"],
    )
    if not bernoulli_methods:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    for idx, bm in enumerate(bernoulli_methods):
        alpha = bm["alpha"]
        if n_fixed not in results[bm["key"]]:
            continue
        runs = results[bm["key"]][n_fixed]
        # Compute K statistics from reward distributions at different training stages
        # Using the stored history to estimate sigma statistics
        steps_list = [entry["step"] for entry in runs[0]["history"]]
        # Plot expected_reward as proxy for training progress
        y_er = series_from_runs(runs, "expected_reward")
        steps = np.asarray(steps_list, dtype=np.int64)

        # For the Bernoulli estimator, K_zero_frac decreases as rewards improve
        # We approximate: sigma_i = exp((r_i - 1)/alpha), E[K] = N * E[sigma]
        # E[sigma] for reward r ~ current policy = exp((E[r]-1)/alpha) approximately
        er_mean = np.ma.mean(np.ma.masked_invalid(y_er), axis=0).filled(np.nan)

        # Approximate expected K per prompt
        approx_sigma = np.exp((er_mean - 1.0) / alpha)
        approx_K = n_fixed * approx_sigma
        approx_K_zero = (1 - approx_sigma) ** n_fixed

        c = colors[idx % len(colors)]
        axes[0].plot(steps, approx_K_zero, color=c, linewidth=2.0, label=f"α={alpha}")
        axes[1].plot(steps, approx_K, color=c, linewidth=2.0, label=f"α={alpha}")

    axes[0].set_title(f"Approx P(K=0) over training (N={n_fixed})")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("P(K=0)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title(f"Approx E[K] over training (N={n_fixed})")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("E[K]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Advantage batch collection (for visualization)
# ---------------------------------------------------------------------------

def collect_advantage_batches(
    env: Dict[str, np.ndarray], policy_probs: np.ndarray, n_rollouts: int,
    prompt_idx: int, num_batches: int, seed: int, cfg: ExperimentConfig,
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


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks():
    """Run all 6 sanity checks from the spec. Raises AssertionError on failure."""
    print("Running sanity checks...")

    # Test 1: Binary recovery
    print("  Test 1: Binary recovery... ", end="")
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0]])
    # With alpha=0.001, sigma(1)=exp(0)=1.0, sigma(0)=exp(-1000)~=0
    # So o should be [0,0,1,0,1] almost surely, K=2
    # Advantages: positive for r=1 positions, negative for r=0 positions
    pass_count = 0
    for _ in range(20):
        adv_bn = advantages_inference_bernoulli(rewards, alpha=0.001)
        # Check that positive advantages correspond to r=1 positions
        pos_mask = adv_bn > 1e-6
        neg_mask = adv_bn < -1e-6
        if pos_mask.any():
            correct = (rewards[pos_mask] == 1.0).all() and (rewards[neg_mask] == 0.0).all()
            if correct:
                pass_count += 1
        else:
            pass_count += 1  # all zero is valid (K=0)
    assert pass_count >= 15, f"Binary recovery failed: only {pass_count}/20 passed"
    print("OK")

    # Test 2: Softmax approaches uniform as alpha -> inf
    print("  Test 2: Softmax uniform at large alpha... ", end="")
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    adv = advantages_inference_softmax(rewards, alpha=100.0)
    assert adv.abs().max() < 0.01, f"Expected near-zero, got max={adv.abs().max():.4f}"
    print("OK")

    # Test 3: Softmax concentrates on max at small alpha
    print("  Test 3: Softmax concentrates on max... ", end="")
    rewards = torch.tensor([[0.1, 0.5, 0.9]])
    adv = advantages_inference_softmax(rewards, alpha=0.01)
    assert adv[0, 2] > 0.6, f"Expected adv[2]>0.6, got {adv[0,2]:.4f}"
    assert adv[0, 0] < -0.3, f"Expected adv[0]<-0.3, got {adv[0,0]:.4f}"
    print("OK")

    # Test 4: Bernoulli K=0 frequency for low rewards with small alpha
    print("  Test 4: Bernoulli K=0 for low rewards... ", end="")
    alpha = 0.05
    rewards = torch.rand(1, 8) * 0.3
    sigma = torch.exp((rewards - 1.0) / alpha).clamp(0.0, 1.0)
    assert sigma.max() < 1e-5, f"Expected tiny sigma, got {sigma.max():.2e}"
    print("OK")

    # Test 5: Mean-centering
    print("  Test 5: Mean-centering... ", end="")
    rewards = torch.rand(4, 16)
    for alpha in [0.1, 0.2, 0.5, 1.0]:
        adv_sm = advantages_inference_softmax(rewards, alpha)
        assert abs(adv_sm.sum(dim=1).max().item()) < 1e-5, f"Softmax not mean-zero at alpha={alpha}"
        adv_bn = advantages_inference_bernoulli(rewards, alpha)
        assert abs(adv_bn.sum(dim=1).max().item()) < 1e-5, f"Bernoulli not mean-zero at alpha={alpha}"
    print("OK")

    # Test 6: Numerical stability at extreme alpha
    print("  Test 6: Numerical stability... ", end="")
    rewards = torch.tensor([[0.01, 0.99, 0.5, 0.3]])
    for alpha in [0.01, 0.02, 0.05, 10.0, 100.0]:
        adv = advantages_inference_softmax(rewards, alpha)
        assert not torch.isnan(adv).any(), f"NaN at alpha={alpha}"
        assert not torch.isinf(adv).any(), f"Inf at alpha={alpha}"
    print("OK")

    print("All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

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
                vals = [float(run["final_metrics"].get(metric, 0.0)) for run in runs]
                arr = np.asarray(vals, dtype=np.float64)
                metric_stats[metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            aggregated[method_key][str(n_key)] = {"num_runs": len(runs), "metrics": metric_stats}

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


# ---------------------------------------------------------------------------
# Best alpha selection
# ---------------------------------------------------------------------------

def find_best_alpha(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    methods: List[Dict],
    n_fixed: int,
    metric_key: str = "best_of_n",
) -> Tuple[str, str]:
    """Find best softmax and bernoulli method keys at given N by a metric."""
    best_sm_key, best_sm_val = None, -float("inf")
    best_bn_key, best_bn_val = None, -float("inf")

    for m in methods:
        if m["key"] not in results or n_fixed not in results[m["key"]]:
            continue
        vals = [run["final_metrics"][metric_key] for run in results[m["key"]][n_fixed]]
        mean_val = np.mean(vals)
        if m["group"] == "inference_softmax" and mean_val > best_sm_val:
            best_sm_key, best_sm_val = m["key"], mean_val
        if m["group"] == "inference_bernoulli" and mean_val > best_bn_val:
            best_bn_key, best_bn_val = m["key"], mean_val

    # Fallback to alpha=0.2
    if best_sm_key is None:
        best_sm_key = "inference_softmax_a020"
    if best_bn_key is None:
        best_bn_key = "inference_bernoulli_a020"

    return best_sm_key, best_bn_key


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1e: Inference MaxRL on Hard Bandit")
    parser.add_argument("--output-dir", type=str, default="results_1e")
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-trials", type=int, default=300)
    parser.add_argument("--integration-points", type=int, default=200)
    parser.add_argument("--mode", type=str, choices=["fixed_steps", "fixed_rollouts"], default="fixed_steps")
    parser.add_argument("--total-rollouts-budget", type=int, default=128000)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--sanity-only", action="store_true", help="Run sanity checks and exit")
    parser.add_argument("--inference-alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])

    # Phase control: limit which methods run to manage runtime
    parser.add_argument("--phase", type=int, default=0,
                        help="0=all methods, 1=baselines+softmax only, 2=baselines+softmax+bernoulli")
    parser.add_argument("--n-fixed", type=int, default=None,
                        help="If set, only run this single N value (for alpha sweep)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sanity_only:
        run_sanity_checks()
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        inference_alphas=tuple(args.inference_alphas),
    )

    # If --n-fixed is set, override n_values to run only that N
    if args.n_fixed is not None:
        cfg = ExperimentConfig(
            n_values=(args.n_fixed,),
            seeds=cfg.seeds,
            training_steps=cfg.training_steps,
            log_every=cfg.log_every,
            learning_rate=cfg.learning_rate,
            eval_best_trials=cfg.eval_best_trials,
            integration_points=cfg.integration_points,
            mode=cfg.mode,
            total_rollouts_budget=cfg.total_rollouts_budget,
            inference_alphas=cfg.inference_alphas,
        )

    METHODS = _build_methods(cfg)

    # Phase filtering
    if args.phase == 1:
        METHODS = [m for m in METHODS if m["group"] != "inference_bernoulli"]
    elif args.phase == 2:
        pass  # all methods

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
            inference_alphas=tuple(float(x) for x in config.get("inference_alphas", args.inference_alphas)),
        )
        METHODS = _build_methods(cfg)

        results_raw = payload["results"]
        results: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        for method_key, by_n in results_raw.items():
            results[method_key] = {int(n): runs for n, runs in by_n.items()}

        env = build_environment(cfg)

        # Advantage batches
        adv_key = next((k for k in payload.keys() if k.startswith("advantage_batches_n")), None)
        n_ref = max(cfg.n_values)
        if adv_key:
            m = re.match(r"advantage_batches_n(\d+)_prompt0", adv_key)
            if m:
                n_ref = int(m.group(1))
            adv_batches = [
                {
                    "rewards": np.asarray(b["rewards"], dtype=np.float64),
                    "grpo": np.asarray(b["grpo"], dtype=np.float64),
                    "cmax": np.asarray(b["cmax"], dtype=np.float64),
                }
                for b in payload[adv_key]
            ]
        else:
            adv_batches = None

        n_fixed_plot = 32 if 32 in cfg.n_values else min(cfg.n_values)
        best_sm, best_bn = find_best_alpha(results, METHODS, n_fixed_plot)

        make_figure_1_scaling(results, METHODS, cfg, out_dir / "fig1_scaling_curves.png")
        make_figure_2_learning(results, METHODS, cfg, out_dir / "fig2_learning_curves_by_N.png")
        if adv_batches:
            make_figure_3_advantages(adv_batches, n_ref, out_dir / "fig3_advantage_distributions.png")
            make_figure_4_weight_nonlinearity(adv_batches[0], n_ref, cfg, out_dir / "fig4_weight_nonlinearity.png")
            make_figure_8_weight_comparison(adv_batches[0], n_ref, cfg, out_dir / "fig8_weight_comparison.png")
        make_figure_5_alpha_sensitivity(results, METHODS, cfg, n_fixed_plot, out_dir / "fig5_alpha_sensitivity.png")
        make_figure_6_learning_at_best_alpha(results, METHODS, cfg, best_sm, best_bn, n_fixed_plot, out_dir / "fig6_learning_best_alpha.png")
        make_figure_7_n_scaling_comparison(results, METHODS, cfg, best_sm, best_bn, out_dir / "fig7_n_scaling_comparison.png")
        make_figure_9_bernoulli_diagnostics(results, METHODS, cfg, n_fixed_plot, out_dir / "fig9_bernoulli_diagnostics.png")

        print("Plots regenerated from existing metrics.")
        print(f"Metrics source: {metrics_path}")
        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved figures to: {out_dir}")
        return

    # ---- Run experiment ----
    run_sanity_checks()

    env = build_environment(cfg)
    print_environment_verification(cfg, env)

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
        if args.gpu_ids is None:
            selected_gpu_ids = list(range(visible_gpu_count))
        else:
            selected_gpu_ids = [int(g) for g in args.gpu_ids]
            invalid = [g for g in selected_gpu_ids if g < 0 or g >= visible_gpu_count]
            if invalid:
                raise ValueError(f"Invalid --gpu-ids {invalid}. Visible GPU ids are 0..{visible_gpu_count - 1}.")
        if not selected_gpu_ids:
            raise ValueError("No GPUs selected.")

    if args.num_workers <= 0:
        if device == "cuda":
            num_workers = len(selected_gpu_ids)
        else:
            num_workers = min(len(cfg.seeds), os.cpu_count() or 1)
    else:
        num_workers = max(1, args.num_workers)

    print(f"\nExperiment 1e: Inference MaxRL on Hard Bandit")
    print(f"  Environment: 1d hard setup v2")
    print(f"  Methods: {len(METHODS)} ({sum(1 for m in METHODS if m['group']=='inference_softmax')} softmax, "
          f"{sum(1 for m in METHODS if m['group']=='inference_bernoulli')} bernoulli, "
          f"{sum(1 for m in METHODS if m['group'] in ('baseline','survival'))} baseline/survival)")
    print(f"  Inference alphas: {list(cfg.inference_alphas)}")
    print(f"  N values: {list(cfg.n_values)}")
    print(f"  Seeds: {list(cfg.seeds)}")
    print(f"  Mode: {cfg.mode}, steps={cfg.training_steps}")
    print(f"  Device: {device}, workers={num_workers}")
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
        print(f"  GPUs: {active_gpu_ids}, slots: " + ", ".join(f"{g}:{gpu_slots.count(g)}" for g in active_gpu_ids))
    else:
        active_gpu_ids = []
        gpu_slots = []

    results_map: Dict[str, Dict[int, Dict[int, Dict[str, object]]]] = {
        m["key"]: {n: {} for n in cfg.n_values} for m in METHODS
    }
    representative_policies: Dict[str, np.ndarray] = {}
    n_ref = max(cfg.n_values)

    jobs: List[Dict[str, object]] = []
    gpu_rr = 0
    for n in cfg.n_values:
        steps_n = steps_for_n(cfg, n)
        print(f"  N={n}: steps={steps_n}")
        for method in METHODS:
            for seed in cfg.seeds:
                gpu_id = None
                if device == "cuda":
                    gpu_id = gpu_slots[gpu_rr % len(gpu_slots)]
                    gpu_rr += 1
                jobs.append({
                    "method_key": method["key"], "n_rollouts": n, "seed": seed,
                    "env": env, "cfg": cfg, "device": device, "gpu_id": gpu_id,
                })
    total_jobs = len(jobs)
    print(f"  Total runs: {total_jobs}\n")

    if num_workers == 1:
        for idx, job in enumerate(jobs, start=1):
            print(f"[{idx}/{total_jobs}] method={job['method_key']} N={job['n_rollouts']} seed={job['seed']}")
            job_out = run_single_job(job)
            m_key = job_out["method_key"]
            n = job_out["n_rollouts"]
            s = job_out["seed"]
            results_map[m_key][n][s] = job_out["run"]
            if n == n_ref and s == 42 and m_key in {"grpo", "continuous_maxrl_normalized"}:
                representative_policies[m_key] = job_out["run"]["final_policy"]
    else:
        spawn_ctx = mp.get_context("spawn") if device == "cuda" else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=spawn_ctx) as executor:
            futures = [executor.submit(run_single_job, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                job_out = future.result()
                m_key = job_out["method_key"]
                n = job_out["n_rollouts"]
                s = job_out["seed"]
                results_map[m_key][n][s] = job_out["run"]
                if n == n_ref and s == 42 and m_key in {"grpo", "continuous_maxrl_normalized"}:
                    representative_policies[m_key] = job_out["run"]["final_policy"]
                print(f"[{idx}/{total_jobs}] done method={m_key} N={n} seed={s}" +
                      (f" gpu={job_out.get('gpu_id')}" if device == "cuda" else ""))

    results: Dict[str, Dict[int, List[Dict[str, object]]]] = {
        m["key"]: {n: [results_map[m["key"]][n][s] for s in cfg.seeds] for n in cfg.n_values}
        for m in METHODS
    }

    if "continuous_maxrl_normalized" not in representative_policies:
        representative_policies["continuous_maxrl_normalized"] = results["continuous_maxrl_normalized"][n_ref][0]["final_policy"]
    if "grpo" not in representative_policies:
        representative_policies["grpo"] = results["grpo"][n_ref][0]["final_policy"]

    adv_batches = collect_advantage_batches(
        env=env, policy_probs=representative_policies["continuous_maxrl_normalized"],
        n_rollouts=n_ref, prompt_idx=0, num_batches=10, seed=2026, cfg=cfg,
    )

    metrics_payload = {
        "config": {
            "num_prompts": cfg.num_prompts, "num_arms": cfg.num_arms,
            "num_type_a": cfg.num_type_a, "num_type_b": cfg.num_type_b,
            "num_type_c": cfg.num_arms - cfg.num_type_a - cfg.num_type_b,
            "env_seed": cfg.env_seed,
            "n_values": list(cfg.n_values), "seeds": list(cfg.seeds),
            "methods": [m["key"] for m in METHODS],
            "mode": cfg.mode, "training_steps": cfg.training_steps,
            "log_every": cfg.log_every, "learning_rate": cfg.learning_rate,
            "eval_best_trials": cfg.eval_best_trials, "eval_best16_n": cfg.eval_best16_n,
            "integration_points": cfg.integration_points,
            "total_rollouts_budget": cfg.total_rollouts_budget,
            "inference_alphas": list(cfg.inference_alphas),
            "device": device, "num_workers": num_workers,
            "gpu_ids": active_gpu_ids if device == "cuda" else [],
        },
        "environment": {
            "alpha1": env["alpha1"], "beta1": env["beta1"],
            "alpha2": env["alpha2"], "beta2": env["beta2"],
            "mix_prob": env["mix_prob"], "dist_kind": env["dist_kind"],
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

    # Generate all figures
    n_fixed_plot = 32 if 32 in cfg.n_values else min(cfg.n_values)
    best_sm, best_bn = find_best_alpha(results, METHODS, n_fixed_plot)
    print(f"\nBest alpha (by best_of_n at N={n_fixed_plot}): softmax={best_sm}, bernoulli={best_bn}")

    make_figure_1_scaling(results, METHODS, cfg, out_dir / "fig1_scaling_curves.png")
    make_figure_2_learning(results, METHODS, cfg, out_dir / "fig2_learning_curves_by_N.png")
    make_figure_3_advantages(adv_batches, n_ref, out_dir / "fig3_advantage_distributions.png")
    make_figure_4_weight_nonlinearity(adv_batches[0], n_ref, cfg, out_dir / "fig4_weight_nonlinearity.png")
    make_figure_5_alpha_sensitivity(results, METHODS, cfg, n_fixed_plot, out_dir / "fig5_alpha_sensitivity.png")
    make_figure_6_learning_at_best_alpha(results, METHODS, cfg, best_sm, best_bn, n_fixed_plot, out_dir / "fig6_learning_best_alpha.png")
    make_figure_7_n_scaling_comparison(results, METHODS, cfg, best_sm, best_bn, out_dir / "fig7_n_scaling_comparison.png")
    make_figure_8_weight_comparison(adv_batches[0], n_ref, cfg, out_dir / "fig8_weight_comparison.png")
    make_figure_9_bernoulli_diagnostics(results, METHODS, cfg, n_fixed_plot, out_dir / "fig9_bernoulli_diagnostics.png")

    print(f"\nDone.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
