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
from matplotlib.patches import Patch
from scipy.special import betainc
from scipy.stats import beta as beta_dist


TYPE_C = 0
TYPE_A = 1
TYPE_B = 2
TYPE_LABELS = {TYPE_A: "A", TYPE_B: "B", TYPE_C: "C"}
TYPE_COLORS = {TYPE_A: "tab:red", TYPE_B: "tab:blue", TYPE_C: "tab:gray"}


@dataclass
class ExperimentConfig:
    num_prompts: int = 100
    num_arms: int = 50
    num_type_a: int = 5
    num_type_b: int = 5
    rollouts_per_prompt: int = 16
    training_steps: int = 2000
    log_every: int = 50
    learning_rate: float = 3e-3
    eval_best_of_n_trials: int = 200
    integration_points: int = 200
    env_seed: int = 42
    perturbation: float = 0.3
    min_alpha_beta: float = 0.1
    eps: float = 1e-8
    pass_eval_threshold: float = 0.8
    binary_threshold_low: float = 0.5
    binary_threshold_high: float = 0.8


METHODS = [
    {"key": "reinforce_baseline", "label": "REINFORCE+baseline", "color": "black"},
    {"key": "grpo", "label": "GRPO", "color": "tab:blue"},
    {"key": "continuous_maxrl_normalized", "label": "Continuous MaxRL (normalized)", "color": "tab:red"},
    {"key": "continuous_maxrl", "label": "Continuous MaxRL (unnormalized)", "color": "orange"},
    {"key": "binary_maxrl_05", "label": "Binary MaxRL@0.5", "color": "tab:green"},
    {"key": "binary_maxrl_08", "label": "Binary MaxRL@0.8", "color": "tab:purple"},
]


def make_tau_grid(cfg: ExperimentConfig) -> np.ndarray:
    return np.linspace(0.0, 1.0 - 1e-6, cfg.integration_points, dtype=np.float64)


def beta_survival(alpha: np.ndarray, beta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return 1.0 - betainc(alpha, beta, tau)


def beta_mean_var(alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = alpha / (alpha + beta)
    var = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    return mean, var


def distribution_survival(spec: Dict[str, float], tau: np.ndarray) -> np.ndarray:
    if spec["mode"] == "beta":
        return beta_survival(spec["alpha"], spec["beta"], tau)
    low_surv = beta_survival(spec["low_alpha"], spec["low_beta"], tau)
    high_surv = beta_survival(spec["high_alpha"], spec["high_beta"], tau)
    return (1.0 - spec["p_high"]) * low_surv + spec["p_high"] * high_surv


def distribution_mean_std(spec: Dict[str, float]) -> Tuple[float, float]:
    if spec["mode"] == "beta":
        mean, var = beta_mean_var(spec["alpha"], spec["beta"])
        return float(mean), float(np.sqrt(var))

    low_mean, low_var = beta_mean_var(spec["low_alpha"], spec["low_beta"])
    high_mean, high_var = beta_mean_var(spec["high_alpha"], spec["high_beta"])
    p = spec["p_high"]
    mean = (1.0 - p) * low_mean + p * high_mean
    second_moment = (1.0 - p) * (low_var + low_mean**2) + p * (high_var + high_mean**2)
    var = second_moment - mean**2
    return float(mean), float(np.sqrt(max(0.0, var)))


def distribution_pdf(spec: Dict[str, float], x: np.ndarray) -> np.ndarray:
    if spec["mode"] == "beta":
        return beta_dist.pdf(x, spec["alpha"], spec["beta"])
    low_pdf = beta_dist.pdf(x, spec["low_alpha"], spec["low_beta"])
    high_pdf = beta_dist.pdf(x, spec["high_alpha"], spec["high_beta"])
    return (1.0 - spec["p_high"]) * low_pdf + spec["p_high"] * high_pdf


def distribution_spec_str(spec: Dict[str, float]) -> str:
    if spec["mode"] == "beta":
        return f"Beta({spec['alpha']:.1f},{spec['beta']:.1f})"
    return (
        "Mixture((1-p)*Beta("
        f"{spec['low_alpha']:.1f},{spec['low_beta']:.1f}) + p*Beta({spec['high_alpha']:.1f},{spec['high_beta']:.1f}), "
        f"p={spec['p_high']:.2f})"
    )


def distribution_metrics(spec: Dict[str, float], cfg: ExperimentConfig) -> Dict[str, float]:
    tau_grid = make_tau_grid(cfg)
    survival = distribution_survival(spec, tau_grid)
    survival = np.clip(survival, 1e-12, 1.0)
    mean, std = distribution_mean_std(spec)
    best_of_n = float(np.trapezoid(1.0 - (1.0 - survival) ** cfg.rollouts_per_prompt, tau_grid))
    ils = float(np.trapezoid(np.log(survival), tau_grid))
    return {
        "mean": float(mean),
        "std": float(std),
        "p_gt_05": float(distribution_survival(spec, cfg.binary_threshold_low)),
        "p_gt_08": float(distribution_survival(spec, cfg.binary_threshold_high)),
        "p_gt_095": float(distribution_survival(spec, 0.95)),
        "best_of_16": best_of_n,
        "integrated_log_survival": ils,
    }


def dominates_on_core_metrics(a_metrics: Dict[str, float], b_metrics: Dict[str, float]) -> bool:
    core = ["mean", "best_of_16", "integrated_log_survival", "p_gt_08"]
    return all(a_metrics[k] >= b_metrics[k] for k in core)


def build_environment(
    cfg: ExperimentConfig,
    a_spec: Dict[str, float],
    b_spec: Dict[str, float],
    c_spec: Dict[str, float],
) -> Dict[str, np.ndarray]:
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
                spec = a_spec
            elif t == TYPE_B:
                spec = b_spec
            else:
                spec = c_spec

            if spec["mode"] == "beta":
                alpha1[i, a] = max(
                    cfg.min_alpha_beta, spec["alpha"] + rng.uniform(-cfg.perturbation, cfg.perturbation)
                )
                beta1[i, a] = max(
                    cfg.min_alpha_beta, spec["beta"] + rng.uniform(-cfg.perturbation, cfg.perturbation)
                )
                alpha2[i, a] = alpha1[i, a]
                beta2[i, a] = beta1[i, a]
                mix_prob[i, a] = 0.0
                dist_kind[i, a] = 0
            else:
                alpha1[i, a] = max(
                    cfg.min_alpha_beta, spec["low_alpha"] + rng.uniform(-cfg.perturbation, cfg.perturbation)
                )
                beta1[i, a] = max(
                    cfg.min_alpha_beta, spec["low_beta"] + rng.uniform(-cfg.perturbation, cfg.perturbation)
                )
                alpha2[i, a] = max(
                    cfg.min_alpha_beta, spec["high_alpha"] + rng.uniform(-cfg.perturbation, cfg.perturbation)
                )
                beta2[i, a] = max(
                    cfg.min_alpha_beta, spec["high_beta"] + rng.uniform(-cfg.perturbation, cfg.perturbation)
                )
                mix_prob[i, a] = float(spec["p_high"])
                dist_kind[i, a] = 1

    mask_a = (type_ids == TYPE_A).astype(np.float64)
    mask_b = (type_ids == TYPE_B).astype(np.float64)
    mask_c = (type_ids == TYPE_C).astype(np.float64)

    tau_grid = make_tau_grid(cfg)
    surv1 = beta_survival(alpha1[:, :, None], beta1[:, :, None], tau_grid[None, None, :])
    surv2 = beta_survival(alpha2[:, :, None], beta2[:, :, None], tau_grid[None, None, :])
    survival_grid = np.where(
        dist_kind[:, :, None] == 1,
        (1.0 - mix_prob[:, :, None]) * surv1 + mix_prob[:, :, None] * surv2,
        surv1,
    )
    surv1_at_08 = beta_survival(alpha1, beta1, cfg.pass_eval_threshold)
    surv2_at_08 = beta_survival(alpha2, beta2, cfg.pass_eval_threshold)
    survival_at_08 = np.where(
        dist_kind == 1, (1.0 - mix_prob) * surv1_at_08 + mix_prob * surv2_at_08, surv1_at_08
    )
    surv1_at_095 = beta_survival(alpha1, beta1, 0.95)
    surv2_at_095 = beta_survival(alpha2, beta2, 0.95)
    survival_at_095 = np.where(
        dist_kind == 1, (1.0 - mix_prob) * surv1_at_095 + mix_prob * surv2_at_095, surv1_at_095
    )

    mean1, _ = beta_mean_var(alpha1, beta1)
    mean2, _ = beta_mean_var(alpha2, beta2)
    mean_rewards = np.where(dist_kind == 1, (1.0 - mix_prob) * mean1 + mix_prob * mean2, mean1)

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
        "survival_at_08": survival_at_08,
        "survival_at_095": survival_at_095,
        "mean_rewards": mean_rewards,
}


def sample_rewards_for_actions(
    env: Dict[str, np.ndarray], prompt_idx: int, actions: np.ndarray, rng: np.random.Generator
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
    choose_high = rng.random(size=actions.shape) < p
    rewards2 = rng.beta(a2, b2)
    mix_rewards = np.where(choose_high, rewards2, rewards1)
    return np.where(kind == 1, mix_rewards, rewards1)


def estimate_best_of_n_mc(
    probs: np.ndarray,
    env: Dict[str, np.ndarray],
    n_rollouts: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_prompts, num_arms = probs.shape
    out = np.zeros(num_prompts, dtype=np.float64)
    for i in range(num_prompts):
        actions = rng.choice(num_arms, size=(trials, n_rollouts), p=probs[i])
        rewards = sample_rewards_for_actions(env, i, actions, rng)
        out[i] = rewards.max(axis=1).mean()
    return out


def restricted_best_of_n(
    probs: np.ndarray,
    mask: np.ndarray,
    survival_grid: np.ndarray,
    tau_grid: np.ndarray,
    n_rollouts: int,
) -> np.ndarray:
    mass = (probs * mask).sum(axis=1, keepdims=True)
    safe_mass = np.clip(mass, 1e-12, None)
    restricted_probs = (probs * mask) / safe_mass
    p_tau = np.einsum("pk,pkt->pt", restricted_probs, survival_grid)
    return np.trapezoid(1.0 - (1.0 - np.clip(p_tau, 0.0, 1.0)) ** n_rollouts, tau_grid, axis=1)


def summarize(values: np.ndarray) -> float:
    return float(np.mean(values))


def evaluate_policy(
    theta: torch.Tensor,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    rng_eval: np.random.Generator,
) -> Dict[str, float]:
    probs = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    p_tau = np.einsum("pk,pkt->pt", probs, env["survival_grid"])
    p_tau = np.clip(p_tau, 1e-12, 1.0)

    expected_reward_prompt = (probs * env["mean_rewards"]).sum(axis=1)
    best_of_16_prompt = estimate_best_of_n_mc(
        probs=probs,
        env=env,
        n_rollouts=cfg.rollouts_per_prompt,
        trials=cfg.eval_best_of_n_trials,
        rng=rng_eval,
    )
    ils_prompt = np.trapezoid(np.log(p_tau), env["tau_grid"], axis=1)
    pass_at_08_prompt = (probs * env["survival_at_08"]).sum(axis=1)

    pi_type_a_prompt = (probs * env["mask_a"]).sum(axis=1)
    pi_type_b_prompt = (probs * env["mask_b"]).sum(axis=1)
    pi_type_c_prompt = (probs * env["mask_c"]).sum(axis=1)

    best_of_16_type_a_prompt = restricted_best_of_n(
        probs=probs,
        mask=env["mask_a"],
        survival_grid=env["survival_grid"],
        tau_grid=env["tau_grid"],
        n_rollouts=cfg.rollouts_per_prompt,
    )
    best_of_16_type_b_prompt = restricted_best_of_n(
        probs=probs,
        mask=env["mask_b"],
        survival_grid=env["survival_grid"],
        tau_grid=env["tau_grid"],
        n_rollouts=cfg.rollouts_per_prompt,
    )

    return {
        "expected_reward": summarize(expected_reward_prompt),
        "best_of_16": summarize(best_of_16_prompt),
        "integrated_log_survival": summarize(ils_prompt),
        "pass_at_08": summarize(pass_at_08_prompt),
        "pi_type_a": summarize(pi_type_a_prompt),
        "pi_type_b": summarize(pi_type_b_prompt),
        "pi_type_c": summarize(pi_type_c_prompt),
        "best_of_16_type_a_policy": summarize(best_of_16_type_a_prompt),
        "best_of_16_type_b_policy": summarize(best_of_16_type_b_prompt),
    }


def representative_type_policy_summary(
    env: Dict[str, np.ndarray], cfg: ExperimentConfig, prompt_idx: int
) -> Dict[str, Dict[str, float]]:
    out = {}
    for type_name, mask_key in [("A", "mask_a"), ("B", "mask_b"), ("C", "mask_c")]:
        mask = env[mask_key][prompt_idx].astype(bool)
        probs = np.zeros(cfg.num_arms, dtype=np.float64)
        probs[mask] = 1.0 / mask.sum()

        p_tau = (probs[:, None] * env["survival_grid"][prompt_idx]).sum(axis=0)
        p_tau = np.clip(p_tau, 1e-12, 1.0)
        expected_reward = float((probs * env["mean_rewards"][prompt_idx]).sum())
        best_of_16 = float(
            np.trapezoid(1.0 - (1.0 - p_tau) ** cfg.rollouts_per_prompt, env["tau_grid"])
        )
        ils = float(np.trapezoid(np.log(p_tau), env["tau_grid"]))
        pass_at_08 = float((probs * env["survival_at_08"][prompt_idx]).sum())
        pass_at_095 = float((probs * env["survival_at_095"][prompt_idx]).sum())

        out[type_name] = {
            "expected_reward": expected_reward,
            "best_of_16": best_of_16,
            "integrated_log_survival": ils,
            "pass_at_08": pass_at_08,
            "pass_at_095": pass_at_095,
        }
    return out


def format_type_line(name: str, spec: Dict[str, float], metrics: Dict[str, float]) -> str:
    return (
        f"Type {name} ({distribution_spec_str(spec)}): mean={metrics['mean']:.4f}, std={metrics['std']:.4f}, "
        f"P(r>0.5)={metrics['p_gt_05']:.4f}, P(r>0.8)={metrics['p_gt_08']:.4f}, "
        f"P(r>0.95)={metrics['p_gt_095']:.6f}, "
        f"best-of-16 expected max={metrics['best_of_16']:.4f}, "
        f"integrated log-survival={metrics['integrated_log_survival']:.4f}"
    )


def choose_environment_with_fallback(cfg: ExperimentConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, object], str]:
    c_spec = {"mode": "beta", "alpha": 2.0, "beta": 8.0}
    candidate_specs = [
        {
            "a_spec": {"mode": "beta", "alpha": 20.0, "beta": 5.0},
            "b_spec": {"mode": "beta", "alpha": 1.5, "beta": 3.0},
        },
        {
            "a_spec": {"mode": "beta", "alpha": 20.0, "beta": 5.0},
            "b_spec": {"mode": "beta", "alpha": 1.2, "beta": 2.5},
        },
        {
            "a_spec": {"mode": "beta", "alpha": 20.0, "beta": 5.0},
            "b_spec": {"mode": "beta", "alpha": 1.0, "beta": 2.0},
        },
        {
            "a_spec": {"mode": "beta", "alpha": 20.0, "beta": 5.0},
            "b_spec": {
                "mode": "mixture",
                "low_alpha": 1.0,
                "low_beta": 5.0,
                "high_alpha": 8.0,
                "high_beta": 1.0,
                "p_high": 0.30,
            },
        },
        {
            "a_spec": {"mode": "beta", "alpha": 40.0, "beta": 10.0},
            "b_spec": {"mode": "beta", "alpha": 1.0, "beta": 2.0},
        },
        {
            "a_spec": {"mode": "beta", "alpha": 40.0, "beta": 10.0},
            "b_spec": {
                "mode": "mixture",
                "low_alpha": 1.0,
                "low_beta": 5.0,
                "high_alpha": 8.0,
                "high_beta": 1.0,
                "p_high": 0.30,
            },
        },
    ]

    analysis_lines = []
    selected = None
    selected_env = None

    for candidate in candidate_specs:
        a_spec = candidate["a_spec"]
        b_spec = candidate["b_spec"]
        a_metrics = distribution_metrics(a_spec, cfg)
        b_metrics = distribution_metrics(b_spec, cfg)

        env = build_environment(cfg=cfg, a_spec=a_spec, b_spec=b_spec, c_spec=c_spec)
        rep = representative_type_policy_summary(env, cfg, prompt_idx=0)
        rep_core_dom = (
            rep["A"]["expected_reward"] >= rep["B"]["expected_reward"]
            and rep["A"]["best_of_16"] >= rep["B"]["best_of_16"]
            and rep["A"]["integrated_log_survival"] >= rep["B"]["integrated_log_survival"]
            and rep["A"]["pass_at_08"] >= rep["B"]["pass_at_08"]
        )

        global_dom = dominates_on_core_metrics(a_metrics, b_metrics) and a_metrics["p_gt_095"] >= b_metrics["p_gt_095"]
        analysis_lines.append(
            (
                f"Candidate A={distribution_spec_str(a_spec)}, "
                f"B={distribution_spec_str(b_spec)} -> "
                f"global_A_dominates_B={global_dom}, representative_A_dominates_B={rep_core_dom}"
            )
        )

        if not rep_core_dom:
            selected = {
                "a_spec": a_spec,
                "b_spec": b_spec,
                "c_spec": c_spec,
            }
            selected_env = env
            break

    if selected is None:
        fallback = candidate_specs[-1]
        selected = {
            "a_spec": fallback["a_spec"],
            "b_spec": fallback["b_spec"],
            "c_spec": c_spec,
        }
        selected_env = build_environment(cfg=cfg, a_spec=selected["a_spec"], b_spec=selected["b_spec"], c_spec=selected["c_spec"])
        analysis_lines.append(
            "WARNING: Representative prompt still shows A dominating B on core metrics after fallback."
        )

    a_metrics = distribution_metrics(selected["a_spec"], cfg)
    b_metrics = distribution_metrics(selected["b_spec"], cfg)
    c_metrics = distribution_metrics(selected["c_spec"], cfg)
    rep = representative_type_policy_summary(selected_env, cfg, prompt_idx=0)
    rep_dom = (
        rep["A"]["expected_reward"] >= rep["B"]["expected_reward"]
        and rep["A"]["best_of_16"] >= rep["B"]["best_of_16"]
        and rep["A"]["integrated_log_survival"] >= rep["B"]["integrated_log_survival"]
        and rep["A"]["pass_at_08"] >= rep["B"]["pass_at_08"]
    )

    analysis_lines.append("")
    analysis_lines.append("Selected arm-type base parameters:")
    analysis_lines.append(f"  Type A: {distribution_spec_str(selected['a_spec'])}")
    analysis_lines.append(f"  Type B: {distribution_spec_str(selected['b_spec'])}")
    analysis_lines.append(f"  Type C: {distribution_spec_str(selected['c_spec'])}")
    analysis_lines.append("")
    analysis_lines.append("Base distribution properties:")
    analysis_lines.append("  " + format_type_line("A", selected["a_spec"], a_metrics))
    analysis_lines.append("  " + format_type_line("B", selected["b_spec"], b_metrics))
    analysis_lines.append("  " + format_type_line("C", selected["c_spec"], c_metrics))
    analysis_lines.append("")
    analysis_lines.append("Representative prompt #0 (uniform policy within each type):")
    analysis_lines.append(f"  Type A best-of-16 expected max: {rep['A']['best_of_16']:.4f}")
    analysis_lines.append(f"  Type B best-of-16 expected max: {rep['B']['best_of_16']:.4f}")
    analysis_lines.append(f"  Type A integrated log-survival: {rep['A']['integrated_log_survival']:.4f}")
    analysis_lines.append(f"  Type B integrated log-survival: {rep['B']['integrated_log_survival']:.4f}")
    analysis_lines.append(f"  Type A pass@1 (r>0.8): {rep['A']['pass_at_08']:.4f}")
    analysis_lines.append(f"  Type B pass@1 (r>0.8): {rep['B']['pass_at_08']:.4f}")
    analysis_lines.append(f"  Type A pass@1 (r>0.95): {rep['A']['pass_at_095']:.6f}")
    analysis_lines.append(f"  Type B pass@1 (r>0.95): {rep['B']['pass_at_095']:.6f}")
    analysis_lines.append(
        "  Dominance check on core metrics (expected, best-of-16, integrated log-survival, pass@0.8): "
        + ("A dominates B" if rep_dom else "tradeoff present")
    )
    return selected_env, selected, "\n".join(analysis_lines)


def advantages_reinforce_baseline(rewards: torch.Tensor) -> torch.Tensor:
    return rewards - rewards.mean(dim=1, keepdim=True)


def advantages_grpo(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    out = (rewards - mean) / (std + eps)
    mask = std < eps
    if mask.any():
        out = torch.where(mask, torch.zeros_like(out), out)
    return out


def advantages_binary_maxrl_threshold(rewards: torch.Tensor, threshold: float) -> torch.Tensor:
    n = rewards.size(1)
    b = (rewards > threshold).to(rewards.dtype)
    k = b.sum(dim=1, keepdim=True)
    fail_adv = -torch.ones_like(rewards) / float(n)
    success_adv = (n - k) / (k * float(n) + 1e-12)
    out = torch.where(b > 0, success_adv, fail_adv)
    zero_success = k.squeeze(1) == 0
    if zero_success.any():
        out[zero_success] = 0.0
    return out


def advantages_continuous_maxrl(
    rewards: torch.Tensor,
    normalize: bool,
    eps: float,
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

    centered_sorted_weights = raw_sorted_weights - raw_sorted_weights.mean(dim=1, keepdim=True)
    if normalize:
        std = centered_sorted_weights.std(dim=1, keepdim=True, unbiased=False)
        centered_sorted_weights = centered_sorted_weights / (std + eps)
        zero_std = std < eps
        if zero_std.any():
            centered_sorted_weights = torch.where(
                zero_std, torch.zeros_like(centered_sorted_weights), centered_sorted_weights
            )

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
) -> Dict[str, object]:
    probs = torch.softmax(theta, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample((cfg.rollouts_per_prompt,)).transpose(0, 1)
    log_probs = torch.log_softmax(theta, dim=-1).gather(1, actions)

    actions_np = actions.detach().cpu().numpy()
    rewards_np = np.zeros((cfg.num_prompts, cfg.rollouts_per_prompt), dtype=np.float64)
    for i in range(cfg.num_prompts):
        rewards_np[i] = sample_rewards_for_actions(env, i, actions_np[i], rng_train)
    rewards = torch.from_numpy(rewards_np).to(dtype=theta.dtype, device=theta.device)

    diag = {
        "ess": None,
        "top_weight_frac": None,
        "ess_by_top_type": None,
    }

    if method_key == "reinforce_baseline":
        advantages = advantages_reinforce_baseline(rewards)
    elif method_key == "grpo":
        advantages = advantages_grpo(rewards, eps=cfg.eps)
    elif method_key == "continuous_maxrl_normalized":
        advantages, ess, top_frac = advantages_continuous_maxrl(rewards, normalize=True, eps=cfg.eps)
        ess_np = ess.detach().cpu().numpy()
        top_frac_np = top_frac.detach().cpu().numpy()
        diag["ess"] = float(ess_np.mean())
        diag["top_weight_frac"] = float(top_frac_np.mean())

        top_rollout_idx = rewards_np.argmax(axis=1)
        top_actions = actions_np[np.arange(cfg.num_prompts), top_rollout_idx]
        top_types = env["type_ids"][np.arange(cfg.num_prompts), top_actions]

        ess_by_type = {}
        for t, label in [(TYPE_A, "A"), (TYPE_B, "B"), (TYPE_C, "C")]:
            mask = top_types == t
            ess_by_type[label] = float(ess_np[mask].mean()) if np.any(mask) else float("nan")
        diag["ess_by_top_type"] = ess_by_type
    elif method_key == "continuous_maxrl":
        advantages, _, _ = advantages_continuous_maxrl(rewards, normalize=False, eps=cfg.eps)
    elif method_key == "binary_maxrl_05":
        advantages = advantages_binary_maxrl_threshold(rewards, threshold=cfg.binary_threshold_low)
    elif method_key == "binary_maxrl_08":
        advantages = advantages_binary_maxrl_threshold(rewards, threshold=cfg.binary_threshold_high)
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
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng_train = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 10000)

    theta = torch.zeros((cfg.num_prompts, cfg.num_arms), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=cfg.learning_rate)
    history: List[Dict[str, object]] = []

    initial = evaluate_policy(theta, env, cfg, rng_eval)
    initial["step"] = 0
    initial["gradient_norm"] = 0.0
    initial["ess"] = None
    initial["top_weight_frac"] = None
    initial["ess_by_top_type"] = None
    history.append(initial)

    for step in range(1, cfg.training_steps + 1):
        step_out = training_step(
            method_key=method_key,
            theta=theta,
            optimizer=optimizer,
            env=env,
            cfg=cfg,
            rng_train=rng_train,
        )
        if step % cfg.log_every == 0:
            metrics = evaluate_policy(theta, env, cfg, rng_eval)
            metrics["step"] = step
            metrics["gradient_norm"] = step_out["gradient_norm"]
            metrics["ess"] = step_out["diagnostics"]["ess"]
            metrics["top_weight_frac"] = step_out["diagnostics"]["top_weight_frac"]
            metrics["ess_by_top_type"] = step_out["diagnostics"]["ess_by_top_type"]
            history.append(metrics)

    final_policy = torch.softmax(theta.detach(), dim=-1).cpu().numpy()
    return {"history": history, "final_policy": final_policy}


def stack_series(method_runs: List[Dict[str, object]], key: str) -> np.ndarray:
    rows = []
    for run in method_runs:
        row = []
        for entry in run["history"]:
            value = entry.get(key, None)
            if value is None:
                value = np.nan
            row.append(value)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def stack_nested_series(method_runs: List[Dict[str, object]], parent_key: str, child_key: str) -> np.ndarray:
    rows = []
    for run in method_runs:
        row = []
        for entry in run["history"]:
            parent = entry.get(parent_key, None)
            value = np.nan if parent is None else parent.get(child_key, np.nan)
            row.append(value)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def plot_with_shade(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
    masked = np.ma.masked_invalid(y)
    mean = np.ma.mean(masked, axis=0).filled(np.nan)
    std = np.ma.std(masked, axis=0).filled(np.nan)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - std, mean + std, where=~np.isnan(mean), color=color, alpha=0.2)


def make_figure_1(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    specs = [
        ("expected_reward", "Expected Reward"),
        ("best_of_16", "Best-of-16 Reward"),
        ("integrated_log_survival", "Integrated Log-Survival"),
        ("pass_at_08", "Pass@1 (r > 0.8)"),
    ]
    for ax, (metric_key, title) in zip(axes.flatten(), specs):
        for method in METHODS:
            runs = all_results[method["key"]]
            steps = np.asarray([h["step"] for h in runs[0]["history"]], dtype=np.int64)
            y = stack_series(runs, metric_key)
            plot_with_shade(ax, steps, y, method["label"], method["color"])
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_2a(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    specs = [
        ("pi_type_a", "pi(Type A)"),
        ("pi_type_b", "pi(Type B)"),
        ("pi_type_c", "pi(Type C)"),
    ]
    for ax, (metric_key, title) in zip(axes, specs):
        for method in METHODS:
            runs = all_results[method["key"]]
            steps = np.asarray([h["step"] for h in runs[0]["history"]], dtype=np.int64)
            y = stack_series(runs, metric_key)
            plot_with_shade(ax, steps, y, method["label"], method["color"])
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Average probability mass")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_2b(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2 * len(METHODS), 4), sharey=True)
    if len(METHODS) == 1:
        axes = [axes]

    for ax, method in zip(axes, METHODS):
        runs = all_results[method["key"]]
        steps = np.asarray([h["step"] for h in runs[0]["history"]], dtype=np.int64)
        y_a = np.nanmean(stack_series(runs, "pi_type_a"), axis=0)
        y_b = np.nanmean(stack_series(runs, "pi_type_b"), axis=0)
        y_c = np.nanmean(stack_series(runs, "pi_type_c"), axis=0)
        ax.stackplot(
            steps,
            y_a,
            y_b,
            y_c,
            labels=["Type A", "Type B", "Type C"],
            colors=[TYPE_COLORS[TYPE_A], TYPE_COLORS[TYPE_B], TYPE_COLORS[TYPE_C]],
            alpha=0.9,
        )
        ax.set_title(method["label"])
        ax.set_xlabel("Step")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Probability mass")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_3(all_results: Dict[str, List[Dict[str, object]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    ax = axes[0]
    for method in METHODS:
        runs = all_results[method["key"]]
        steps = np.asarray([h["step"] for h in runs[0]["history"]], dtype=np.int64)
        y = stack_series(runs, "gradient_norm")
        plot_with_shade(ax, steps, y, method["label"], method["color"])
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Training step")
    ax.set_ylabel("L2 norm")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    cont_norm_runs = all_results["continuous_maxrl_normalized"]
    steps = np.asarray([h["step"] for h in cont_norm_runs[0]["history"]], dtype=np.int64)
    ax = axes[1]
    for label, color in [("A", TYPE_COLORS[TYPE_A]), ("B", TYPE_COLORS[TYPE_B]), ("C", TYPE_COLORS[TYPE_C])]:
        y = stack_nested_series(cont_norm_runs, "ess_by_top_type", label)
        plot_with_shade(ax, steps, y, f"Top type {label}", color)
    ax.set_title("ESS by Encountered Top Type\n(Continuous MaxRL normalized)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("ESS")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    y = stack_series(cont_norm_runs, "top_weight_frac")
    plot_with_shade(ax, steps, y, "Top sample weight fraction", "tab:red")
    ax.set_title("Top Sample Weight Fraction\n(Continuous MaxRL normalized)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("w_(N) / sum(raw w)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_4(
    all_results: Dict[str, List[Dict[str, object]]],
    env: Dict[str, np.ndarray],
    seeds: List[int],
    out_path: Path,
) -> None:
    prompts = [0, 50, 99]
    prompt_labels = [f"Prompt #{p+1}" for p in prompts]
    seed_idx = seeds.index(42) if 42 in seeds else 0
    arms = np.arange(env["type_ids"].shape[1])

    fig, axes = plt.subplots(len(prompts), len(METHODS), figsize=(4.0 * len(METHODS), 9), sharey=True)
    if len(METHODS) == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, method in enumerate(METHODS):
        final_policy = all_results[method["key"]][seed_idx]["final_policy"]
        for row, prompt_idx in enumerate(prompts):
            ax = axes[row, col]
            probs = final_policy[prompt_idx]
            type_row = env["type_ids"][prompt_idx]
            colors = [TYPE_COLORS[int(t)] for t in type_row]
            ax.bar(arms, probs, color=colors, alpha=0.85, width=0.9)
            if row == 0:
                ax.set_title(method["label"])
            if col == 0:
                ax.set_ylabel(prompt_labels[row])
            ax.set_ylim(0.0, max(0.2, probs.max() * 1.1))
            ax.set_xticks([])
            ax.grid(alpha=0.2, axis="y")

    legend_handles = [
        Patch(color=TYPE_COLORS[TYPE_A], label="Type A"),
        Patch(color=TYPE_COLORS[TYPE_B], label="Type B"),
        Patch(color=TYPE_COLORS[TYPE_C], label="Type C"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_5(env: Dict[str, np.ndarray], out_path: Path) -> None:
    prompt_idx = 0
    x = np.linspace(1e-4, 1.0 - 1e-4, 600)
    fig, ax = plt.subplots(figsize=(8, 5))

    for t, label in [(TYPE_A, "Type A"), (TYPE_B, "Type B"), (TYPE_C, "Type C")]:
        idx = np.where(env["type_ids"][prompt_idx] == t)[0][0]
        spec = {
            "mode": "mixture" if int(env["dist_kind"][prompt_idx, idx]) == 1 else "beta",
            "alpha": float(env["alpha1"][prompt_idx, idx]),
            "beta": float(env["beta1"][prompt_idx, idx]),
            "low_alpha": float(env["alpha1"][prompt_idx, idx]),
            "low_beta": float(env["beta1"][prompt_idx, idx]),
            "high_alpha": float(env["alpha2"][prompt_idx, idx]),
            "high_beta": float(env["beta2"][prompt_idx, idx]),
            "p_high": float(env["mix_prob"][prompt_idx, idx]),
        }
        y = distribution_pdf(spec, x)
        mean, _ = distribution_mean_std(spec)
        ax.plot(x, y, color=TYPE_COLORS[t], linewidth=2.0, label=f"{label} {distribution_spec_str(spec)}")
        ax.axvline(mean, color=TYPE_COLORS[t], linestyle="--", alpha=0.7, linewidth=1.2)

    ax.axvspan(0.8, 1.0, color="gold", alpha=0.15, label="r > 0.8")
    ax.set_title("Representative Reward Distributions (Prompt #1)")
    ax.set_xlabel("Reward r")
    ax.set_ylabel("PDF")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
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


def main():
    parser = argparse.ArgumentParser(description="Experiment 1b: Hard Bandit Mean vs Tail Tradeoffs")
    parser.add_argument("--output-dir", type=str, default="results_1b")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-trials", type=int, default=200)
    parser.add_argument("--integration-points", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
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

    env, selected_params, analysis_text = choose_environment_with_fallback(cfg)
    analysis_path = out_dir / "arm_type_analysis.txt"
    analysis_path.write_text(analysis_text + "\n", encoding="utf-8")
    print(analysis_text)
    print(f"\nSaved arm-type analysis to: {analysis_path}")

    all_results: Dict[str, List[Dict[str, object]]] = {m["key"]: [] for m in METHODS}
    for method in METHODS:
        print(f"\n=== Running method: {method['label']} ===")
        for seed in args.seeds:
            print(f"Seed {seed}...")
            run_out = run_method_seed(method["key"], seed, env, cfg)
            all_results[method["key"]].append(run_out)

    metrics_payload = {
        "config": {
            "num_prompts": cfg.num_prompts,
            "num_arms": cfg.num_arms,
            "num_type_a": cfg.num_type_a,
            "num_type_b": cfg.num_type_b,
            "num_type_c": cfg.num_arms - cfg.num_type_a - cfg.num_type_b,
            "rollouts_per_prompt": cfg.rollouts_per_prompt,
            "training_steps": cfg.training_steps,
            "log_every": cfg.log_every,
            "learning_rate": cfg.learning_rate,
            "eval_best_of_n_trials": cfg.eval_best_of_n_trials,
            "integration_points": cfg.integration_points,
            "env_seed": cfg.env_seed,
            "perturbation": cfg.perturbation,
            "seeds": args.seeds,
            "binary_threshold_low": cfg.binary_threshold_low,
            "binary_threshold_high": cfg.binary_threshold_high,
            "pass_eval_threshold": cfg.pass_eval_threshold,
        },
        "selected_type_params": selected_params,
        "environment": {
            "alpha1": env["alpha1"],
            "beta1": env["beta1"],
            "alpha2": env["alpha2"],
            "beta2": env["beta2"],
            "mix_prob": env["mix_prob"],
            "dist_kind": env["dist_kind"],
            "type_ids": env["type_ids"],
        },
        "methods": all_results,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(metrics_payload), f, indent=2)

    make_figure_1(all_results, out_dir / "fig1_learning_curves.png")
    make_figure_2a(all_results, out_dir / "fig2a_arm_allocation_lines.png")
    make_figure_2b(all_results, out_dir / "fig2b_arm_allocation_stacked.png")
    make_figure_3(all_results, out_dir / "fig3_diagnostics.png")
    make_figure_4(all_results, env, args.seeds, out_dir / "fig4_final_policies.png")
    make_figure_5(env, out_dir / "fig5_reward_distributions.png")

    print("\nDone.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
