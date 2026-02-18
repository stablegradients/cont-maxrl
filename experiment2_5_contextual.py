#!/usr/bin/env python3
import argparse
import json
import math
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from scipy.special import betainc


TYPE_C = 0
TYPE_A = 1
TYPE_B = 2


@dataclass
class ExperimentConfig:
    # Environment
    num_prompts: int = 1000
    num_train_prompts: int = 800
    num_arms: int = 100
    num_type_a: int = 5
    num_type_b: int = 3
    env_seed: int = 42
    perturbation: float = 0.15
    min_alpha_beta: float = 0.1

    # Hard setup (same reward family as exp 1d/"exp2 hard")
    a_alpha: float = 5.0
    a_beta: float = 5.0
    b_low_alpha: float = 1.0
    b_low_beta: float = 8.0
    b_high_alpha: float = 15.0
    b_high_beta: float = 1.0
    b_mix_prob_high: float = 0.10
    c_alpha: float = 1.0
    c_beta: float = 8.0

    # Model
    arm_input_dim: int = 5
    arm_hidden_dim: int = 64
    arm_emb_dim: int = 32
    score_hidden_dim: int = 64
    temperature: float = 1.0

    # Training
    n_values: Tuple[int, ...] = (4, 16, 64, 256, 1024)
    seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    training_steps: int = 5000
    eval_every: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0

    # Eval
    eval_best_trials: int = 300
    eval_best16_n: int = 16
    integration_points: int = 200
    pass_threshold: float = 0.5
    eps: float = 1e-8

    # Figure 3/4 reference run
    analysis_n: int = 256
    analysis_seed: int = 42


METHODS = [
    {"key": "grpo", "label": "GRPO", "color": "tab:blue"},
    {"key": "continuous_maxrl_normalized", "label": "Cont. MaxRL (norm)", "color": "tab:orange"},
    {"key": "continuous_maxrl_unnormalized", "label": "Cont. MaxRL (unnorm)", "color": "tab:red"},
    {"key": "reinforce_baseline", "label": "REINFORCE+baseline", "color": "black"},
]


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
    if cfg.num_train_prompts >= cfg.num_prompts:
        raise ValueError("num_train_prompts must be smaller than num_prompts")

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
        type_ids[i, a_idx] = TYPE_A
        type_ids[i, b_idx] = TYPE_B

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
    survival_pass = np.where(dist_kind == 1, (1.0 - mix_prob) * surv1_pass + mix_prob * surv2_pass, surv1_pass)

    mean1, var1 = beta_mean_var(alpha1, beta1)
    mean2, var2 = beta_mean_var(alpha2, beta2)
    mean_rewards = np.where(dist_kind == 1, (1.0 - mix_prob) * mean1 + mix_prob * mean2, mean1)
    var_rewards = np.where(dist_kind == 1, (1.0 - mix_prob) * var1 + mix_prob * var2, var1)

    mask_a = (type_ids == TYPE_A).astype(np.float64)
    mask_b = (type_ids == TYPE_B).astype(np.float64)
    mask_c = (type_ids == TYPE_C).astype(np.float64)

    # Per-arm features fed to the policy network.
    arm_features = np.stack([alpha1, beta1, alpha2, beta2, mix_prob], axis=-1).astype(np.float32)

    split_perm = np.random.default_rng(cfg.env_seed + 1).permutation(cfg.num_prompts)
    train_prompt_indices = np.sort(split_perm[: cfg.num_train_prompts])
    test_prompt_indices = np.sort(split_perm[cfg.num_train_prompts :])

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
        "arm_features": arm_features,
        "tau_grid": tau_grid,
        "survival_grid": survival_grid,
        "survival_pass": survival_pass,
        "mean_rewards": mean_rewards,
        "var_rewards": var_rewards,
        "train_prompt_indices": train_prompt_indices,
        "test_prompt_indices": test_prompt_indices,
    }


def print_environment_verification(cfg: ExperimentConfig, env: Dict[str, np.ndarray]) -> None:
    a_mean = cfg.a_alpha / (cfg.a_alpha + cfg.a_beta)
    b_low_mean = cfg.b_low_alpha / (cfg.b_low_alpha + cfg.b_low_beta)
    b_high_mean = cfg.b_high_alpha / (cfg.b_high_alpha + cfg.b_high_beta)
    b_mean = (1.0 - cfg.b_mix_prob_high) * b_low_mean + cfg.b_mix_prob_high * b_high_mean
    c_mean = cfg.c_alpha / (cfg.c_alpha + cfg.c_beta)

    b_low_tail_08 = float(beta_survival(np.asarray(cfg.b_low_alpha), np.asarray(cfg.b_low_beta), np.asarray(0.8)))
    b_high_tail_08 = float(beta_survival(np.asarray(cfg.b_high_alpha), np.asarray(cfg.b_high_beta), np.asarray(0.8)))
    b_tail_08 = (1.0 - cfg.b_mix_prob_high) * b_low_tail_08 + cfg.b_mix_prob_high * b_high_tail_08

    mask_a = env["mask_a"] > 0
    mask_b = env["mask_b"] > 0
    mask_c = env["mask_c"] > 0
    mean_rewards = env["mean_rewards"]

    print("Environment verification (Experiment 2.5 contextual hard setup):")
    print(f"  Type A: {cfg.num_type_a} arms, mean≈{a_mean:.3f} (observed≈{float(mean_rewards[mask_a].mean()):.3f})")
    print(
        f"  Type B: {cfg.num_type_b} arms, mean≈{b_mean:.3f} (observed≈{float(mean_rewards[mask_b].mean()):.3f}), "
        f"P(r>0.8)≈{b_tail_08:.3f}"
    )
    print(
        f"  Type C: {cfg.num_arms - cfg.num_type_a - cfg.num_type_b} arms, "
        f"mean≈{c_mean:.3f} (observed≈{float(mean_rewards[mask_c].mean()):.3f})"
    )
    print(f"  Total: {cfg.num_arms} arms")


def print_feature_examples(env: Dict[str, np.ndarray], prompt_idx: int = 0) -> None:
    type_row = env["type_ids"][prompt_idx]
    features = env["arm_features"][prompt_idx]

    idx_a = int(np.where(type_row == TYPE_A)[0][0])
    idx_b = int(np.where(type_row == TYPE_B)[0][0])
    idx_c = int(np.where(type_row == TYPE_C)[0][0])

    def fmt(v: np.ndarray) -> str:
        return np.array2string(v, precision=3, suppress_small=True)

    print("Feature sanity (prompt 0, [alpha1,beta1,alpha2,beta2,mix_prob]):")
    print(f"  Type A arm {idx_a}: {fmt(features[idx_a])}")
    print(f"  Type B arm {idx_b}: {fmt(features[idx_b])}")
    print(f"  Type C arm {idx_c}: {fmt(features[idx_c])}")


class ContextualBanditPolicy(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.arm_encoder = nn.Sequential(
            nn.Linear(cfg.arm_input_dim, cfg.arm_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.arm_hidden_dim, cfg.arm_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.arm_hidden_dim, cfg.arm_emb_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(cfg.arm_emb_dim * 2, cfg.score_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.score_hidden_dim, 1),
        )

        # Start near-uniform policy at initialization.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(self, arm_features: torch.Tensor) -> torch.Tensor:
        # arm_features: [B, A, 5]
        bsz, num_arms, feat_dim = arm_features.shape
        arm_flat = arm_features.view(bsz * num_arms, feat_dim)
        arm_emb = self.arm_encoder(arm_flat).view(bsz, num_arms, -1)
        prompt_emb = arm_emb.mean(dim=1, keepdim=True).expand(-1, num_arms, -1)
        x = torch.cat([arm_emb, prompt_emb], dim=-1)
        logits = self.score_head(x.view(bsz * num_arms, -1)).view(bsz, num_arms)
        return logits


def check_initial_policy_uniform(cfg: ExperimentConfig, env: Dict[str, np.ndarray], device: str) -> None:
    model = ContextualBanditPolicy(cfg).to(device)
    with torch.no_grad():
        feats = torch.from_numpy(env["arm_features"][: min(128, cfg.num_prompts)]).to(device=device, dtype=torch.float32)
        probs = torch.softmax(model(feats) / cfg.temperature, dim=-1)
        max_p = float(probs.max().item())
        min_p = float(probs.min().item())
    print(f"Init policy sanity: pi.max={max_p:.6f}, pi.min={min_p:.6f}")
    if max_p >= 0.05 or min_p <= 0.001:
        print("Warning: initialization is not close to uniform (sanity bound violated).")


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


def sample_rewards_for_batch(
    env: Dict[str, np.ndarray],
    prompt_indices: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    # actions: [B, N]
    bsz = actions.shape[0]
    out = np.zeros_like(actions, dtype=np.float64)
    for row in range(bsz):
        out[row] = sample_rewards_for_actions(env, int(prompt_indices[row]), actions[row], rng)
    return out


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


def advantages_continuous_maxrl_unnormalized(rewards: torch.Tensor) -> torch.Tensor:
    num_prompts, n = rewards.shape
    sorted_rewards, sort_idx = torch.sort(rewards, dim=1)
    prev = torch.cat(
        [torch.zeros((num_prompts, 1), dtype=rewards.dtype, device=rewards.device), sorted_rewards[:, :-1]], dim=1
    )
    gaps = sorted_rewards - prev
    survivors = torch.arange(n, 0, -1, dtype=rewards.dtype, device=rewards.device).view(1, n)
    w_sorted = torch.cumsum(gaps / survivors, dim=1)
    centered = w_sorted - w_sorted.mean(dim=1, keepdim=True)

    out = torch.zeros_like(rewards)
    out.scatter_(1, sort_idx, centered)
    return out


def advantages_continuous_maxrl_normalized(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    num_prompts, n = rewards.shape
    sorted_rewards, sort_idx = torch.sort(rewards, dim=1)
    prev = torch.cat(
        [torch.zeros((num_prompts, 1), dtype=rewards.dtype, device=rewards.device), sorted_rewards[:, :-1]], dim=1
    )
    gaps = sorted_rewards - prev
    survivors = torch.arange(n, 0, -1, dtype=rewards.dtype, device=rewards.device).view(1, n)
    w_sorted = torch.cumsum(gaps / survivors, dim=1)
    centered = w_sorted - w_sorted.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, keepdim=True, unbiased=False)
    normalized = centered / (std + eps)
    zero_std = std < eps
    if zero_std.any():
        normalized = torch.where(zero_std, torch.zeros_like(normalized), normalized)

    out = torch.zeros_like(rewards)
    out.scatter_(1, sort_idx, normalized)
    return out


def compute_advantages(method_key: str, rewards: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    if method_key == "grpo":
        return advantages_grpo(rewards, cfg.eps)
    if method_key == "continuous_maxrl_normalized":
        return advantages_continuous_maxrl_normalized(rewards, cfg.eps)
    if method_key == "continuous_maxrl_unnormalized":
        return advantages_continuous_maxrl_unnormalized(rewards)
    if method_key == "reinforce_baseline":
        return advantages_reinforce_baseline(rewards)
    raise ValueError(f"Unknown method: {method_key}")


def compute_policy_probs(
    model: ContextualBanditPolicy,
    arm_features: np.ndarray,
    prompt_indices: np.ndarray,
    cfg: ExperimentConfig,
    device: str,
    chunk_size: int = 256,
) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompt_indices), chunk_size):
            idx = prompt_indices[start : start + chunk_size]
            feats = torch.from_numpy(arm_features[idx]).to(device=device, dtype=torch.float32)
            logits = model(feats) / cfg.temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)


def estimate_best_of_k_mc(
    probs: np.ndarray,
    env: Dict[str, np.ndarray],
    prompt_indices: np.ndarray,
    k: int,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_prompts, num_arms = probs.shape
    out = np.zeros(num_prompts, dtype=np.float64)
    for local_idx in range(num_prompts):
        prompt_idx = int(prompt_indices[local_idx])
        actions = rng.choice(num_arms, size=(trials, k), p=probs[local_idx])
        rewards = sample_rewards_for_actions(env, prompt_idx, actions, rng)
        out[local_idx] = rewards.max(axis=1).mean()
    return out


def evaluate_split(
    model: ContextualBanditPolicy,
    env: Dict[str, np.ndarray],
    prompt_indices: np.ndarray,
    cfg: ExperimentConfig,
    train_n: int,
    rng_eval: np.random.Generator,
    device: str,
    return_per_prompt: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    probs = compute_policy_probs(model, env["arm_features"], prompt_indices, cfg, device)

    mean_rewards = env["mean_rewards"][prompt_indices]
    survival_grid = env["survival_grid"][prompt_indices]
    survival_pass = env["survival_pass"][prompt_indices]
    mask_a = env["mask_a"][prompt_indices]
    mask_b = env["mask_b"][prompt_indices]

    expected_reward_prompt = (probs * mean_rewards).sum(axis=1)
    best16_prompt = estimate_best_of_k_mc(
        probs=probs,
        env=env,
        prompt_indices=prompt_indices,
        k=cfg.eval_best16_n,
        trials=cfg.eval_best_trials,
        rng=rng_eval,
    )
    bestn_prompt = estimate_best_of_k_mc(
        probs=probs,
        env=env,
        prompt_indices=prompt_indices,
        k=train_n,
        trials=cfg.eval_best_trials,
        rng=rng_eval,
    )

    p_tau = np.einsum("pa,pat->pt", probs, survival_grid)
    p_tau = np.clip(p_tau, 1e-12, 1.0)
    ils_prompt = integrate_trapezoid(np.log(p_tau), env["tau_grid"], axis=1)
    pass_prompt = (probs * survival_pass).sum(axis=1)

    pi_type_a_prompt = (probs * mask_a).sum(axis=1)
    pi_type_b_prompt = (probs * mask_b).sum(axis=1)

    metrics = {
        "expected_reward": float(expected_reward_prompt.mean()),
        "best_of_16": float(best16_prompt.mean()),
        "best_of_n": float(bestn_prompt.mean()),
        "integrated_log_survival": float(ils_prompt.mean()),
        "pass_at_05": float(pass_prompt.mean()),
        "pi_type_b": float(pi_type_b_prompt.mean()),
        "pi_type_a": float(pi_type_a_prompt.mean()),
    }

    if not return_per_prompt:
        return metrics, None

    per_prompt = {
        "expected_reward": expected_reward_prompt,
        "best_of_16": best16_prompt,
        "best_of_n": bestn_prompt,
        "integrated_log_survival": ils_prompt,
        "pass_at_05": pass_prompt,
        "pi_type_b": pi_type_b_prompt,
        "pi_type_a": pi_type_a_prompt,
    }
    return metrics, per_prompt


def merge_train_test_metrics(train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Dict[str, float]:
    out = {f"train_{k}": float(v) for k, v in train_metrics.items()}
    out.update({k: float(v) for k, v in test_metrics.items()})
    return out


def training_step(
    method_key: str,
    model: ContextualBanditPolicy,
    optimizer: torch.optim.Optimizer,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    n_rollouts: int,
    rng_train: np.random.Generator,
    device: str,
) -> float:
    model.train()

    train_pool = env["train_prompt_indices"]
    prompt_batch = rng_train.choice(train_pool, size=cfg.batch_size, replace=True)

    feats = torch.from_numpy(env["arm_features"][prompt_batch]).to(device=device, dtype=torch.float32)
    logits = model(feats) / cfg.temperature

    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample((n_rollouts,)).transpose(0, 1)  # [B, N]

    actions_np = actions.detach().cpu().numpy()
    rewards_np = sample_rewards_for_batch(env, prompt_batch, actions_np, rng_train)
    rewards = torch.from_numpy(rewards_np).to(device=device, dtype=torch.float32)

    log_probs = torch.log_softmax(logits, dim=-1).gather(1, actions)
    advantages = compute_advantages(method_key, rewards, cfg)

    loss = -(advantages * log_probs).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item())
    optimizer.step()
    return grad_norm


def run_method_seed_n(
    method_key: str,
    seed: int,
    n_rollouts: int,
    env: Dict[str, np.ndarray],
    cfg: ExperimentConfig,
    device: str,
    checkpoint_dir: Path,
    gpu_id: Optional[int] = None,
    collect_prompt_analysis: bool = False,
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

    model = ContextualBanditPolicy(cfg).to(device_name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    history: List[Dict[str, float]] = []

    train_metrics_0, _ = evaluate_split(
        model=model,
        env=env,
        prompt_indices=env["train_prompt_indices"],
        cfg=cfg,
        train_n=n_rollouts,
        rng_eval=rng_eval,
        device=device_name,
        return_per_prompt=False,
    )
    test_metrics_0, _ = evaluate_split(
        model=model,
        env=env,
        prompt_indices=env["test_prompt_indices"],
        cfg=cfg,
        train_n=n_rollouts,
        rng_eval=rng_eval,
        device=device_name,
        return_per_prompt=False,
    )
    initial = merge_train_test_metrics(train_metrics_0, test_metrics_0)
    initial["step"] = 0
    initial["gradient_norm"] = 0.0
    history.append(initial)

    for step in range(1, cfg.training_steps + 1):
        grad_norm = training_step(
            method_key=method_key,
            model=model,
            optimizer=optimizer,
            env=env,
            cfg=cfg,
            n_rollouts=n_rollouts,
            rng_train=rng_train,
            device=device_name,
        )

        if step % cfg.eval_every == 0 or step == cfg.training_steps:
            train_metrics, _ = evaluate_split(
                model=model,
                env=env,
                prompt_indices=env["train_prompt_indices"],
                cfg=cfg,
                train_n=n_rollouts,
                rng_eval=rng_eval,
                device=device_name,
                return_per_prompt=False,
            )
            test_metrics, _ = evaluate_split(
                model=model,
                env=env,
                prompt_indices=env["test_prompt_indices"],
                cfg=cfg,
                train_n=n_rollouts,
                rng_eval=rng_eval,
                device=device_name,
                return_per_prompt=False,
            )
            entry = merge_train_test_metrics(train_metrics, test_metrics)
            entry["step"] = step
            entry["gradient_norm"] = grad_norm
            history.append(entry)

    train_metrics_f, _ = evaluate_split(
        model=model,
        env=env,
        prompt_indices=env["train_prompt_indices"],
        cfg=cfg,
        train_n=n_rollouts,
        rng_eval=rng_eval,
        device=device_name,
        return_per_prompt=False,
    )
    test_metrics_f, _ = evaluate_split(
        model=model,
        env=env,
        prompt_indices=env["test_prompt_indices"],
        cfg=cfg,
        train_n=n_rollouts,
        rng_eval=rng_eval,
        device=device_name,
        return_per_prompt=False,
    )
    final_metrics = merge_train_test_metrics(train_metrics_f, test_metrics_f)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{method_key}_N{n_rollouts}_seed{seed}.pt"
    torch.save(
        {
            "method": method_key,
            "seed": seed,
            "n_rollouts": n_rollouts,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "arm_input_dim": cfg.arm_input_dim,
                "arm_hidden_dim": cfg.arm_hidden_dim,
                "arm_emb_dim": cfg.arm_emb_dim,
                "score_hidden_dim": cfg.score_hidden_dim,
                "temperature": cfg.temperature,
            },
        },
        ckpt_path,
    )

    prompt_analysis = None
    if collect_prompt_analysis:
        _, per_prompt = evaluate_split(
            model=model,
            env=env,
            prompt_indices=env["test_prompt_indices"],
            cfg=cfg,
            train_n=n_rollouts,
            rng_eval=np.random.default_rng(seed + 55555 + n_rollouts),
            device=device_name,
            return_per_prompt=True,
        )
        prompt_analysis = {
            "pi_type_b": per_prompt["pi_type_b"],
            "best_of_16": per_prompt["best_of_16"],
        }

    return {
        "history": history,
        "final_metrics": final_metrics,
        "checkpoint_path": str(ckpt_path),
        "prompt_analysis": prompt_analysis,
    }


def run_single_job(job: Dict[str, object]) -> Dict[str, object]:
    run = run_method_seed_n(
        method_key=job["method_key"],
        seed=job["seed"],
        n_rollouts=job["n_rollouts"],
        env=job["env"],
        cfg=job["cfg"],
        device=job["device"],
        checkpoint_dir=Path(job["checkpoint_dir"]),
        gpu_id=job.get("gpu_id", None),
        collect_prompt_analysis=bool(job.get("collect_prompt_analysis", False)),
    )
    return {
        "method_key": job["method_key"],
        "n_rollouts": job["n_rollouts"],
        "seed": job["seed"],
        "gpu_id": job.get("gpu_id", None),
        "run": run,
    }


def series_from_runs(runs: List[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([[entry.get(key, np.nan) for entry in run["history"]] for run in runs], dtype=np.float64)


def plot_with_shade(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
    masked = np.ma.masked_invalid(y)
    mean = np.ma.mean(masked, axis=0).filled(np.nan)
    std = np.ma.std(masked, axis=0).filled(np.nan)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
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
        ("pass_at_05", "Pass@1 @ 0.5"),
        ("pi_type_b", "pi(Type B)"),
    ]
    n_values = np.asarray(cfg.n_values, dtype=np.float64)

    for ax, (metric_key, title) in zip(axes.flatten(), specs):
        for method in METHODS:
            y_by_n = []
            for n in cfg.n_values:
                vals = [run["final_metrics"][metric_key] for run in results[method["key"]][n]]
                y_by_n.append(vals)
            y = np.asarray(y_by_n, dtype=np.float64)
            mean = y.mean(axis=1)
            std = y.std(axis=1)
            ax.plot(n_values, mean, color=method["color"], linewidth=2.0, marker="o", label=method["label"])
            ax.fill_between(n_values, mean - std, mean + std, color=method["color"], alpha=0.2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_values, labels=[str(int(v)) for v in n_values])
        ax.set_title(title)
        ax.set_xlabel("Training rollouts N")
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel("Metric value")
    axes[1, 0].set_ylabel("Metric value")
    fig.legend(handles=method_legend_handles(), loc="upper center", ncol=3, frameon=False)
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
    fig, axes = plt.subplots(2, num_cols, figsize=(max(16.0, 4.0 * num_cols), 9), sharex=False)
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

    axes[0, 0].set_ylabel("Held-out metric")
    axes[1, 0].set_ylabel("Held-out metric")
    fig.legend(handles=method_legend_handles(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_3_train_vs_test(
    results: Dict[str, Dict[int, List[Dict[str, object]]]],
    cfg: ExperimentConfig,
    analysis_n: int,
    out_path: Path,
) -> None:
    metrics = [
        ("expected_reward", "Expected reward"),
        ("best_of_16", "Best-of-16"),
        ("best_of_n", "Best-of-N"),
        ("integrated_log_survival", "Integrated log-survival"),
        ("pass_at_05", "Pass@1 @ 0.5"),
        ("pi_type_b", "pi(Type B)"),
        ("pi_type_a", "pi(Type A)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    method_positions = np.arange(len(METHODS), dtype=np.float64)
    width = 0.36

    for idx, (metric_key, title) in enumerate(metrics):
        ax = axes.flat[idx]
        train_means = []
        test_means = []
        train_stds = []
        test_stds = []

        for method in METHODS:
            runs = results[method["key"]][analysis_n]
            train_vals = np.asarray([r["final_metrics"][f"train_{metric_key}"] for r in runs], dtype=np.float64)
            test_vals = np.asarray([r["final_metrics"][metric_key] for r in runs], dtype=np.float64)
            train_means.append(train_vals.mean())
            test_means.append(test_vals.mean())
            train_stds.append(train_vals.std())
            test_stds.append(test_vals.std())

        ax.bar(
            method_positions - width / 2,
            train_means,
            width,
            yerr=train_stds,
            capsize=3,
            color="tab:gray",
            alpha=0.65,
            label="Train",
        )
        ax.bar(
            method_positions + width / 2,
            test_means,
            width,
            yerr=test_stds,
            capsize=3,
            color=[m["color"] for m in METHODS],
            alpha=0.85,
            label="Test",
        )
        ax.set_title(title)
        ax.set_xticks(method_positions, [m["label"] for m in METHODS], rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")

    # Hide the unused 8th panel.
    axes.flat[-1].axis("off")

    handles = [
        Line2D([0], [0], color="tab:gray", lw=6, alpha=0.65, label="Train"),
        Line2D([0], [0], color="tab:blue", lw=6, alpha=0.85, label="Test"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Train vs Test (N={analysis_n})", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_figure_4_per_prompt_analysis(
    analysis_points: Dict[str, Dict[str, np.ndarray]],
    analysis_n: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for method in METHODS:
        key = method["key"]
        if key not in analysis_points:
            continue
        x = np.asarray(analysis_points[key]["pi_type_b"], dtype=np.float64)
        y = np.asarray(analysis_points[key]["best_of_16"], dtype=np.float64)
        ax.scatter(x, y, s=18, alpha=0.55, color=method["color"], label=method["label"])

    ax.set_title(f"Per-prompt: pi(Type B) vs Best-of-16 on Test Prompts (N={analysis_n})")
    ax.set_xlabel("pi(Type B) per prompt")
    ax.set_ylabel("Best-of-16 per prompt")
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
            # Primary comparison is on held-out (non-train_ prefixed) metrics.
            if metric.startswith("train_"):
                continue
            candidates = []
            for method_key, by_n in aggregated.items():
                if n_key in by_n:
                    candidates.append((method_key, by_n[n_key]["metrics"][metric]["mean"]))
            if candidates:
                winner = max(candidates, key=lambda x: x[1])
                best_by_n[n_key][metric] = {"method": winner[0], "value": float(winner[1])}

    config = payload.get("config", {})
    config_subset = {
        "n_values": config.get("n_values"),
        "seeds": config.get("seeds"),
        "methods": config.get("methods"),
        "training_steps": config.get("training_steps"),
        "batch_size": config.get("batch_size"),
        "device": config.get("device"),
    }

    return {
        "config_subset": config_subset,
        "aggregated_final_by_method_and_n": aggregated,
        "best_method_by_n_and_metric": best_by_n,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2.5: Contextual bandit with shared MLP policy")
    parser.add_argument("--output-dir", type=str, default="results_2_5")
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 16, 64, 256, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-trials", type=int, default=300)
    parser.add_argument("--integration-points", type=int, default=200)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--analysis-n", type=int, default=256)
    parser.add_argument("--analysis-seed", type=int, default=42)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--metrics-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"

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
            eval_every=int(config.get("eval_every", args.eval_every)),
            batch_size=int(config.get("batch_size", args.batch_size)),
            learning_rate=float(config.get("learning_rate", args.lr)),
            grad_clip_norm=float(config.get("grad_clip_norm", args.grad_clip)),
            eval_best_trials=int(config.get("eval_best_trials", args.eval_trials)),
            integration_points=int(config.get("integration_points", args.integration_points)),
            analysis_n=int(config.get("analysis_n", args.analysis_n)),
            analysis_seed=int(config.get("analysis_seed", args.analysis_seed)),
        )

        results_raw = payload["results"]
        results: Dict[str, Dict[int, List[Dict[str, object]]]] = {}
        for method_key, by_n in results_raw.items():
            results[method_key] = {int(n): runs for n, runs in by_n.items()}

        analysis_key = next((k for k in payload.keys() if k.startswith("analysis_points_n")), None)
        if analysis_key is None:
            raise ValueError("No analysis_points key found in metrics payload.")
        m = re.match(r"analysis_points_n(\d+)_seed(\d+)_test", analysis_key)
        analysis_n = int(m.group(1)) if m else cfg.analysis_n
        analysis_points_raw = payload[analysis_key]
        analysis_points = {
            mk: {
                "pi_type_b": np.asarray(v["pi_type_b"], dtype=np.float64),
                "best_of_16": np.asarray(v["best_of_16"], dtype=np.float64),
            }
            for mk, v in analysis_points_raw.items()
        }

        make_figure_1_scaling(results, cfg, out_dir / "fig1_scaling_curves.png")
        make_figure_2_learning(results, cfg, out_dir / "fig2_learning_curves.png")
        make_figure_3_train_vs_test(results, cfg, analysis_n, out_dir / "fig3_train_vs_test.png")
        make_figure_4_per_prompt_analysis(analysis_points, analysis_n, out_dir / "fig4_per_prompt_analysis.png")

        print("Plots regenerated from existing metrics.")
        print(f"Metrics source: {metrics_path}")
        print(f"Saved summary metrics to: {summary_path}")
        print(f"Saved figures to: {out_dir}")
        return

    cfg = ExperimentConfig(
        n_values=tuple(args.n_values),
        seeds=tuple(args.seeds),
        training_steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_clip_norm=args.grad_clip,
        eval_best_trials=args.eval_trials,
        integration_points=args.integration_points,
        analysis_n=args.analysis_n,
        analysis_seed=args.analysis_seed,
    )

    env = build_environment(cfg)
    print_environment_verification(cfg, env)
    print_feature_examples(env)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but no GPU detected; falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    check_initial_policy_uniform(cfg, env, device=("cuda:0" if device == "cuda" else "cpu"))

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
                raise ValueError(f"Invalid --gpu-ids {invalid}. Visible IDs: 0..{visible_gpu_count - 1}")
        if not selected_gpu_ids:
            raise ValueError("No GPUs selected. Provide at least one id via --gpu-ids.")

    if args.num_workers <= 0:
        if device == "cuda":
            num_workers = len(selected_gpu_ids)
        else:
            num_workers = min(len(cfg.seeds), os.cpu_count() or 1)
    else:
        num_workers = max(1, args.num_workers)

    print("Experiment 2.5 setup:")
    print("  Type A = Beta(5,5), Type B = 0.9*Beta(1,8) + 0.1*Beta(15,1), Type C = Beta(1,8)")
    print(f"  prompts={cfg.num_prompts} (train={len(env['train_prompt_indices'])}, test={len(env['test_prompt_indices'])}), arms={cfg.num_arms}")
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
        print(f"Using GPUs: {active_gpu_ids}")
        print("Worker slots per GPU: " + ", ".join(f"{g}:{gpu_slots.count(g)}" for g in active_gpu_ids))
    else:
        active_gpu_ids = []
        gpu_slots = []

    print(f"Workers: {num_workers}")

    analysis_n = cfg.analysis_n if cfg.analysis_n in cfg.n_values else min(cfg.n_values, key=lambda x: abs(x - cfg.analysis_n))

    results_map: Dict[str, Dict[int, Dict[int, Dict[str, object]]]] = {
        m["key"]: {n: {} for n in cfg.n_values} for m in METHODS
    }

    analysis_points: Dict[str, Dict[str, np.ndarray]] = {}

    jobs: List[Dict[str, object]] = []
    gpu_rr = 0
    for n in cfg.n_values:
        print(f"N={n}: steps={cfg.training_steps}")
        for method in METHODS:
            for seed in cfg.seeds:
                gpu_id = None
                if device == "cuda":
                    gpu_id = gpu_slots[gpu_rr % len(gpu_slots)]
                    gpu_rr += 1
                collect_prompt_analysis = bool(n == analysis_n and seed == cfg.analysis_seed)
                jobs.append(
                    {
                        "method_key": method["key"],
                        "n_rollouts": n,
                        "seed": seed,
                        "env": env,
                        "cfg": cfg,
                        "device": device,
                        "gpu_id": gpu_id,
                        "checkpoint_dir": str(checkpoints_dir),
                        "collect_prompt_analysis": collect_prompt_analysis,
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
            mkey = job_out["method_key"]
            n = job_out["n_rollouts"]
            s = job_out["seed"]
            run = job_out["run"]
            results_map[mkey][n][s] = run
            if run.get("prompt_analysis") is not None:
                analysis_points[mkey] = run["prompt_analysis"]
    else:
        spawn_ctx = mp.get_context("spawn") if device == "cuda" else None
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=spawn_ctx) as executor:
            futures = [executor.submit(run_single_job, job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), start=1):
                job_out = future.result()
                mkey = job_out["method_key"]
                n = job_out["n_rollouts"]
                s = job_out["seed"]
                run = job_out["run"]
                results_map[mkey][n][s] = run
                if run.get("prompt_analysis") is not None:
                    analysis_points[mkey] = run["prompt_analysis"]
                if device == "cuda":
                    print(f"[{idx}/{total_jobs}] done method={mkey} N={n} seed={s} gpu={job_out.get('gpu_id')}")
                else:
                    print(f"[{idx}/{total_jobs}] done method={mkey} N={n} seed={s}")

    results: Dict[str, Dict[int, List[Dict[str, object]]]] = {
        m["key"]: {n: [results_map[m["key"]][n][s] for s in cfg.seeds] for n in cfg.n_values}
        for m in METHODS
    }

    analysis_key = f"analysis_points_n{analysis_n}_seed{cfg.analysis_seed}_test"

    metrics_payload = {
        "config": {
            "num_prompts": cfg.num_prompts,
            "num_train_prompts": cfg.num_train_prompts,
            "num_test_prompts": cfg.num_prompts - cfg.num_train_prompts,
            "num_arms": cfg.num_arms,
            "num_type_a": cfg.num_type_a,
            "num_type_b": cfg.num_type_b,
            "num_type_c": cfg.num_arms - cfg.num_type_a - cfg.num_type_b,
            "env_seed": cfg.env_seed,
            "n_values": list(cfg.n_values),
            "seeds": list(cfg.seeds),
            "methods": [m["key"] for m in METHODS],
            "training_steps": cfg.training_steps,
            "eval_every": cfg.eval_every,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "grad_clip_norm": cfg.grad_clip_norm,
            "eval_best_trials": cfg.eval_best_trials,
            "eval_best16_n": cfg.eval_best16_n,
            "integration_points": cfg.integration_points,
            "pass_threshold": cfg.pass_threshold,
            "temperature": cfg.temperature,
            "analysis_n": analysis_n,
            "analysis_seed": cfg.analysis_seed,
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
            "train_prompt_indices": env["train_prompt_indices"],
            "test_prompt_indices": env["test_prompt_indices"],
        },
        "results": results,
        analysis_key: analysis_points,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(metrics_payload), f, indent=2)

    summary_payload = build_metrics_summary(metrics_payload)
    summary_path = out_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(summary_payload), f, indent=2)

    make_figure_1_scaling(results, cfg, out_dir / "fig1_scaling_curves.png")
    make_figure_2_learning(results, cfg, out_dir / "fig2_learning_curves.png")
    make_figure_3_train_vs_test(results, cfg, analysis_n, out_dir / "fig3_train_vs_test.png")
    make_figure_4_per_prompt_analysis(analysis_points, analysis_n, out_dir / "fig4_per_prompt_analysis.png")

    print("\nDone.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved figures to: {out_dir}")
    print(f"Saved checkpoints to: {checkpoints_dir}")


if __name__ == "__main__":
    main()
