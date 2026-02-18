from __future__ import annotations

from typing import Literal

import torch

AdvantageMethod = Literal[
    "grpo",
    "maxrl_unnorm",
    "maxrl_norm",
    "reinforce_baseline",
]


def _ensure_1d(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1:
        raise ValueError(f"Expected 1D rewards, got shape {tuple(rewards.shape)}")
    return rewards


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standard GRPO z-score advantage on one prompt group of rewards."""
    rewards = _ensure_1d(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def reinforce_baseline_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with per-group mean baseline."""
    rewards = _ensure_1d(rewards)
    return rewards - rewards.mean()


def maxrl_unnormalized_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Continuous MaxRL (unnormalized) gap-over-survivors estimator."""
    rewards = _ensure_1d(rewards)
    group_size = rewards.numel()
    if group_size <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards)
    prev = torch.cat(
        [torch.zeros(1, dtype=rewards.dtype, device=rewards.device), sorted_rewards[:-1]],
        dim=0,
    )
    gaps = sorted_rewards - prev
    survivors = torch.arange(
        group_size,
        0,
        -1,
        dtype=rewards.dtype,
        device=rewards.device,
    )
    w_sorted = torch.cumsum(gaps / survivors, dim=0)
    w_centered = w_sorted - w_sorted.mean()

    advantages = torch.zeros_like(rewards)
    advantages[sort_indices] = w_centered
    return advantages


def maxrl_normalized_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Continuous MaxRL with per-group std normalization after centering."""
    centered = maxrl_unnormalized_advantage(rewards)
    std = centered.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(centered)
    return centered / (std + eps)


def compute_advantage_1d(
    rewards: torch.Tensor,
    method: AdvantageMethod,
    eps: float = 1e-8,
) -> torch.Tensor:
    if method == "grpo":
        return grpo_advantage(rewards, eps=eps)
    if method == "maxrl_unnorm":
        return maxrl_unnormalized_advantage(rewards)
    if method == "maxrl_norm":
        return maxrl_normalized_advantage(rewards, eps=eps)
    if method == "reinforce_baseline":
        return reinforce_baseline_advantage(rewards)
    raise ValueError(f"Unknown advantage method: {method}")


def compute_grouped_advantages(
    rewards: torch.Tensor,
    method: AdvantageMethod,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-group advantages.

    Args:
        rewards: [num_prompts, group_size]
    Returns:
        advantages: [num_prompts, group_size]
    """
    if rewards.ndim != 2:
        raise ValueError(f"Expected 2D rewards tensor, got shape {tuple(rewards.shape)}")

    if method == "reinforce_baseline":
        return rewards - rewards.mean(dim=1, keepdim=True)

    if method == "grpo":
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards - mean) / (std + eps)
        zero_std = std < eps
        if zero_std.any():
            advantages = torch.where(zero_std, torch.zeros_like(advantages), advantages)
        return advantages

    advantages = torch.zeros_like(rewards)
    for row_idx in range(rewards.shape[0]):
        advantages[row_idx] = compute_advantage_1d(rewards[row_idx], method=method, eps=eps)
    return advantages


def compute_advantages_from_group_indices(
    rewards: torch.Tensor,
    group_indices: torch.Tensor,
    method: AdvantageMethod,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute advantages when batch is flattened and grouped by prompt index.

    Args:
        rewards: [batch]
        group_indices: [batch] integer prompt ids
    Returns:
        advantages: [batch]
    """
    if rewards.ndim != 1:
        raise ValueError(f"Expected 1D rewards, got shape {tuple(rewards.shape)}")
    if group_indices.ndim != 1:
        raise ValueError(
            f"Expected 1D group indices, got shape {tuple(group_indices.shape)}"
        )
    if rewards.shape[0] != group_indices.shape[0]:
        raise ValueError("rewards and group_indices must have equal length")

    advantages = torch.zeros_like(rewards)
    unique_ids = torch.unique(group_indices, sorted=True)
    for group_id in unique_ids.tolist():
        group_mask = group_indices == group_id
        advantages[group_mask] = compute_advantage_1d(
            rewards[group_mask],
            method=method,
            eps=eps,
        )
    return advantages


def broadcast_sequence_advantages_to_tokens(
    sequence_advantages: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Broadcast one scalar advantage per sequence to all response tokens.

    Args:
        sequence_advantages: [batch]
        token_mask: optional [batch, seq_len] mask (1 for active tokens)
    Returns:
        token_advantages: [batch, seq_len]
    """
    if sequence_advantages.ndim != 1:
        raise ValueError(
            f"Expected 1D sequence advantages, got shape {tuple(sequence_advantages.shape)}"
        )

    if token_mask is None:
        return sequence_advantages.unsqueeze(1)

    if token_mask.ndim != 2:
        raise ValueError(f"Expected 2D token mask, got shape {tuple(token_mask.shape)}")
    if token_mask.shape[0] != sequence_advantages.shape[0]:
        raise ValueError("token_mask batch dimension must match sequence_advantages")

    expanded = sequence_advantages.unsqueeze(1).expand_as(token_mask)
    return expanded * token_mask.to(dtype=sequence_advantages.dtype)
