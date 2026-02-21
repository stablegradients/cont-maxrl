"""
Pure per-group advantage functions. No verl dependency.

Each function takes a 1D rewards tensor of shape [G] (one scalar reward per
completion for a single prompt) and returns advantages of shape [G].

See spec §4 for full mathematical derivation and worked examples.
"""
from __future__ import annotations

import torch


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    GRPO: z-score normalization within the group.

    A_i = (r_i - mean(r)) / (std(r) + eps)

    If all rewards are identical (std < eps), returns all zeros.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


def maxrl_unnormalized_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Continuous MaxRL gap-over-survivors advantage (unnormalized).

    Algorithm:
      1. Sort rewards to get order statistics r_(1) <= ... <= r_(G)
      2. Compute gaps: gap_k = r_(k) - r_(k-1)  (gap_1 = r_(1) - 0)
      3. Survivors at rank k: G - k + 1
      4. Cumulative weight: w_(k) = cumsum(gaps / survivors)
      5. Mean-center: A_(k) = w_(k) - mean(w)
      6. Scatter back to original order

    Args:
        rewards: shape [G] -- one scalar reward per completion

    Returns:
        advantages: shape [G] -- per-completion advantage, mean-centered
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)

    # Step 1: Sort to get order statistics
    sorted_rewards, sort_indices = torch.sort(rewards)

    # Step 2: Gaps between consecutive order statistics
    prev = torch.cat([
        torch.zeros(1, dtype=rewards.dtype, device=rewards.device),
        sorted_rewards[:-1],
    ])
    gaps = sorted_rewards - prev  # shape [G]

    # Step 3: Survivors at each rank: [G, G-1, ..., 1]
    survivors = torch.arange(
        G, 0, -1, dtype=rewards.dtype, device=rewards.device
    )

    # Step 4: Cumulative weight = cumsum(gaps / survivors)
    w_sorted = torch.cumsum(gaps / survivors, dim=0)

    # Step 5: Mean-center
    w_centered = w_sorted - w_sorted.mean()

    # Step 6: Scatter back to original order
    advantages = torch.zeros_like(rewards)
    advantages[sort_indices] = w_centered

    return advantages


def maxrl_normalized_advantage(
    rewards: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Continuous MaxRL gap-over-survivors, mean-centered, then std-normalized.

    Identical to unnormalized, but divided by std after centering.
    Dampens signal on prompts with high reward variance.
    """
    centered = maxrl_unnormalized_advantage(rewards)
    std = centered.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(centered)
    return centered / (std + eps)


def reinforce_baseline_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    REINFORCE with per-group mean baseline.

    A_i = r_i - mean(r)
    """
    return rewards - rewards.mean()
