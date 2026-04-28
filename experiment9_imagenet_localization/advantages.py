
"""Advantage estimators for ImageNet localization RL experiment.

All functions are **polymorphic over the last tensor dimension**:

    rewards.shape == (N,)      -> advantages.shape == (N,)
    rewards.shape == (B, N)    -> advantages.shape == (B, N)
    rewards.shape == (*batch, N) -> advantages.shape == (*batch, N)

The per-image loop over the batch dim (Python for b in range(B)) is
therefore unnecessary — just call `advantage_fn(rewards)` once with the
full (B, N) tensor. This matters at large N (e.g. N=16384): the dispatch
overhead of 128 sequential torch calls per step becomes the bottleneck.

Methods:
- gos: Gap-Over-Survivors (our method, formerly continuous_maxrl)
- binary_maxrl: Tajwar et al. binarized at r = 1.0 (exact equality)
- grpo: z-score normalized expected reward baseline
- rloo: Leave-One-Out baseline
- reinforce: mean-centered baseline (no std normalization)
- dg: Delightful Policy Gradient (sigmoid-sharpened)
"""

import torch


SURVIVAL_FLOOR_EPS: float = 1e-4
"""Clamp the empirical (1 - F̂(τ)) at this floor inside gos_advantage —
mirrors localization_exp_iou_ord_ce_loss's clamp(min=1e-4) on P(IoU > τ).
The cumulative-hazard integrand 1/(1 − F̂) is then bounded at 1/ε = 1e4,
which keeps gos's per-step gradient scale comparable to J_ord at any N.
"""


def gos_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Gap-Over-Survivors (GOS) advantage for continuous rewards.

    Operates on the last dim: (N,) → (N,) or (*batch, N) → (*batch, N).

    Algorithm (per-row, last dim):
      1. Sort rewards ascending.
      2. Gaps: gap_i = r_(i) - r_(i-1), with r_(0) = 0.
      3. Cumulative Σ (gap_i / max(N - i + 1, ε·N)); scale by N.
         The clamp on the survivor count is the empirical analog of
         ``localization_exp_iou_ord_ce_loss``'s ``clamp(min=1e-4)`` on
         ``P(IoU > τ)`` — ensures the cumulative-hazard integrand is
         bounded by 1/ε.
      4. Mean-center along the last dim.
      5. Scatter back to original order.

    Args:
        rewards: tensor shaped (*batch, N) of reward values in [0, 1].

    Returns:
        Mean-centered advantages, same shape / dtype / device as input.
        Returns zeros when the last dim is empty or of size ≤ 1.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards, dim=-1, stable=True)

    # Shift sorted_rewards right by 1 along last dim, padding with 0.
    batch_shape = sorted_rewards.shape[:-1]
    zero_col = torch.zeros(
        *batch_shape, 1, device=rewards.device, dtype=rewards.dtype,
    )
    prev = torch.cat([zero_col, sorted_rewards[..., :-1]], dim=-1)
    gaps = sorted_rewards - prev

    # survivors shape (N,) broadcasts against (*batch, N). Clamp at N·ε so
    # 1/(1-F̂) ≤ 1/ε, matching exp_iou_ord_ce's clamp on P(IoU > τ).
    min_surv = max(1.0, float(N) * SURVIVAL_FLOOR_EPS)
    survivors = torch.arange(
        N, 0, -1, device=rewards.device, dtype=rewards.dtype,
    ).clamp(min=min_surv)
    increments = gaps / survivors
    weights = torch.cumsum(increments, dim=-1) * N
    weights = weights - weights.mean(dim=-1, keepdim=True)

    advantages = torch.zeros_like(rewards)
    advantages.scatter_(dim=-1, index=sort_indices, src=weights)
    return advantages


def binary_maxrl_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Binary MaxRL advantage (Tajwar et al.), binarized at r == 1.0 (exact).

    Operates on the last dim. Success = exact match (r == 1.0); failure = anything
    else. For rows where all samples succeed or all fail, returns zeros.

    Scaled so that (1/N)·Σ adv_i · score_i has O(1) magnitude:
      success advantage = +(N - K) / K
      failure advantage = -1

    Args:
        rewards: tensor shaped (*batch, N).

    Returns:
        Advantages, same shape / dtype / device as input.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    successes = (rewards == 1.0)                                   # (*, N) bool
    K = successes.sum(dim=-1, keepdim=True).to(rewards.dtype)      # (*, 1)

    safe_K = K.clamp(min=1)                                         # avoid div-by-0
    adv_success = (N - K) / safe_K                                 # (*, 1)

    advantages = torch.where(
        successes,
        adv_success.expand_as(rewards),
        torch.full_like(rewards, -1.0),
    )
    degenerate = (K == 0) | (K == N)                               # (*, 1) bool
    return torch.where(
        degenerate.expand_as(rewards),
        torch.zeros_like(rewards),
        advantages,
    )


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: z-score normalized rewards.

    Operates on the last dim. Rows with near-zero std (< 1e-4) get zero
    advantages to avoid division instability.

    Args:
        rewards: tensor shaped (*batch, N).
        eps: numerical stability constant added to std denominator.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    normalized = (rewards - mean) / (std + eps)
    zero_mask = (std < 1e-4).expand_as(rewards)
    return torch.where(zero_mask, torch.zeros_like(rewards), normalized)


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-One-Out (RLOO) baseline advantage.

    A_i = (N · r_i - Σ r) / (N - 1)

    Operates on the last dim. Returns zeros when the last dim has ≤ 1 element.
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    total = rewards.sum(dim=-1, keepdim=True)
    return (N * rewards - total) / (N - 1)


def reinforce_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with baseline: mean-centered along the last dim.

    A_i = r_i - mean(r)
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    return rewards - rewards.mean(dim=-1, keepdim=True)


def dg_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Delightful Policy Gradient (DG) advantage.

    Sharpens reward differences via sigmoid with temperature 0.1 along the
    last dim:
      s_i = sigmoid((r_i - mean(r)) / 0.1)
      A_i = s_i - mean(s)
    """
    if rewards.numel() == 0:
        return torch.zeros_like(rewards)
    N = rewards.shape[-1]
    if N <= 1:
        return torch.zeros_like(rewards)

    mean_r = rewards.mean(dim=-1, keepdim=True)
    s = torch.sigmoid((rewards - mean_r) / 0.1)
    return s - s.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Dispatch dict for convenience.
# Keys in the canonical order for this experiment.
# ---------------------------------------------------------------------------

ADVANTAGE_FNS = {
    "gos": gos_advantage,
    "binary_maxrl": binary_maxrl_advantage,
    "grpo": grpo_advantage,
    "rloo": rloo_advantage,
    "reinforce": reinforce_advantage,
    "dg": dg_advantage,
}
