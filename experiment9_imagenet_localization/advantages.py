
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


GOS_SURVIVAL_EPS: float = 1e-4
"""Survival-floor ε for the GOS increment mask.

At each quality-level interval j in the GOS sum, the empirical survival
fraction is (N − j) / N.  Intervals where this fraction < ε are masked
(their increment is zeroed), mirroring exp_iou_ord_ce_loss's
clamp(min=ε) on P(IoU > τ): thresholds where P(IoU > τ) ≤ ε contribute
zero gradient in the supervised loss, so the RL estimator must skip them
too.  In the N → ∞ limit the mask converges to 𝟙(S(τ) ≥ ε), and the
GOS gradient converges to ∇ L_ε exactly.

Practical effect: at ε = 1e-4, the mask only binds when N > 1/ε = 10000;
at N = 16384 it zeros a single increment (survivors = 1), at N = 131072
about 13 increments.  For N ≤ 10000 the mask is all-True (no-op).
"""


def gos_advantage(
    rewards: torch.Tensor,
    survival_eps: float = GOS_SURVIVAL_EPS,
) -> torch.Tensor:
    """Gap-Over-Survivors (GOS) advantage for continuous rewards.

    Operates on the last dim: (N,) → (N,) or (*batch, N) → (*batch, N).

    Algorithm (per-row, last dim):
      1. Sort rewards ascending.
      2. Gaps: gap_i = r_(i) - r_(i-1), with r_(0) = 0.
      3. Increments: gap_i / survivors_i, where survivors_i = N − i + 1.
         Increments at positions where the empirical survival fraction
         survivors_i / N < survival_eps are zeroed (masked out).
         This is the exact finite-N analog of exp_iou_ord_ce_loss's
         clamp(min=eps) on P(IoU > τ): both omit quality levels where
         the survival probability is below ε.
      4. Cumulative sum, scale by N, mean-center.
      5. Scatter back to original order.

    Args:
        rewards: tensor shaped (*batch, N) of reward values in [0, 1].
        survival_eps: floor on the empirical survival fraction.  Intervals
            where (N − j) / N < survival_eps are masked out.  Must match
            the ``eps`` used in ``localization_exp_iou_ord_ce_loss`` for
            the N → ∞ identity to hold.  Default 1e-4.

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

    # Survivors: [N, N-1, ..., 1].  No clamp on the counts themselves.
    survivors = torch.arange(
        N, 0, -1, device=rewards.device, dtype=rewards.dtype,
    )
    increments = gaps / survivors

    # Mask out quality-level intervals where the empirical survival
    # fraction drops below ε, matching exp_iou_ord_ce_loss's clamp.
    # survivors is shape (N,); broadcasts against (*batch, N).
    if survival_eps > 0:
        mask = (survivors / N >= survival_eps)          # (N,) bool
        increments = increments * mask.to(increments.dtype)

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
