"""Advantage estimators for ImageNet localization RL experiment.

All functions operate on per-image rollouts (shape [N]) and return
mean-centered advantages (shape [N]).

Methods:
- gos: Gap-Over-Survivors (our method, formerly continuous_maxrl)
- binary_maxrl: Tajwar et al. binarized at r = 1.0 (exact equality)
- grpo: z-score normalized expected reward baseline
- rloo: Leave-One-Out baseline
- reinforce: mean-centered baseline (no std normalization)
- dg: Delightful Policy Gradient (sigmoid-sharpened)
"""

import torch


def gos_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Gap-Over-Survivors (GOS) advantage for continuous rewards.

    Computes the continuous MaxRL / GOS advantage by sorting rewards,
    computing gap-over-survivors increments, taking the cumulative sum,
    scaling by N (to compensate for the 1/N in the policy-gradient mean loss),
    mean-centering, and scattering back to the original order.

    Args:
        rewards: (N,) tensor of reward values in [0, 1]

    Returns:
        advantages: (N,) mean-centered advantages, same dtype and device as input.
                    Returns zeros for N <= 1.
    """
    N = rewards.numel()
    if N == 0:
        return torch.zeros_like(rewards)
    if N <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards, stable=True)

    # Gaps: gap_i = r_(i) - r_(i-1), with r_(0) = 0
    prev = torch.cat([
        torch.zeros(1, device=rewards.device, dtype=rewards.dtype),
        sorted_rewards[:-1],
    ])
    gaps = sorted_rewards - prev                         # (N,)
    survivors = torch.arange(N, 0, -1, device=rewards.device).to(rewards.dtype)  # [N, N-1, ..., 1]

    # Cumulative gap-over-survivors weight
    increments = gaps / survivors                        # (N,)
    weights = torch.cumsum(increments, dim=0)            # (N,)

    # Scale by N so that (1/N)*sum(w_i * score_i) has O(1) magnitude.
    # Without this, the 1/N in the policy-gradient loss causes gradient vanishing.
    weights = weights * N

    # Mean-center
    weights = weights - weights.mean()

    # Unsort back to original order
    advantages = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    advantages[sort_indices] = weights

    return advantages


def binary_maxrl_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Binary MaxRL advantage (Tajwar et al.), binarized at r == 1.0 (exact).

    Success = exact match (r == 1.0). Failure = anything else.
    If all rewards are the same (all success or all failure), returns zeros.

    Scaled by N so that (1/N)*sum(adv_i * score_i) has O(1) magnitude:
      success advantage = +(N - K) / K
      failure advantage = -1

    Args:
        rewards: (N,) tensor of reward values

    Returns:
        advantages: (N,) advantages, same dtype and device as input.
                    Returns zeros for N <= 1 or degenerate (all same).
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    successes = (rewards == 1.0)
    K = successes.sum().item()

    advantages = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    if K == 0 or K == N:
        return advantages  # all zeros — no signal when all same

    advantages[successes] = (N - K) / K
    advantages[~successes] = -1.0

    return advantages


def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: z-score normalized rewards.

    Normalizes rewards to zero mean and unit standard deviation.
    Returns zeros when std < 1e-4 (all rewards nearly equal) to avoid
    division by near-zero.

    Args:
        rewards: (N,) tensor of reward values
        eps: numerical stability constant added to std denominator

    Returns:
        advantages: (N,) z-score normalized advantages, same dtype/device.
                    Returns zeros for N <= 1 or near-constant rewards.
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    mean_r = rewards.mean()
    std_r = rewards.std()

    if std_r < 1e-4:
        return torch.zeros_like(rewards)

    return (rewards - mean_r) / (std_r + eps)


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-One-Out (RLOO) baseline advantage.

    A_i = (N * r_i - sum(r)) / (N - 1)

    Each sample is compared against the mean of all other samples.
    Returns zeros for N == 1 (leave-one-out is undefined).

    Args:
        rewards: (N,) tensor of reward values

    Returns:
        advantages: (N,) RLOO advantages, same dtype and device as input.
                    Returns zeros for N <= 1.
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    total = rewards.sum()
    return (N * rewards - total) / (N - 1)


def reinforce_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with baseline: mean-centered, no std normalization.

    A_i = r_i - mean(r)

    Args:
        rewards: (N,) tensor of reward values

    Returns:
        advantages: (N,) mean-centered advantages, same dtype and device.
                    Returns zeros for N <= 1.
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    return rewards - rewards.mean()


def dg_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """Delightful Policy Gradient (DG) advantage.

    Sharpens reward differences via a sigmoid with temperature 0.1:
      s_i = sigmoid((r_i - mean(r)) / 0.1)
      A_i = s_i - mean(s)

    The sigmoid keeps advantages bounded while amplifying rank differences.
    Returns zeros for N <= 1.

    Args:
        rewards: (N,) tensor of reward values

    Returns:
        advantages: (N,) sigmoid-sharpened centered advantages, same dtype/device.
                    Returns zeros for N <= 1.
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    mean_r = rewards.mean()
    s = torch.sigmoid((rewards - mean_r) / 0.1)
    return s - s.mean()


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
