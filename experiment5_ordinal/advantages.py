"""Advantage estimators for ordinal regression RL methods.

All functions operate on per-image rollouts (shape [N]) and return
mean-centered advantages (shape [N]).

Methods:
- continuous_maxrl: Gap-over-survivors (our method)
- binary_maxrl: Tajwar et al. binarized at tau=1.0
- grpo: z-score normalized expected reward baseline
- reinforce: mean-centered baseline (no std normalization)
"""

import torch


def continuous_maxrl_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Gap-over-survivors advantage for continuous rewards.

    Args:
        rewards: (N,) tensor of reward values in {0, 1/(L-1), ..., 1}

    Returns:
        advantages: (N,) mean-centered advantages
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards)

    # Gaps: gap_i = r_(i) - r_(i-1), with r_(0) = 0
    prev = torch.cat([torch.zeros(1, device=rewards.device), sorted_rewards[:-1]])
    gaps = sorted_rewards - prev                    # (N,)
    survivors = torch.arange(N, 0, -1, device=rewards.device).float()  # [N, N-1, ..., 1]

    # Cumulative gap-over-survivors weight
    increments = gaps / survivors                   # (N,)
    weights = torch.cumsum(increments, dim=0)       # (N,)

    # Scale by N so that (1/N)*sum(w_i * score_i) has O(1) magnitude.
    # Without this, weights are O(H_G/G) each, and the 1/N in the policy
    # gradient loss introduces an extra 1/G factor that makes the gradient vanish.
    weights = weights * N

    # Mean-center
    weights = weights - weights.mean()

    # Unsort back to original order
    advantages = torch.zeros(N, device=rewards.device)
    advantages[sort_indices] = weights

    return advantages


def binary_maxrl_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Binary MaxRL advantages (Tajwar et al., binarized at tau=1.0).

    Success = exact match (r = 1.0). Failure = anything else.

    Args:
        rewards: (N,) tensor of reward values

    Returns:
        advantages: (N,) mean-centered advantages
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    successes = (rewards == 1.0)
    K = successes.sum().item()

    advantages = torch.zeros(N, device=rewards.device)

    if K == 0:
        return advantages  # all zeros

    # From Tajwar et al., scaled by N so (1/N)*sum is O(1):
    # Original: success = +(N-K)/(K*N), failure = -1/N
    # Scaled:   success = +(N-K)/K,     failure = -1
    advantages[successes] = (N - K) / K
    advantages[~successes] = -1.0

    return advantages


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantage: z-score normalized.

    Args:
        rewards: (N,) tensor of reward values
        eps: numerical stability constant

    Returns:
        advantages: (N,) z-score normalized advantages
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    mean_r = rewards.mean()
    std_r = rewards.std()

    if std_r < 1e-4:
        return torch.zeros_like(rewards)

    return (rewards - mean_r) / (std_r + eps)


def reinforce_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE with baseline: mean-centered, no std normalization.

    Args:
        rewards: (N,) tensor of reward values

    Returns:
        advantages: (N,) mean-centered advantages
    """
    N = rewards.numel()
    if N <= 1:
        return torch.zeros_like(rewards)

    return rewards - rewards.mean()


# Dispatch dict for convenience
ADVANTAGE_FNS = {
    "cont_maxrl": continuous_maxrl_advantages,
    "binary_maxrl": binary_maxrl_advantages,
    "grpo": grpo_advantages,
    "reinforce": reinforce_advantages,
}
