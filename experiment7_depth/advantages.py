import torch


def continuous_maxrl_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Gap-over-survivors advantage (GOS), vectorized over arbitrary leading dims.

    Args:
        rewards: (..., G) tensor of rewards in [0, 1]
    Returns:
        advantages: (..., G) mean-centered per-instance (last dim)

    Preserves the xG scaling that compensates for 1/G averaging in the PG loss.
    """
    G = rewards.shape[-1]
    if G <= 1:
        return torch.zeros_like(rewards)

    sorted_rewards, sort_indices = torch.sort(rewards, dim=-1)
    zeros = torch.zeros_like(sorted_rewards[..., :1])
    prev = torch.cat([zeros, sorted_rewards[..., :-1]], dim=-1)
    gaps = sorted_rewards - prev                                 # (..., G)
    # survivors = [G, G-1, ..., 1]
    survivors = torch.arange(G, 0, -1, device=rewards.device, dtype=rewards.dtype)  # (G,)
    increments = gaps / survivors
    weights = torch.cumsum(increments, dim=-1)                   # (..., G)

    # CRITICAL: xG scaling (same as UTKFace)
    weights = weights * G

    # Mean-center per instance (along last dim)
    weights = weights - weights.mean(dim=-1, keepdim=True)

    # Unsort back to original order
    advantages = torch.zeros_like(rewards)
    advantages.scatter_(-1, sort_indices, weights)
    return advantages


def continuous_maxrl_no_baseline(rewards: torch.Tensor) -> torch.Tensor:
    """GOS without mean-centering."""
    G = rewards.shape[-1]
    if G <= 1:
        return torch.zeros_like(rewards)
    sorted_rewards, sort_indices = torch.sort(rewards, dim=-1)
    zeros = torch.zeros_like(sorted_rewards[..., :1])
    prev = torch.cat([zeros, sorted_rewards[..., :-1]], dim=-1)
    gaps = sorted_rewards - prev
    survivors = torch.arange(G, 0, -1, device=rewards.device, dtype=rewards.dtype)
    increments = gaps / survivors
    weights = torch.cumsum(increments, dim=-1) * G
    advantages = torch.zeros_like(rewards)
    advantages.scatter_(-1, sort_indices, weights)
    return advantages


def binary_maxrl_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Binary MaxRL: success = exact match (reward == 1.0).

    Formula per instance with K_count = num_successes:
        if K_count > 0: success -> (G - K_count) / K_count,  failure -> -1
        if K_count == 0: all advantages = 0
    """
    G = rewards.shape[-1]
    if G <= 1:
        return torch.zeros_like(rewards)

    successes = (rewards == 1.0)                                      # (..., G) bool
    K_counts = successes.sum(dim=-1, keepdim=True).to(rewards.dtype)  # (..., 1)

    # Broadcast-safe advantage computation
    success_adv = torch.where(
        K_counts > 0,
        (G - K_counts) / K_counts.clamp(min=1.0),
        torch.zeros_like(K_counts),
    )                                                             # (..., 1)
    failure_adv = torch.full_like(rewards, -1.0)                  # (..., G)

    advantages = torch.where(
        successes,
        success_adv.expand_as(rewards),
        failure_adv,
    )

    # Zero out instances with K_counts == 0
    has_success = (K_counts > 0).expand_as(rewards)
    advantages = torch.where(has_success, advantages, torch.zeros_like(advantages))
    return advantages


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO z-score per instance, vectorized."""
    G = rewards.shape[-1]
    if G <= 1:
        return torch.zeros_like(rewards)
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True)
    normalized = (rewards - mean_r) / (std_r + eps)
    # Zero out degenerate instances (std too small)
    return torch.where(std_r < 1e-4, torch.zeros_like(rewards), normalized)


def reinforce_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Mean-centered REINFORCE, vectorized."""
    G = rewards.shape[-1]
    if G <= 1:
        return torch.zeros_like(rewards)
    return rewards - rewards.mean(dim=-1, keepdim=True)


ADVANTAGE_FNS = {
    "cont_maxrl": continuous_maxrl_advantages,
    "cont_maxrl_no_bl": continuous_maxrl_no_baseline,
    "binary_maxrl": binary_maxrl_advantages,
    "grpo": grpo_advantages,
    "reinforce": reinforce_advantages,
}
