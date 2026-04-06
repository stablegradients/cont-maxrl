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

    # Scale by G so the 1/G in the policy gradient loss doesn't shrink the gradient
    w_sorted = w_sorted * G

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


def rloo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    RLOO: leave-one-out baseline (Ahmadian et al. 2024, Kool et al. 2019).

    A_i = r_i - (1/(G-1)) * sum_{j != i} r_j
        = G/(G-1) * (r_i - mean(r))

    Scale by G so the 1/G in the policy gradient loss doesn't shrink the
    gradient (same correction as maxrl_unnormalized_advantage line 73).

    For G=1, returns zeros (no leave-one-out possible).
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    return (G / (G - 1)) * (rewards - mean) * G


def binary_maxrl_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary MaxRL estimator from Tajwar et al. 2026 (arXiv:2602.02710).

    Formula: A_i = (r_i - mean(r)) / mean(r)

    On binary rewards r_i in {0,1} with C successes out of G:
      success (r_i=1): (1 - C/G) / (C/G) = (G-C)/C
      failure (r_i=0): -(C/G) / (C/G)    = -1

    This is algebraically identical to GOS (maxrl_unnormalized_advantage)
    on binary rewards. This identity is the binary-recovery proposition
    in the paper.

    Included as a baseline (Tajwar et al.), not as our method. GOS
    (maxrl_unnormalized_advantage) is the paper's contribution.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    if mean < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / mean


def dg_advantage_per_token(
    rewards: torch.Tensor,           # [G] per-rollout scalar reward
    old_log_probs: torch.Tensor,     # [G, T] per-token log π_old
    response_mask: torch.Tensor,     # [G, T]
    eta: float = 1.0,
    surprisal_clip: float = 20.0,
) -> torch.Tensor:
    """
    Delightful Policy Gradient per-token advantage (Osband 2026, arXiv:2603.14608).

    U_tau  = r_tau - mean(r_group)                       # REINFORCE baseline, [G]
    ell_t  = clip(-log pi_old(a_t), 0, surprisal_clip)   # surprisal, [G, T]
    w_t    = sigmoid(U_tau * ell_t / eta)                # delight gate, [G, T]
    coef_t = w_t * U_tau                                 # effective coef, [G, T]

    Args:
        rewards: [G] scalar reward per rollout
        old_log_probs: [G, T] log pi_old(a_t) at each response token
        response_mask: [G, T] 1 for response tokens, 0 for pads
        eta: gate temperature (paper uses 1.0)
        surprisal_clip: max |log pi| before clipping, in nats

    Returns:
        advantages: [G, T], already masked (pads are 0)
    """
    G, T = old_log_probs.shape
    if G <= 1:
        return torch.zeros(G, T, dtype=rewards.dtype, device=rewards.device)

    # Per-rollout advantage: mean-centered, no std. [G]
    U = rewards - rewards.mean()

    # Surprisal from OLD log-probs, clipped. [G, T]
    # -log pi in [0, +inf); clip at 20 nats => pi >= ~2e-9.
    ell = (-old_log_probs).clamp(min=0.0, max=surprisal_clip)
    ell = ell * response_mask  # zero out pad positions

    # Delight and gate, per-token. [G, T]
    U_bcast = U.unsqueeze(-1)          # [G, 1]
    chi = U_bcast * ell                 # [G, T]
    w = torch.sigmoid(chi / eta)        # [G, T], in (0, 1)

    coef = w * U_bcast                  # [G, T]

    return coef * response_mask
