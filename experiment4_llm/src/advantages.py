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


def gos_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    GOS (gap-over-survivors) advantage — formerly `maxrl_unnormalized_advantage`
    / adv_estimator `maxrl_unnorm`; renamed 2026-06-13, math unchanged.

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

    # NOTE: no *G scaling (removed 2026-06-26). The earlier *G was inert under
    # AdamW — Adam's second-moment normalization absorbs any uniform gradient
    # scaling, so *G did not change the effective LR — and it is absent from
    # reference maxrl (compute_maxrl_outcome_advantage = (r-mean)/mean). Dropping
    # it puts gos/rloo/grpo on a common scale so grad_clip acts uniformly.

    # Step 5: Mean-center
    w_centered = w_sorted - w_sorted.mean()

    # Step 6: Scatter back to original order
    advantages = torch.zeros_like(rewards)
    advantages[sort_indices] = w_centered

    return advantages


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

    For G=1, returns zeros (no leave-one-out possible). The earlier *G scaling
    was removed 2026-06-26 (inert under AdamW; absent from reference maxrl) so
    this is now the standard leave-one-out advantage.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    return (G / (G - 1)) * (rewards - mean)


def binary_maxrl_advantage(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary MaxRL estimator from Tajwar et al. 2026 (arXiv:2602.02710).

    Formula: A_i = (r_i - mean(r)) / mean(r)

    On binary rewards r_i in {0,1} with C successes out of G:
      success (r_i=1): (1 - C/G) / (C/G) = (G-C)/C
      failure (r_i=0): -(C/G) / (C/G)    = -1

    This is algebraically identical to GOS (gos_advantage) on binary
    rewards. This identity is the binary-recovery proposition in the paper.

    Included as a baseline (Tajwar et al.), not as our method. GOS
    (gos_advantage) is the paper's contribution.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    if mean < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / mean


# ---------------------------------------------------------------------------
# PKPO (Walder & Karkhanis 2025, arXiv:2505.15201) — direct port of Listing 1
# ---------------------------------------------------------------------------
import math


def _pkpo_m_normed(N: int, K: int, i: int, j: int) -> float:
    """Normalized weight m_{ij} / C(N,K). Paper Listing 1, _m_normed."""
    if i == j and i >= K - 1:
        # numpy: K/(N-K+1) * np.prod(np.arange(i-K+2, i+1) / np.arange(N-K+2, N+1))
        num = 1.0
        den = 1.0
        for a in range(i - K + 2, i + 1):
            num *= a
        for a in range(N - K + 2, N + 1):
            den *= a
        return (K / (N - K + 1)) * (num / den)
    if j > i and j >= K - 1 and K >= 2:
        # numpy: K/(N-K+1) * (K-1)/N * np.prod(np.arange(j-K+2, j) / np.arange(N-K+2, N))
        num = 1.0
        den = 1.0
        for a in range(j - K + 2, j):
            num *= a
        for a in range(N - K + 2, N):
            den *= a
        return (K / (N - K + 1)) * ((K - 1) / N) * (num / den)
    return 0.0


def _pkpo_m_diagonal(N: int, K: int, dtype, device) -> torch.Tensor:
    return torch.tensor(
        [_pkpo_m_normed(N, K, i, i) for i in range(N)],
        dtype=dtype, device=device,
    )


def _pkpo_deltas(N: int, K: int, dtype, device) -> torch.Tensor:
    # Listing 1: returns length N-2, not N-1.
    return torch.tensor(
        [
            _pkpo_m_normed(N - 1, K, i, i + 1)
            - _pkpo_m_normed(N - 1, K, i + 1, i + 1)
            for i in range(N - 2)
        ],
        dtype=dtype, device=device,
    )


def _pkpo_s_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    """Eq. (19), the raw s_i. g_sorted must already be ascending."""
    N = g_sorted.numel()
    dtype, device = g_sorted.dtype, g_sorted.device
    diag = _pkpo_m_diagonal(N, K, dtype, device)            # [N]
    c = (g_sorted * diag).clone()                            # [N]
    if N >= 2:
        deltas = _pkpo_deltas(N + 1, K, dtype, device)      # [N-1]
        c[: N - 1] = c[: N - 1] + g_sorted[1:] * deltas
    return torch.flip(torch.cumsum(torch.flip(c, [0]), dim=0), [0])


def _pkpo_b_sorted(g_sorted: torch.Tensor, K: int) -> torch.Tensor:
    """Baseline b used by sloo and sloo_minus_one. g_sorted ascending."""
    N = g_sorted.numel()
    dtype, device = g_sorted.dtype, g_sorted.device
    if N <= 1 or K < 1:
        return torch.zeros(N, dtype=dtype, device=device)
    diag_nm1 = _pkpo_m_diagonal(N - 1, K, dtype, device)    # [N-1]
    arange_1_N = torch.arange(1, N, dtype=dtype, device=device)
    w = (diag_nm1 * arange_1_N).clone()                      # [N-1]
    if N >= 3:
        deltas = _pkpo_deltas(N, K, dtype, device)          # [N-2]
        arange_1_Nm1 = torch.arange(1, N - 1, dtype=dtype, device=device)
        w[1:] = w[1:] + deltas * arange_1_Nm1
    c1 = (w * g_sorted[1:]).sum().unsqueeze(0)
    c2 = (g_sorted[:-1] - g_sorted[1:]) * w
    return torch.cumsum(torch.cat([c1, c2], dim=0), dim=0)


def pkpo_advantage(
    rewards: torch.Tensor,
    k_opt: int = 8,
    variant: str = "loo_minus_one",
) -> torch.Tensor:
    """Pass@K Policy Optimization (Walder & Karkhanis 2025, arXiv:2505.15201).

    Reward transformation [G] -> [G] unbiased for grad max_g@k_opt.
    See PKPO_IMPLEMENTATION.md for full spec.
    """
    G = rewards.numel()
    if G <= 1:
        return torch.zeros_like(rewards)
    if not (1 <= k_opt <= G):
        raise ValueError(f"pkpo_advantage: k_opt={k_opt} must be in [1, G={G}]")
    if variant not in ("raw", "loo", "loo_minus_one"):
        raise ValueError(f"pkpo_advantage: unknown variant '{variant}'")

    sorted_rewards, sort_idx = torch.sort(rewards)

    if k_opt == 1:
        s_sorted = sorted_rewards - sorted_rewards.mean()
    else:
        s_sorted = _pkpo_s_sorted(sorted_rewards, k_opt)
        if variant == "raw":
            pass
        elif variant == "loo":
            b = torch.zeros(G, dtype=rewards.dtype, device=rewards.device)
            for i in range(G):
                mask = torch.ones(G, dtype=torch.bool, device=rewards.device)
                mask[i] = False
                g_minus = sorted_rewards[mask]
                k_minus = min(k_opt, G - 1)
                s_minus = _pkpo_s_sorted(g_minus, k_minus)
                b[i] = s_minus.sum() / (G - 1)
            s_sorted = s_sorted - b
        elif variant == "loo_minus_one":
            b = _pkpo_b_sorted(sorted_rewards, k_opt - 1)
            s_sorted = s_sorted - b * k_opt / (k_opt - 1) / G

    s_sorted = s_sorted * G

    advantages = torch.zeros_like(rewards)
    advantages[sort_idx] = s_sorted
    return advantages
