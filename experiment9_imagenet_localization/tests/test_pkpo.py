"""Tests for pkpo_advantage in experiment9_imagenet_localization.advantages.

Reference: Walder & Karkhanis 2025, NeurIPS, arXiv:2505.15201, Listing 1.

The PyTorch implementation is verified bit-exact (within float64 epsilon)
against the paper's numpy reference for several (N, K) pairs and for
batched inputs.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

# Make sure the package is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiment9_imagenet_localization.advantages import (
    ADVANTAGE_FNS,
    pkpo_advantage,
)


# ---------------------------------------------------------------------------
# Reference implementation — verbatim port of the paper's Listing 1.
# ---------------------------------------------------------------------------

def _ref_m_normed(N, K, i, j):
    if i == j and i >= K - 1:
        return (
            K / (N - K + 1)
            * np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    elif j > i and j >= K - 1 and K >= 2:
        return (
            K / (N - K + 1) * (K - 1) / N
            * np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0.0


def _ref_m_diagonal(N, K):
    return np.array([_ref_m_normed(N, K, i, i) for i in range(N)])


def _ref_delta(N, K, i):
    return _ref_m_normed(N, K, i, i + 1) - _ref_m_normed(N, K, i + 1, i + 1)


def _ref_deltas(N, K):
    return np.array([_ref_delta(N - 1, K, i) for i in range(N - 2)])


def _ref_sorted_apply(func):
    def inner(x, *args, **kwargs):
        i_sort = np.argsort(x)
        out = np.zeros_like(x)
        out[i_sort] = func(x[i_sort], *args, **kwargs)
        return out
    return inner


@_ref_sorted_apply
def _ref_s(g, K):
    N = len(g)
    c = g * _ref_m_diagonal(N, K)
    c[: (N - 1)] += g[1:] * _ref_deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


@_ref_sorted_apply
def _ref_b(g, K):
    N = len(g)
    w = (_ref_m_diagonal(N - 1, K) * np.arange(1, N)).astype(float)
    w[1:] += _ref_deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()])
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def _ref_sloo_minus_one(g, K):
    return _ref_s(g, K) - _ref_b(g, K - 1) * K / (K - 1) / len(g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,K", [
    (4, 2), (8, 4), (16, 8), (32, 16), (64, 16),
    (128, 16), (256, 16), (16, 16), (16, 2),
])
def test_pkpo_matches_paper_reference(N, K):
    """PyTorch impl must equal the numpy reference within float64 epsilon."""
    np.random.seed(42)
    g_np = np.random.uniform(0, 1, size=N)
    ref = _ref_sloo_minus_one(g_np, K)
    out = pkpo_advantage(
        torch.tensor(g_np, dtype=torch.float64), k=K
    ).numpy()
    err = np.abs(ref - out).max()
    assert err < 1e-9, f"N={N} K={K}: max_err={err:.2e}"


def test_pkpo_batched_matches_per_row_reference():
    """Batched (B, N) call must equal per-row numpy reference."""
    np.random.seed(0)
    B, N, K = 8, 64, 16
    g_np = np.random.uniform(0, 1, size=(B, N))
    ref = np.stack([_ref_sloo_minus_one(g_np[b], K) for b in range(B)])
    out = pkpo_advantage(
        torch.tensor(g_np, dtype=torch.float64), k=K
    ).numpy()
    err = np.abs(ref - out).max()
    assert err < 1e-9, f"batched N={N} K={K}: max_err={err:.2e}"


def test_pkpo_default_k_from_env():
    """Calling pkpo_advantage(rewards) without k must use PKPO_K_DEFAULT."""
    from experiment9_imagenet_localization.advantages import PKPO_K_DEFAULT
    np.random.seed(1)
    rewards = torch.rand(32, dtype=torch.float64)
    out_default = pkpo_advantage(rewards)
    out_explicit = pkpo_advantage(rewards, k=PKPO_K_DEFAULT)
    assert torch.allclose(out_default, out_explicit, atol=1e-12)


def test_pkpo_k1_falls_back_to_reinforce():
    """k=1 has K-1=0 in denominator (Eq 33). Paper says k=1 is no transform.
    Our impl returns mean-centered rewards (== reinforce_advantage)."""
    rewards = torch.rand(64)
    out = pkpo_advantage(rewards, k=1)
    expected = rewards - rewards.mean()
    assert torch.allclose(out, expected, atol=1e-6)


def test_pkpo_dispatch_via_advantage_fns():
    """ADVANTAGE_FNS['pkpo'] is callable as fn(rewards) and matches the
    direct call (uses PKPO_K_DEFAULT)."""
    rewards = torch.rand(32)
    out_dispatch = ADVANTAGE_FNS["pkpo"](rewards)
    out_direct = pkpo_advantage(rewards)
    assert torch.allclose(out_dispatch, out_direct, atol=1e-7)


def test_pkpo_shape_dtype_preserved():
    """Output shape == input shape, dtype preserved (no silent upcast)."""
    rewards32 = torch.rand(4, 32, dtype=torch.float32)
    out32 = pkpo_advantage(rewards32, k=8)
    assert out32.shape == (4, 32)
    assert out32.dtype == torch.float32

    rewards64 = torch.rand(4, 32, dtype=torch.float64)
    out64 = pkpo_advantage(rewards64, k=8)
    assert out64.shape == (4, 32)
    assert out64.dtype == torch.float64


def test_pkpo_no_nan_or_inf():
    """No NaN or ±inf advantages on random inputs across several (N, K)."""
    for N, K in [(16, 2), (32, 8), (256, 16), (256, 128)]:
        torch.manual_seed(N + K)
        rewards = torch.rand(8, N)
        out = pkpo_advantage(rewards, k=K)
        assert not torch.isnan(out).any(), f"NaN at N={N} K={K}"
        assert not torch.isinf(out).any(), f"Inf at N={N} K={K}"


def test_pkpo_invalid_k_raises():
    """k outside [1, N] must raise ValueError."""
    rewards = torch.rand(16)
    with pytest.raises(ValueError):
        pkpo_advantage(rewards, k=0)
    with pytest.raises(ValueError):
        pkpo_advantage(rewards, k=17)


def test_pkpo_constant_rewards_zero_advantage():
    """When all rewards are equal, sloo_minus_one weight on each sample
    is the same constant; the advantage is zero (or numerically negligible).

    Reasoning: when g_i are identical, the sort is degenerate and every
    element gets the same effective reward, so the gradient contribution
    is zero in expectation.  The estimator is exact (not just unbiased)
    in this case because every k-subset has the same max.
    """
    rewards = torch.full((4, 32), 0.5)
    out = pkpo_advantage(rewards, k=8)
    assert out.abs().max().item() < 1e-6


def test_pkpo_eq_n_matches_tzsm25_max_form():
    """At k=N, sloo_minus_one reduces to (max(all) - max(without_i)).

    For non-largest samples i: max(all) - max(without_i) = max - max = 0.
    For the largest sample i*:  max(all) - max(without_i*) = top1 - top2.
    This is the special case (k=n) cited as TZSM25 in the paper.
    """
    torch.manual_seed(7)
    N = 8
    rewards = torch.rand(N)
    out = pkpo_advantage(rewards, k=N)

    sorted_rewards, _ = torch.sort(rewards)        # ascending
    largest_idx = torch.argmax(rewards)
    expected_largest = sorted_rewards[-1] - sorted_rewards[-2]   # top1 - top2

    others_mask = torch.ones(N, dtype=torch.bool)
    others_mask[largest_idx] = False
    assert out[others_mask].abs().max().item() < 1e-6
    assert torch.allclose(out[largest_idx], expected_largest, atol=1e-6)
