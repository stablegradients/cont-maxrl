"""Tests for binary MaxRL structural collapse at large K."""

import torch
import pytest


def test_uniform_policy_exact_match_at_large_K():
    """Uniform policy at K=1000 should have exact-match rate ~1/K."""
    K = 1000
    B, H, W, G = 1, 32, 32, 256
    torch.manual_seed(0)
    uniform_probs = torch.full((B * H * W, K), 1.0 / K)
    samples = torch.multinomial(uniform_probs, G, replacement=True).view(B, H, W, G)
    targets = torch.randint(0, K, (B, H, W))
    errors = (samples - targets.unsqueeze(-1)).abs().float()
    rewards = 1.0 - errors / (K - 1)
    exact_rate = (rewards == 1.0).float().mean().item()
    assert exact_rate < 0.005  # near 1/1000 = 0.001


def test_uniform_policy_exact_match_at_small_K():
    """Uniform policy at K=10 should have exact-match rate ~1/K."""
    K = 10
    B, H, W, G = 1, 32, 32, 256
    torch.manual_seed(0)
    uniform_probs = torch.full((B * H * W, K), 1.0 / K)
    samples = torch.multinomial(uniform_probs, G, replacement=True).view(B, H, W, G)
    targets = torch.randint(0, K, (B, H, W))
    errors = (samples - targets.unsqueeze(-1)).abs().float()
    rewards = 1.0 - errors / (K - 1)
    exact_rate = (rewards == 1.0).float().mean().item()
    assert 0.05 < exact_rate < 0.15  # near 1/10 = 0.1


def test_binary_maxrl_zero_advantages_at_large_K():
    """At K=1000 with uniform policy, binary MaxRL should produce ~all-zero advantages."""
    from experiment7_depth.advantages import binary_maxrl_advantages
    K = 1000
    G = 64
    torch.manual_seed(0)
    rewards = torch.rand(100, G) * 0.99  # all < 1.0
    adv = binary_maxrl_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv))
