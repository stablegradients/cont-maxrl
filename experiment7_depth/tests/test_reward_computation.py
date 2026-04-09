"""Tests for dense reward computation formula."""

import torch
import pytest


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_reward_formula_exact_match(K):
    targets = torch.tensor([5, min(50, K - 5), K - 5])
    samples = targets.clone()
    errors = (samples - targets).abs().float()
    rewards = 1.0 - errors / (K - 1)
    assert torch.allclose(rewards, torch.ones(3))


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_reward_max_distance(K):
    targets = torch.tensor([0])
    samples = torch.tensor([K - 1])
    errors = (samples - targets).abs().float()
    rewards = 1.0 - errors / (K - 1)
    assert rewards.item() == pytest.approx(0.0)


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_dense_reward_shape(K):
    B, H, W, G = 2, 8, 8, 16
    samples = torch.randint(0, K, (B, H, W, G))
    targets = torch.randint(0, K, (B, H, W))
    errors = (samples - targets.unsqueeze(-1)).abs().float()
    rewards = 1.0 - errors / (K - 1)
    assert rewards.shape == (B, H, W, G)
    assert (rewards >= 0).all() and (rewards <= 1).all()


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_reward_symmetry(K):
    """Reward is symmetric: |a - b| == |b - a|."""
    targets = torch.tensor([K // 2])
    dist = 5 if K > 10 else 1
    s1 = torch.tensor([K // 2 + dist])
    s2 = torch.tensor([K // 2 - dist])
    r1 = 1.0 - (s1 - targets).abs().float() / (K - 1)
    r2 = 1.0 - (s2 - targets).abs().float() / (K - 1)
    assert torch.allclose(r1, r2)
