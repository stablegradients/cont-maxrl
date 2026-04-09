"""Tests for vectorized (..., G) advantage estimators."""

import torch
import pytest
from experiment7_depth.advantages import (
    continuous_maxrl_advantages,
    continuous_maxrl_no_baseline,
    binary_maxrl_advantages,
    grpo_advantages,
    reinforce_advantages,
)


def test_gos_scaling_preserved():
    """Critical xG scaling."""
    rewards = torch.tensor([0.1, 0.5, 0.9, 1.0])
    adv = continuous_maxrl_advantages(rewards)
    assert adv.abs().max() > 0.1, "Advantages too small — scaling missing"


@pytest.mark.parametrize("G", [4, 16, 64])
def test_gos_mean_centered(G):
    rewards = torch.rand(4, 8, G)
    adv = continuous_maxrl_advantages(rewards)
    mean_per_instance = adv.mean(dim=-1)
    assert torch.allclose(mean_per_instance, torch.zeros_like(mean_per_instance), atol=1e-5)


@pytest.mark.parametrize("G", [4, 16, 64])
def test_gos_vectorized_matches_loop(G):
    """Batched vectorized must match per-instance loop."""
    from experiment5_ordinal.advantages import continuous_maxrl_advantages as loop_gos
    B, H, W = 2, 4, 5
    rewards = torch.rand(B, H, W, G)
    vec_adv = continuous_maxrl_advantages(rewards)

    loop_adv = torch.zeros_like(rewards)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                loop_adv[b, h, w] = loop_gos(rewards[b, h, w])
    assert torch.allclose(vec_adv, loop_adv, atol=1e-5), f"Max diff: {(vec_adv - loop_adv).abs().max()}"


@pytest.mark.parametrize("G", [4, 16, 64])
def test_binary_exact_match(G):
    rewards = torch.rand(3, G)
    rewards[0, 0] = 1.0
    rewards[1, 0] = 1.0
    rewards[1, 1] = 1.0
    rewards[2, :] = 0.5  # no successes

    adv = binary_maxrl_advantages(rewards)
    # Instance 0: 1 success -> success gets (G-1)/1, failures get -1
    assert abs(adv[0, 0].item() - (G - 1)) < 1e-5
    # Instance 2: no success -> all zeros
    assert torch.allclose(adv[2], torch.zeros(G), atol=1e-6)


@pytest.mark.parametrize("G", [4, 16, 64])
def test_binary_all_zero_when_no_success(G):
    rewards = torch.rand(4, G) * 0.9  # < 1.0 everywhere
    adv = binary_maxrl_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv))


@pytest.mark.parametrize("G", [4, 16, 64])
def test_grpo_zscore_vectorized(G):
    if G < 4:
        pytest.skip()
    rewards = torch.randn(2, 3, G) * 2.0 + 5.0
    adv = grpo_advantages(rewards)
    mean = adv.mean(dim=-1)
    std = adv.std(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=0.05)


@pytest.mark.parametrize("G", [4, 16, 64])
def test_grpo_degenerate(G):
    rewards = torch.full((2, 3, G), 0.5)
    adv = grpo_advantages(rewards)
    assert torch.allclose(adv, torch.zeros_like(adv))


@pytest.mark.parametrize("G", [4, 16, 64])
def test_reinforce_vectorized(G):
    rewards = torch.rand(2, 3, G)
    adv = reinforce_advantages(rewards)
    mean_per_instance = adv.mean(dim=-1)
    assert torch.allclose(mean_per_instance, torch.zeros_like(mean_per_instance), atol=1e-5)


def test_dense_shape_preserved():
    """Verify advantages preserve (B, H, W, G) shape."""
    B, H, W, G = 2, 32, 32, 64
    rewards = torch.rand(B, H, W, G)
    for fn in [continuous_maxrl_advantages, binary_maxrl_advantages,
               grpo_advantages, reinforce_advantages]:
        adv = fn(rewards)
        assert adv.shape == rewards.shape


def test_single_element_returns_zeros():
    """G=1 should return zeros for all methods."""
    rewards = torch.tensor([0.5])
    for fn in [continuous_maxrl_advantages, continuous_maxrl_no_baseline,
               binary_maxrl_advantages, grpo_advantages, reinforce_advantages]:
        adv = fn(rewards)
        assert torch.allclose(adv, torch.zeros_like(adv))
