"""Tier 5: End-to-end smoke tests (advantage-level, no GPU required)."""
import torch
import pytest

from experiment4_llm.src.advantages import (
    maxrl_unnormalized_advantage,
    binary_maxrl_advantage,
)


class TestGradientMatchGOSvsBinMaxRL:
    def test_100_trials_binary(self):
        """GOS(binary) and binary_maxrl must produce identical advantages."""
        torch.manual_seed(42)
        for _ in range(100):
            G = 16
            rewards = (torch.rand(G) < 0.3).float()
            if rewards.sum() in (0, G):
                continue
            a = maxrl_unnormalized_advantage(rewards)
            b = binary_maxrl_advantage(rewards)
            assert torch.allclose(a, b, atol=1e-5)

    def test_continuous_rewards_differ(self):
        """On continuous rewards, GOS and binary_maxrl should differ."""
        torch.manual_seed(42)
        rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        gos = maxrl_unnormalized_advantage(rewards)
        bml = binary_maxrl_advantage(rewards)
        assert not torch.allclose(gos, bml, atol=1e-3)
