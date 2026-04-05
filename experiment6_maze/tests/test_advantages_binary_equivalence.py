"""Tier 3: Advantage estimator equivalence tests (CRITICAL)."""
import torch
import pytest

from experiment4_llm.src.advantages import (
    grpo_advantage,
    maxrl_unnormalized_advantage,
    binary_maxrl_advantage,
)


class TestBinaryRecovery:
    def test_hand_verified_g5_c3(self):
        """G=5, C=3: success -> (5-3)/3 = 2/3, failure -> -1."""
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        gos = maxrl_unnormalized_advantage(rewards)
        bml = binary_maxrl_advantage(rewards)
        for i in range(5):
            if rewards[i] == 1.0:
                assert abs(gos[i].item() - 2.0/3.0) < 1e-5
                assert abs(bml[i].item() - 2.0/3.0) < 1e-5
            else:
                assert abs(gos[i].item() - (-1.0)) < 1e-5
                assert abs(bml[i].item() - (-1.0)) < 1e-5
        assert torch.allclose(gos, bml, atol=1e-5)

    def test_hand_verified_g4_c1(self):
        """G=4, C=1: success -> 3, failure -> -1."""
        rewards = torch.tensor([0.0, 0.0, 1.0, 0.0])
        gos = maxrl_unnormalized_advantage(rewards)
        bml = binary_maxrl_advantage(rewards)
        assert torch.allclose(gos, bml, atol=1e-5)
        assert abs(gos[2].item() - 3.0) < 1e-5
        assert abs(gos[0].item() - (-1.0)) < 1e-5

    def test_fuzz_10k(self):
        """10,000 random binary reward groups."""
        torch.manual_seed(0)
        failures = []
        for trial in range(10000):
            G = torch.randint(2, 64, (1,)).item()
            p = torch.rand(1).item() * 0.9 + 0.05
            rewards = (torch.rand(G) < p).float()
            if rewards.sum() == 0 or rewards.sum() == G:
                continue
            gos = maxrl_unnormalized_advantage(rewards)
            bml = binary_maxrl_advantage(rewards)
            if not torch.allclose(gos, bml, atol=1e-4):
                failures.append(f"Trial {trial}: G={G}, C={int(rewards.sum())}, max_diff={torch.max(torch.abs(gos - bml)).item()}")
        assert len(failures) == 0, f"{len(failures)} failures:\n" + "\n".join(failures[:5])

    def test_all_success_is_zero(self):
        rewards = torch.ones(16)
        gos = maxrl_unnormalized_advantage(rewards)
        bml = binary_maxrl_advantage(rewards)
        assert torch.allclose(gos, torch.zeros(16), atol=1e-6)
        assert torch.allclose(bml, torch.zeros(16), atol=1e-4)

    def test_all_failure_is_zero(self):
        rewards = torch.zeros(16)
        gos = maxrl_unnormalized_advantage(rewards)
        bml = binary_maxrl_advantage(rewards)
        assert torch.allclose(gos, torch.zeros(16), atol=1e-6)
        assert torch.allclose(bml, torch.zeros(16), atol=1e-6)


class TestGOSContinuousProperties:
    def test_mean_centered(self):
        torch.manual_seed(0)
        for _ in range(100):
            rewards = torch.rand(16)
            gos = maxrl_unnormalized_advantage(rewards)
            assert abs(gos.sum().item()) < 1e-4

    def test_monotone_in_reward(self):
        rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        gos = maxrl_unnormalized_advantage(rewards)
        sorted_idx = torch.argsort(rewards)
        sorted_adv = gos[sorted_idx]
        assert torch.all(sorted_adv[1:] >= sorted_adv[:-1] - 1e-6)


class TestGRPOProperties:
    def test_zero_mean_unit_std(self):
        torch.manual_seed(0)
        for _ in range(100):
            rewards = torch.rand(16) * 10.0
            grpo = grpo_advantage(rewards)
            if grpo.std(unbiased=False).item() > 1e-6:
                assert abs(grpo.mean().item()) < 1e-4
                assert abs(grpo.std(unbiased=False).item() - 1.0) < 1e-3
