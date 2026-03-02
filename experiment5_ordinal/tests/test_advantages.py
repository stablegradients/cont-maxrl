"""Tests for advantage estimators.

Covers:
- Mean-centering property (sum ≈ 0) for all methods
- Edge cases (G=1, identical rewards, all-zero, all-one)
- Binary reward consistency (cont_maxrl == binary_maxrl on {0,1} rewards)
- GRPO z-score normalization correctness
- REINFORCE mean-centering correctness
- Monotonicity (higher reward → higher advantage)
- Worked examples with exact expected values
- Numerical stability
"""

import pytest
import torch

from experiment5_ordinal.advantages import (
    ADVANTAGE_FNS,
    binary_maxrl_advantages,
    continuous_maxrl_advantages,
    grpo_advantages,
    reinforce_advantages,
)


class TestMeanCentering:
    """All advantage methods should produce mean-centered advantages."""

    @pytest.mark.parametrize("method", ADVANTAGE_FNS.keys())
    def test_mean_centered(self, method):
        """Advantages should sum to approximately zero."""
        fn = ADVANTAGE_FNS[method]
        torch.manual_seed(42)
        for N in [4, 16, 64, 256]:
            rewards = torch.rand(N)
            adv = fn(rewards)
            assert abs(adv.sum().item()) < 1e-4, (
                f"{method} N={N}: sum={adv.sum().item()}"
            )

    @pytest.mark.parametrize("method", ADVANTAGE_FNS.keys())
    def test_output_shape(self, method):
        """Output shape should match input shape."""
        fn = ADVANTAGE_FNS[method]
        for N in [1, 4, 16, 64]:
            rewards = torch.rand(N)
            adv = fn(rewards)
            assert adv.shape == rewards.shape


class TestEdgeCases:
    """Edge cases all methods should handle."""

    @pytest.mark.parametrize("method", ADVANTAGE_FNS.keys())
    def test_single_sample(self, method):
        """G=1 should return zero advantage."""
        fn = ADVANTAGE_FNS[method]
        adv = fn(torch.tensor([0.5]))
        assert adv.item() == 0.0

    @pytest.mark.parametrize("method", ADVANTAGE_FNS.keys())
    def test_all_identical_rewards(self, method):
        """If all rewards are the same, all advantages should be zero."""
        fn = ADVANTAGE_FNS[method]
        rewards = torch.full((10,), 0.7)  # use full() to ensure truly identical values
        adv = fn(rewards)
        assert torch.allclose(adv, torch.zeros(10), atol=1e-4), (
            f"{method}: identical rewards should give zero advantages, got {adv}"
        )

    @pytest.mark.parametrize("method", ADVANTAGE_FNS.keys())
    def test_all_zero_rewards(self, method):
        """All-zero rewards should give zero advantages."""
        fn = ADVANTAGE_FNS[method]
        adv = fn(torch.zeros(8))
        assert torch.allclose(adv, torch.zeros(8), atol=1e-6)

    @pytest.mark.parametrize("method", ADVANTAGE_FNS.keys())
    def test_no_nan_inf(self, method):
        """Output should never contain NaN or Inf."""
        fn = ADVANTAGE_FNS[method]
        for rewards in [
            torch.rand(10),
            torch.zeros(10),
            torch.ones(10),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        ]:
            adv = fn(rewards)
            assert torch.isfinite(adv).all(), f"{method}: non-finite with {rewards}"


class TestContinuousMaxRL:
    """Tests specific to continuous MaxRL (gap-over-survivors)."""

    def test_worked_example_4_samples(self):
        """Verify exact values on a worked example.

        Rewards: [0.0, 0.3, 0.7, 1.0] (sorted), N=4
        Gaps: [0.0, 0.3, 0.4, 0.3]
        Survivors: [4, 3, 2, 1]
        Increments: [0.0, 0.1, 0.2, 0.3]
        Cumulative: [0.0, 0.1, 0.3, 0.6]
        Scaled by N=4: [0.0, 0.4, 1.2, 2.4]
        Mean = 1.0
        Centered: [-1.0, -0.6, 0.2, 1.4]
        """
        rewards = torch.tensor([0.0, 0.3, 0.7, 1.0])
        adv = continuous_maxrl_advantages(rewards)
        expected = torch.tensor([-1.0, -0.6, 0.2, 1.4])
        assert torch.allclose(adv, expected, atol=1e-5), (
            f"Got {adv}, expected {expected}"
        )

    def test_monotonicity(self):
        """Higher reward should get higher (or equal) advantage."""
        rewards = torch.tensor([0.1, 0.3, 0.5, 0.8, 1.0])
        adv = continuous_maxrl_advantages(rewards)
        for i in range(len(rewards) - 1):
            assert adv[i] <= adv[i + 1] + 1e-6, (
                f"adv[{i}]={adv[i]:.4f} > adv[{i+1}]={adv[i+1]:.4f}"
            )

    def test_ordering_preserved_unsorted_input(self):
        """Order should be preserved when input is not sorted."""
        rewards = torch.tensor([0.5, 0.1, 0.9, 0.3])
        adv = continuous_maxrl_advantages(rewards)
        # Reward 0.9 (index 2) should have highest advantage
        assert adv[2] > adv[0]
        assert adv[2] > adv[1]
        assert adv[2] > adv[3]
        # Reward 0.1 (index 1) should have lowest advantage
        assert adv[1] < adv[0]
        assert adv[1] < adv[3]

    def test_top_sample_gets_most_weight(self):
        """The sample with highest reward should get the largest advantage."""
        torch.manual_seed(42)
        rewards = torch.rand(64)
        adv = continuous_maxrl_advantages(rewards)
        max_reward_idx = rewards.argmax().item()
        assert adv[max_reward_idx] == adv.max()

    def test_duplicate_rewards(self):
        """Should handle duplicate reward values."""
        rewards = torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0])
        adv = continuous_maxrl_advantages(rewards)
        assert torch.isfinite(adv).all()
        assert abs(adv.sum().item()) < 1e-6
        # All 0.5 rewards should get the same advantage
        assert abs(adv[0].item() - adv[1].item()) < 1e-6
        assert abs(adv[0].item() - adv[2].item()) < 1e-6


class TestBinaryMaxRL:
    """Tests specific to binary MaxRL (Tajwar et al.)."""

    def test_all_failures(self):
        """K=0: all advantages should be zero."""
        rewards = torch.tensor([0.0, 0.3, 0.5, 0.7])  # none are 1.0
        adv = binary_maxrl_advantages(rewards)
        assert torch.allclose(adv, torch.zeros(4), atol=1e-6)

    def test_all_successes(self):
        """K=N: success adv = 0 (no contrast with failures)."""
        rewards = torch.ones(5)
        adv = binary_maxrl_advantages(rewards)
        # Mean-centered: success = (N-K)/(K*N) = 0, failure doesn't exist
        assert torch.allclose(adv, torch.zeros(5), atol=1e-6)

    def test_one_success(self):
        """K=1: success gets +(N-1)/K = 4, failures get -1."""
        N = 5
        rewards = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        adv = binary_maxrl_advantages(rewards)

        expected_success = (N - 1) / 1  # = 4.0
        expected_failure = -1.0

        assert abs(adv[2].item() - expected_success) < 1e-6
        for i in [0, 1, 3, 4]:
            assert abs(adv[i].item() - expected_failure) < 1e-6

    def test_worked_example(self):
        """Verify formula: K=2, N=5."""
        rewards = torch.tensor([0.0, 1.0, 0.5, 1.0, 0.3])
        adv = binary_maxrl_advantages(rewards)

        K, N = 2, 5
        expected_success = (N - K) / K  # 3/2 = 1.5
        expected_failure = -1.0

        assert abs(adv[1].item() - expected_success) < 1e-6
        assert abs(adv[3].item() - expected_success) < 1e-6
        assert abs(adv[0].item() - expected_failure) < 1e-6
        assert abs(adv[2].item() - expected_failure) < 1e-6
        assert abs(adv[4].item() - expected_failure) < 1e-6


class TestBinaryRecovery:
    """Continuous MaxRL should recover binary MaxRL on binary {0,1} rewards."""

    @pytest.mark.parametrize("K", [0, 1, 2, 3, 5, 8, 10])
    def test_binary_rewards_same_advantages(self, K):
        """On {0,1} rewards, cont_maxrl should equal binary_maxrl."""
        N = 10
        if K > N:
            return

        rewards = torch.zeros(N)
        rewards[:K] = 1.0
        # Shuffle to test ordering
        perm = torch.randperm(N)
        rewards = rewards[perm]

        adv_cont = continuous_maxrl_advantages(rewards)
        adv_bin = binary_maxrl_advantages(rewards)

        assert torch.allclose(adv_cont, adv_bin, atol=1e-5), (
            f"K={K}: cont={adv_cont.tolist()}, bin={adv_bin.tolist()}"
        )

    def test_binary_recovery_large_N(self):
        """Test with larger N and various K values."""
        N = 100
        for K in [0, 1, 5, 25, 50, 99, 100]:
            rewards = torch.zeros(N)
            rewards[:K] = 1.0
            rewards = rewards[torch.randperm(N)]

            adv_cont = continuous_maxrl_advantages(rewards)
            adv_bin = binary_maxrl_advantages(rewards)

            assert torch.allclose(adv_cont, adv_bin, atol=1e-5), (
                f"K={K}, N={N}: max_diff={((adv_cont-adv_bin).abs().max())}"
            )


class TestGRPO:
    """Tests for GRPO (z-score normalization)."""

    def test_z_score_values(self):
        """Verify z-score normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = grpo_advantages(rewards)

        mean_r = rewards.mean()
        std_r = rewards.std()
        expected = (rewards - mean_r) / (std_r + 1e-8)

        assert torch.allclose(adv, expected, atol=1e-6)

    def test_unit_variance(self):
        """Output should have approximately unit variance."""
        torch.manual_seed(42)
        rewards = torch.randn(100).abs()
        adv = grpo_advantages(rewards)
        assert abs(adv.std().item() - 1.0) < 0.1

    def test_zero_std_returns_zeros(self):
        """When std < eps, should return zeros."""
        rewards = torch.full((10,), 3.14)
        adv = grpo_advantages(rewards)
        assert torch.allclose(adv, torch.zeros(10), atol=1e-6)


class TestREINFORCE:
    """Tests for REINFORCE with baseline."""

    def test_mean_centered(self):
        """Output should be exactly mean-centered."""
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
        adv = reinforce_advantages(rewards)
        expected = rewards - rewards.mean()
        assert torch.allclose(adv, expected, atol=1e-6)

    def test_preserves_scale(self):
        """Unlike GRPO, REINFORCE should not normalize by std."""
        rewards = torch.tensor([0.0, 10.0])
        adv = reinforce_advantages(rewards)
        # Mean = 5, so advantages should be [-5, 5]
        assert abs(adv[0].item() - (-5.0)) < 1e-6
        assert abs(adv[1].item() - 5.0) < 1e-6

    def test_vs_grpo_no_normalization(self):
        """REINFORCE and GRPO differ by std normalization."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        adv_reinforce = reinforce_advantages(rewards)
        adv_grpo = grpo_advantages(rewards)

        # GRPO should have unit variance, REINFORCE should not
        assert abs(adv_grpo.std().item() - 1.0) < 0.1
        # REINFORCE std should be the same as original reward std
        assert abs(adv_reinforce.std().item() - rewards.std().item()) < 0.01


class TestCrossMethodComparison:
    """Compare properties across methods."""

    def test_same_ordering(self):
        """All methods should assign highest advantage to highest reward."""
        rewards = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.7])
        for name, fn in ADVANTAGE_FNS.items():
            adv = fn(rewards)
            # Handle binary_maxrl (which ignores non-binary rewards)
            if name == "binary_maxrl":
                continue
            max_idx = adv.argmax().item()
            assert max_idx == 1, f"{name}: max advantage at index {max_idx}, not 1"

    def test_sign_consistency(self):
        """Above-average rewards should get positive advantages (for non-binary methods)."""
        torch.manual_seed(42)
        rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        mean_r = rewards.mean().item()

        for name in ["cont_maxrl", "grpo", "reinforce"]:
            fn = ADVANTAGE_FNS[name]
            adv = fn(rewards)
            for i, r in enumerate(rewards):
                if r.item() > mean_r + 0.01:
                    assert adv[i].item() > 0, (
                        f"{name}: r={r.item():.2f} > mean but adv={adv[i].item():.4f}"
                    )
                elif r.item() < mean_r - 0.01:
                    assert adv[i].item() < 0, (
                        f"{name}: r={r.item():.2f} < mean but adv={adv[i].item():.4f}"
                    )

    def test_continuous_rewards_advantage_spread(self):
        """With continuous rewards, MaxRL should spread weight more towards top."""
        # 100 rewards uniformly in [0, 1]
        torch.manual_seed(42)
        rewards = torch.linspace(0, 1, 100)

        adv_maxrl = continuous_maxrl_advantages(rewards)
        adv_grpo = grpo_advantages(rewards)

        # MaxRL puts more relative weight on the top sample
        top10_maxrl = adv_maxrl[-10:].mean().item()
        top10_grpo = adv_grpo[-10:].mean().item()

        # Normalize by total positive advantage for fair comparison
        pos_maxrl = adv_maxrl[adv_maxrl > 0].sum().item()
        pos_grpo = adv_grpo[adv_grpo > 0].sum().item()

        frac_maxrl = top10_maxrl * 10 / pos_maxrl
        frac_grpo = top10_grpo * 10 / pos_grpo

        assert frac_maxrl > frac_grpo, (
            f"MaxRL should concentrate more on top: "
            f"maxrl_frac={frac_maxrl:.3f}, grpo_frac={frac_grpo:.3f}"
        )
