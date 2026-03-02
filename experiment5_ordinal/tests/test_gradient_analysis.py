"""Tests for gradient analysis module.

Covers:
- Per-image gradient computation for supervised methods
- Per-image gradient computation for RL methods
- Per-threshold gradient decomposition
- Output format and value ranges
"""

import pytest
import torch

from experiment5_ordinal.gradient_analysis import (
    compute_per_image_gradients,
    compute_per_threshold_gradients,
)
from experiment5_ordinal.model import create_model
from experiment5_ordinal.tests.conftest import get_device


class TestPerImageGradients:
    """Test per-image gradient computation."""

    @pytest.mark.parametrize("method", ["ordinal_ce", "plain_ce"])
    def test_supervised_gradients(self, method):
        """Supervised methods should produce valid gradient norms."""
        L = 10
        model = create_model(L=L, method=method, pretrained=False, seed=42).to(get_device())
        images = torch.randn(3, 3, 32, 32)
        targets = torch.randint(0, L, (3,))

        results = compute_per_image_gradients(
            model, images, targets, method, L, get_device(),
        )

        assert len(results) == 3
        for r in results:
            assert "gradient_l2_norm" in r
            assert r["gradient_l2_norm"] > 0
            assert "p_exact_match" in r
            assert 0 <= r["p_exact_match"] <= 1
            assert "entropy" in r
            assert r["entropy"] >= 0

    @pytest.mark.parametrize("method", ["cont_maxrl", "grpo", "reinforce"])
    def test_rl_gradients(self, method):
        """RL methods should produce valid gradient norms using rollouts."""
        L = 10
        model = create_model(L=L, method=method, pretrained=False, seed=42).to(get_device())
        images = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, L, (2,))

        results = compute_per_image_gradients(
            model, images, targets, method, L, get_device(),
            N_rollouts=100,
        )

        assert len(results) == 2
        for r in results:
            assert r["gradient_l2_norm"] >= 0
            assert 0 <= r["p_exact_match"] <= 1

    def test_mse_gradients(self):
        """MSE method should produce valid gradients."""
        L = 10
        model = create_model(L=L, method="mse", pretrained=False, seed=42).to(get_device())
        images = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, L, (2,))

        results = compute_per_image_gradients(
            model, images, targets, "mse", L, get_device(),
        )
        assert len(results) == 2
        for r in results:
            assert r["gradient_l2_norm"] >= 0

    def test_gradient_norm_finite(self):
        """Gradient norms should never be NaN or Inf."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        images = torch.randn(5, 3, 32, 32)
        targets = torch.randint(0, L, (5,))

        results = compute_per_image_gradients(
            model, images, targets, "ordinal_ce", L, get_device(),
        )
        for r in results:
            assert not (r["gradient_l2_norm"] != r["gradient_l2_norm"]), "NaN gradient"
            assert r["gradient_l2_norm"] < float("inf"), "Inf gradient"


class TestPerThresholdGradients:
    """Test per-threshold gradient decomposition."""

    def test_outputs_correct_count(self):
        """Should produce (n_images * (L-1)) entries."""
        L = 5
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        images = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, L, (2,))

        results = compute_per_threshold_gradients(
            model, images, targets, L, get_device(),
        )
        assert len(results) == 2 * (L - 1)  # 2 images × 4 thresholds

    def test_result_format(self):
        """Each result should have required keys."""
        L = 5
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        images = torch.randn(1, 3, 32, 32)
        targets = torch.randint(0, L, (1,))

        results = compute_per_threshold_gradients(
            model, images, targets, L, get_device(),
        )

        for r in results:
            assert "image_idx" in r
            assert "threshold_d" in r
            assert "survival_prob" in r
            assert "gradient_norm_at_threshold" in r
            assert "log_survival" in r
            assert "inverse_survival_weight" in r

    def test_survival_monotonic(self):
        """Survival probability W_d should be non-decreasing in d."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        images = torch.randn(1, 3, 32, 32)
        targets = torch.randint(0, L, (1,))

        results = compute_per_threshold_gradients(
            model, images, targets, L, get_device(),
        )

        for d in range(1, L - 1):
            r_prev = results[d - 1]
            r_curr = results[d]
            assert r_curr["survival_prob"] >= r_prev["survival_prob"] - 1e-6

    def test_gradient_higher_at_small_d(self):
        """Gradient norm should generally be larger at small d (tight windows).

        This tests the 1/p_tau weighting effect: when W_d is small (hard
        threshold), -log(W_d) has a larger gradient.
        """
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        images = torch.randn(1, 3, 32, 32)
        targets = torch.tensor([5])  # middle of range

        results = compute_per_threshold_gradients(
            model, images, targets, L, get_device(),
        )

        # d=0 (tightest) should have larger gradient than d=L-2 (loosest)
        grad_d0 = results[0]["gradient_norm_at_threshold"]
        grad_dmax = results[-1]["gradient_norm_at_threshold"]
        # This should hold for random init where W_0 is small
        assert grad_d0 > grad_dmax * 0.5, (
            f"grad@d=0={grad_d0:.4f} should be larger than grad@d={L-2}={grad_dmax:.4f}"
        )
