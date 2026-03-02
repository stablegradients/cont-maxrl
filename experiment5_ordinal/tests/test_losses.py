"""Tests for ordinal CE loss and other loss functions.

Covers:
- L=2 binary recovery (ordinal CE == plain CE)
- Perfect model gives zero loss
- Uniform logits match analytical formula
- Gradient correctness via finite differences
- Loss monotonicity (better predictions → lower loss)
- Numerical stability (extreme logits, near-zero probabilities)
- Batch consistency (same result regardless of batching)
- W_d computation correctness
- MSE loss basic properties
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiment5_ordinal.losses import (
    compute_window_probs,
    mse_loss,
    ordinal_ce_loss,
)


class TestOrdinalCEBinaryRecovery:
    """Sanity check: at L=2, ordinal CE == plain CE."""

    def test_l2_matches_cross_entropy(self):
        """Ordinal CE with L=2 should exactly equal F.cross_entropy."""
        torch.manual_seed(42)
        for _ in range(10):
            B = torch.randint(1, 32, (1,)).item()
            logits = torch.randn(B, 2)
            targets = torch.randint(0, 2, (B,))
            oce = ordinal_ce_loss(logits, targets, L=2)
            pce = F.cross_entropy(logits, targets)
            assert torch.allclose(oce, pce, atol=1e-5), (
                f"L=2 mismatch: ordinal_ce={oce.item():.8f}, "
                f"plain_ce={pce.item():.8f}"
            )

    def test_l2_gradient_matches(self):
        """Gradients should also match at L=2."""
        torch.manual_seed(42)
        logits1 = torch.randn(8, 2, requires_grad=True)
        logits2 = logits1.detach().clone().requires_grad_(True)
        targets = torch.randint(0, 2, (8,))

        ordinal_ce_loss(logits1, targets, L=2).backward()
        F.cross_entropy(logits2, targets).backward()

        assert torch.allclose(logits1.grad, logits2.grad, atol=1e-5)


class TestOrdinalCEPerfectModel:
    """Perfect prediction should give zero loss."""

    @pytest.mark.parametrize("L", [2, 5, 10, 50, 100])
    def test_perfect_prediction_zero_loss(self, L):
        """When all probability is on the correct bin, loss = 0."""
        B = 4
        targets = torch.randint(0, L, (B,))
        logits = torch.full((B, L), -100.0)
        for i in range(B):
            logits[i, targets[i]] = 100.0
        loss = ordinal_ce_loss(logits, targets, L)
        assert loss.item() < 1e-6, f"Perfect model loss should be ~0, got {loss.item()}"

    def test_near_perfect_low_loss(self):
        """When most probability is on correct bin, loss is small."""
        L = 100
        B = 4
        targets = torch.tensor([25, 50, 75, 10])
        logits = torch.zeros(B, L)
        for i in range(B):
            logits[i, targets[i]] = 10.0  # dominant but not extreme
        loss = ordinal_ce_loss(logits, targets, L)
        assert loss.item() < 0.5, f"Near-perfect loss should be small, got {loss.item()}"


class TestOrdinalCEUniformLogits:
    """Uniform logits should match analytical formula."""

    @pytest.mark.parametrize("L", [5, 10, 50, 100])
    def test_uniform_matches_analytical(self, L):
        """Uniform π(b|x) = 1/L → W_d = min(2d+1, L)/L."""
        logits = torch.zeros(1, L)
        # Target in the middle for maximum symmetry
        target = torch.tensor([L // 2])

        loss = ordinal_ce_loss(logits, target, L)

        # Analytical: sum_{d=0}^{L-2} -log(W_d) / (L-1)
        # W_d = min(2d+1, L) / L for target at L//2
        t = target.item()
        analytical = 0.0
        for d in range(L - 1):
            low = max(0, t - d)
            high = min(L - 1, t + d)
            W_d = (high - low + 1) / L
            analytical += -math.log(max(W_d, 1e-12))
        analytical /= (L - 1)

        assert abs(loss.item() - analytical) < 0.01, (
            f"L={L}: loss={loss.item():.6f}, analytical={analytical:.6f}"
        )

    def test_uniform_different_targets(self):
        """Loss should vary by target position (edge vs center) with uniform logits."""
        L = 100
        logits = torch.zeros(1, L)

        loss_center = ordinal_ce_loss(logits, torch.tensor([50]), L)
        loss_edge = ordinal_ce_loss(logits, torch.tensor([0]), L)

        # Edge targets have asymmetric windows → different loss
        assert loss_center.item() != loss_edge.item()


class TestOrdinalCEGradients:
    """Gradient correctness via finite differences."""

    @pytest.mark.parametrize("L", [5, 10, 20])
    def test_finite_difference_gradient(self, L):
        """Autograd vs central finite differences."""
        B = 2
        torch.manual_seed(42)
        logits = torch.randn(B, L, requires_grad=True, dtype=torch.float64)
        targets = torch.randint(0, L, (B,))

        loss = ordinal_ce_loss(logits, targets, L)
        loss.backward()
        autograd = logits.grad.clone()

        eps = 1e-5
        fd = torch.zeros_like(logits)
        for i in range(B):
            for j in range(L):
                logits_p = logits.detach().clone()
                logits_p[i, j] += eps
                logits_m = logits.detach().clone()
                logits_m[i, j] -= eps
                fd[i, j] = (
                    ordinal_ce_loss(logits_p, targets, L).item()
                    - ordinal_ce_loss(logits_m, targets, L).item()
                ) / (2 * eps)

        rel_error = (autograd - fd).abs() / (fd.abs() + 1e-12)
        max_rel = rel_error.max().item()
        assert max_rel < 1e-4, f"L={L}: gradient rel error {max_rel:.2e}"

    def test_gradient_nonzero(self):
        """Gradient should be nonzero for non-perfect predictions."""
        L = 100
        torch.manual_seed(42)
        logits = torch.randn(4, L, requires_grad=True)
        targets = torch.randint(0, L, (4,))
        ordinal_ce_loss(logits, targets, L).backward()
        assert logits.grad.abs().max().item() > 1e-6


class TestOrdinalCEMonotonicity:
    """Better predictions should yield lower loss."""

    def test_more_mass_on_correct_lowers_loss(self):
        """Increasing probability on correct bin should decrease loss."""
        L = 100
        target = torch.tensor([42])

        logits_bad = torch.zeros(1, L)
        logits_good = torch.zeros(1, L)
        logits_good[0, 42] = 5.0  # more mass on correct bin

        loss_bad = ordinal_ce_loss(logits_bad, target, L)
        loss_good = ordinal_ce_loss(logits_good, target, L)

        assert loss_good < loss_bad

    def test_concentrated_near_target_better(self):
        """Mass concentrated near (but not at) target is better than far."""
        L = 100
        target = torch.tensor([50])

        # Mass near target
        logits_near = torch.full((1, L), -10.0)
        for b in range(45, 56):
            logits_near[0, b] = 0.0

        # Mass far from target
        logits_far = torch.full((1, L), -10.0)
        for b in range(0, 11):
            logits_far[0, b] = 0.0

        loss_near = ordinal_ce_loss(logits_near, target, L)
        loss_far = ordinal_ce_loss(logits_far, target, L)

        assert loss_near < loss_far


class TestOrdinalCENumericalStability:
    """Numerical edge cases."""

    def test_very_large_logits(self):
        """Should not produce NaN/Inf with extreme logits."""
        L = 100
        logits = torch.zeros(2, L)
        logits[0, 25] = 100.0
        logits[1, 75] = -100.0
        targets = torch.tensor([25, 75])
        loss = ordinal_ce_loss(logits, targets, L)
        assert torch.isfinite(loss)

    def test_all_same_logits(self):
        """Uniform distribution should not produce NaN."""
        L = 100
        logits = torch.ones(4, L) * 5.0  # uniform
        targets = torch.tensor([0, 49, 50, 99])
        loss = ordinal_ce_loss(logits, targets, L)
        assert torch.isfinite(loss)

    def test_requires_grad_works(self):
        """Should work with gradient tracking."""
        L = 100
        logits = torch.randn(4, L, requires_grad=True)
        targets = torch.randint(0, L, (4,))
        loss = ordinal_ce_loss(logits, targets, L)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestOrdinalCEBatchConsistency:
    """Same images should give same loss regardless of batching."""

    def test_batch_vs_individual(self):
        """Batch average should equal average of individual losses."""
        L = 50
        torch.manual_seed(42)
        B = 8
        logits = torch.randn(B, L)
        targets = torch.randint(0, L, (B,))

        # Batch loss
        batch_loss = ordinal_ce_loss(logits, targets, L)

        # Individual losses
        individual_losses = []
        for i in range(B):
            l = ordinal_ce_loss(logits[i:i + 1], targets[i:i + 1], L)
            individual_losses.append(l.item())
        avg_individual = sum(individual_losses) / B

        assert abs(batch_loss.item() - avg_individual) < 1e-5


class TestWindowProbs:
    """Tests for compute_window_probs."""

    def test_w0_equals_exact_prob(self):
        """W_0(x) should equal π(y*|x)."""
        L = 10
        torch.manual_seed(42)
        logits = torch.randn(4, L)
        probs = F.softmax(logits, dim=-1)
        targets = torch.randint(0, L, (4,))

        w_probs = compute_window_probs(probs, targets, L, [0])
        for i in range(4):
            expected = probs[i, targets[i]].item()
            actual = w_probs[0][i].item()
            assert abs(expected - actual) < 1e-6, (
                f"W_0 mismatch: expected={expected}, got={actual}"
            )

    def test_wL1_equals_one(self):
        """W_{L-1}(x) should be 1 (all bins within max distance)."""
        L = 10
        torch.manual_seed(42)
        logits = torch.randn(4, L)
        probs = F.softmax(logits, dim=-1)
        targets = torch.randint(0, L, (4,))

        w_probs = compute_window_probs(probs, targets, L, [L - 1])
        for val in w_probs[L - 1]:
            assert abs(val.item() - 1.0) < 1e-5

    def test_monotonicity(self):
        """W_d should be non-decreasing in d."""
        L = 20
        torch.manual_seed(42)
        logits = torch.randn(8, L)
        probs = F.softmax(logits, dim=-1)
        targets = torch.randint(0, L, (8,))

        d_values = list(range(L))
        w_probs = compute_window_probs(probs, targets, L, d_values)

        for i in range(8):
            for d in range(1, L):
                assert w_probs[d][i].item() >= w_probs[d - 1][i].item() - 1e-6

    def test_w_d_symmetry(self):
        """For uniform probs, W_d should depend only on d and L, not target."""
        L = 10
        probs = torch.ones(2, L) / L
        targets = torch.tensor([3, 7])

        w_probs = compute_window_probs(probs, targets, L, [0, 1, 2])
        # W_d = (2d+1)/L for interior targets with L=10
        # target=3, d=2: window [1,5] = 5 bins → 0.5
        # target=7, d=2: window [5,9] = 5 bins → 0.5
        assert abs(w_probs[2][0].item() - w_probs[2][1].item()) < 1e-6


class TestMSELoss:
    """Tests for MSE regression loss."""

    def test_perfect_prediction_zero(self):
        """Perfect output should give zero loss."""
        L = 100
        targets = torch.tensor([50])
        # sigmoid(output) = 50/99 → output = logit(50/99)
        expected = 50.0 / 99.0
        output = torch.tensor([[math.log(expected / (1 - expected))]])
        loss = mse_loss(output, targets, L)
        assert loss.item() < 1e-6

    def test_positive_loss(self):
        """Non-perfect prediction gives positive loss."""
        L = 100
        output = torch.tensor([[0.0]])  # sigmoid(0) = 0.5
        targets = torch.tensor([0])  # normalized = 0
        loss = mse_loss(output, targets, L)
        assert loss.item() > 0

    def test_gradient_flows(self):
        """MSE loss should produce valid gradients."""
        L = 100
        output = torch.randn(4, 1, requires_grad=True)
        targets = torch.randint(0, L, (4,))
        loss = mse_loss(output, targets, L)
        loss.backward()
        assert output.grad is not None
        assert torch.isfinite(output.grad).all()


class TestOrdinalCEvsPlainCE:
    """Compare ordinal CE and plain CE behavior."""

    def test_ordinal_ce_lower_than_plain_ce_on_near_misses(self):
        """Ordinal CE should reward near-miss predictions more than plain CE.

        If mass is on bins adjacent to the correct one, ordinal CE should be
        lower (better) than if the same mass were on distant bins, even though
        plain CE treats both equally (both are wrong).
        """
        L = 100
        target = torch.tensor([50])

        # Prediction concentrated at bin 51 (near miss)
        logits_near = torch.full((1, L), -100.0)
        logits_near[0, 51] = 100.0

        # Prediction concentrated at bin 0 (far miss)
        logits_far = torch.full((1, L), -100.0)
        logits_far[0, 0] = 100.0

        oce_near = ordinal_ce_loss(logits_near, target, L)
        oce_far = ordinal_ce_loss(logits_far, target, L)

        # Plain CE is the same for both (both get the wrong bin)
        pce_near = F.cross_entropy(logits_near, target)
        pce_far = F.cross_entropy(logits_far, target)
        assert abs(pce_near.item() - pce_far.item()) < 0.1

        # But ordinal CE is lower for the near miss
        assert oce_near.item() < oce_far.item()

    def test_at_d0_plain_ce_dominates(self):
        """The d=0 term of ordinal CE is exactly plain CE (up to scaling)."""
        L = 10
        torch.manual_seed(42)
        logits = torch.randn(4, L)
        targets = torch.randint(0, L, (4,))

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # d=0 contribution to ordinal CE = -log(W_0) / (L-1) = -log(pi(y*|x)) / (L-1)
        # Plain CE = -log(pi(y*|x))
        # So d=0 term = plain_CE / (L-1)
        plain_ce = F.cross_entropy(logits, targets, reduction="none")
        w0 = compute_window_probs(probs, targets, L, [0])[0]
        d0_term = -torch.log(w0) / (L - 1)  # per-sample contribution from d=0

        assert torch.allclose(d0_term, plain_ce / (L - 1), atol=1e-5)
