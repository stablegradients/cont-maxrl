"""Tests for evaluation metrics.

Covers:
- MAE correctness
- Expected reward analytical computation
- Ordinal CE objective matches loss
- Cumulative accuracy at thresholds
- Entropy computation
- Best-of-N sampling
- Per-difficulty tier metrics
- MSE method evaluation
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment5_ordinal.evaluate import evaluate_model
from experiment5_ordinal.model import create_model
from experiment5_ordinal.tests.conftest import get_device


def make_eval_loader(B=16, L=10, n_batches=2):
    """Create a dummy eval DataLoader."""
    images = torch.randn(B * n_batches, 3, 32, 32)
    targets = torch.randint(0, L, (B * n_batches,))
    ages = targets.clone()

    class DummyDS:
        def __init__(self, imgs, tgts, ags):
            self.data = list(zip(imgs, tgts, ags))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    return DataLoader(DummyDS(images, targets, ages), batch_size=B)


class TestEvaluateBasic:
    """Basic evaluation tests."""

    def test_returns_dict(self):
        """evaluate_model should return a dict with expected keys."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=4, L=L, n_batches=2)

        metrics = evaluate_model(
            model, loader, L, device=get_device(),
            best_of_n_trials=10,
            best_of_n_values=(1, 8),
        )

        assert isinstance(metrics, dict)
        assert "val/mae" in metrics
        assert "val/expected_reward" in metrics
        assert "val/ordinal_ce_objective" in metrics
        assert "val/plain_ce_objective" in metrics
        assert "val/acc_exact" in metrics
        assert "val/entropy_mean" in metrics

    def test_mae_range(self):
        """MAE should be non-negative and bounded by L-1."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=8, L=L, n_batches=3)

        metrics = evaluate_model(model, loader, L, get_device(),
                                 best_of_n_trials=5, best_of_n_values=(1,))
        assert 0 <= metrics["val/mae"] <= L - 1

    def test_expected_reward_range(self):
        """Expected reward should be in [0, 1]."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=8, L=L, n_batches=3)

        metrics = evaluate_model(model, loader, L, get_device(),
                                 best_of_n_trials=5, best_of_n_values=(1,))
        assert 0 <= metrics["val/expected_reward"] <= 1

    def test_entropy_non_negative(self):
        """Entropy should be non-negative."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=8, L=L, n_batches=2)

        metrics = evaluate_model(model, loader, L, get_device(),
                                 best_of_n_trials=5, best_of_n_values=(1,))
        assert metrics["val/entropy_mean"] >= 0

    def test_acc_monotonicity(self):
        """Acc@d should be non-decreasing in d."""
        L = 20
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=8, L=L, n_batches=3)

        metrics = evaluate_model(model, loader, L, get_device(),
                                 best_of_n_trials=5, best_of_n_values=(1,))
        accs = [
            metrics["val/acc_exact"],
            metrics["val/acc_within_1"],
            metrics["val/acc_within_2"],
            metrics["val/acc_within_5"],
            metrics["val/acc_within_10"],
            metrics["val/acc_within_20"],
        ]
        for i in range(len(accs) - 1):
            assert accs[i] <= accs[i + 1] + 1e-6, (
                f"Acc not monotonic: {accs}"
            )


class TestPerfectModel:
    """Test metrics with a model that always predicts correctly."""

    def test_perfect_model_metrics(self):
        """A perfect model should get MAE=0, reward=1, acc_exact=1."""
        L = 10

        # Create a model and manually set it to predict target
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())

        # Create loader with known targets
        B = 4
        images = torch.randn(B, 3, 32, 32)
        targets = torch.tensor([3, 3, 3, 3])
        ages = targets.clone()

        # Set model FC to strongly predict bin 3
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[3] = 100.0  # put all mass on bin 3

        class DS:
            def __init__(self):
                self.data = list(zip(images, targets, ages))
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        loader = DataLoader(DS(), batch_size=B)
        metrics = evaluate_model(
            model, loader, L, get_device(),
            best_of_n_trials=10, best_of_n_values=(1,),
        )

        assert metrics["val/mae"] < 0.1
        assert metrics["val/expected_reward"] > 0.99
        assert metrics["val/acc_exact"] > 0.99


class TestBestOfN:
    """Test best-of-N reward computation."""

    def test_best_of_1_approx_expected(self):
        """Best-of-1 should approximately equal expected reward."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=16, L=L, n_batches=5)

        metrics = evaluate_model(
            model, loader, L, get_device(),
            best_of_n_trials=200,
            best_of_n_values=(1, 8),
        )

        # Best-of-1 should be close to expected reward
        assert abs(metrics["val/best_of_1_reward"] - metrics["val/expected_reward"]) < 0.05

    def test_best_of_n_monotonic(self):
        """Best-of-N should increase with N."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=16, L=L, n_batches=3)

        metrics = evaluate_model(
            model, loader, L, get_device(),
            best_of_n_trials=50,
            best_of_n_values=(1, 8, 64),
        )

        assert metrics["val/best_of_1_reward"] <= metrics["val/best_of_8_reward"] + 0.01
        assert metrics["val/best_of_8_reward"] <= metrics["val/best_of_64_reward"] + 0.01


class TestMSEEvaluation:
    """Test evaluation for MSE regression method."""

    def test_mse_returns_basic_metrics(self):
        """MSE evaluation should return at least MAE and expected reward."""
        L = 10
        model = create_model(L=L, method="mse", pretrained=False, seed=42).to(get_device())
        loader = make_eval_loader(B=8, L=L, n_batches=2)

        metrics = evaluate_model(
            model, loader, L, get_device(),
            method="mse",
        )
        assert "val/mae" in metrics
        assert "val/expected_reward" in metrics


class TestPerDifficultyMetrics:
    """Test per-difficulty tier metrics."""

    def test_tier_metrics_present(self):
        """Should have metrics for each difficulty tier."""
        L = 100

        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())

        # Create data with known ages spanning all tiers
        B = 12
        images = torch.randn(B, 3, 32, 32)
        # easy: 20-39, medium: 0-4, 60-79, hard: 5-19, 80-99
        ages_list = [25, 30, 35, 1, 2, 65, 70, 10, 15, 85, 90, 95]
        targets = torch.tensor(ages_list)
        ages = targets.clone()

        class DS:
            def __init__(self):
                self.data = list(zip(images, targets, ages))
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        loader = DataLoader(DS(), batch_size=B)

        metrics = evaluate_model(
            model, loader, L, get_device(),
            best_of_n_trials=5, best_of_n_values=(1,),
        )

        # Check tier metrics exist
        for tier in ["easy", "medium", "hard"]:
            assert f"val_{tier}/mae" in metrics, f"Missing {tier}/mae"
            assert f"val_{tier}/expected_reward" in metrics
