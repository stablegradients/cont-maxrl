"""Tests for training loops.

Covers:
- Supervised training reduces loss
- RL training produces valid loss and advantages
- Gradient norm computation
- Step logging format
- Method-specific properties (RL stats)
- Training loop doesn't crash on edge cases
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiment5_ordinal.model import create_model
from experiment5_ordinal.train import (
    _compute_grad_norm,
    train_rl_epoch,
    train_supervised_epoch,
)
from experiment5_ordinal.tests.conftest import get_device


def make_dummy_loader(B=16, L=10, n_batches=3):
    """Create a dummy DataLoader that mimics UTKFace format."""
    images = torch.randn(B * n_batches, 3, 32, 32)  # small images for speed
    targets = torch.randint(0, L, (B * n_batches,))
    ages = targets.clone()  # raw ages = targets for L=100-like

    class DummyDataset:
        def __init__(self, imgs, tgts, ags):
            self.imgs = imgs
            self.tgts = tgts
            self.ags = ags

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            return self.imgs[idx], self.tgts[idx], self.ags[idx]

    ds = DummyDataset(images, targets, ages)
    return DataLoader(ds, batch_size=B, shuffle=False, drop_last=True)


def make_small_model(L=10, method="ordinal_ce"):
    """Create a small ResNet-18 model (no pretrained for speed)."""
    return create_model(L=L, method=method, pretrained=False, seed=42).to(get_device())


class TestGradNorm:
    """Test gradient norm computation."""

    def test_grad_norm_positive(self):
        model = make_small_model()
        x = torch.randn(2, 3, 32, 32, device=get_device())
        loss = model(x).sum()
        loss.backward()
        norm = _compute_grad_norm(model)
        assert norm > 0

    def test_grad_norm_zero_before_backward(self):
        model = make_small_model()
        model.zero_grad()
        norm = _compute_grad_norm(model)
        assert norm == 0.0


class TestSupervisedTraining:
    """Test supervised training loop."""

    @pytest.mark.parametrize("method", ["ordinal_ce", "plain_ce"])
    def test_supervised_runs(self, method):
        """Supervised training should complete without error."""
        L = 10
        model = make_small_model(L=L, method=method)
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        total_steps = len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        result = train_supervised_epoch(
            model, loader, optimizer, scheduler, method, L,
            device=get_device(), epoch=1,
        )

        assert "epoch_loss" in result
        assert "step_logs" in result
        assert result["epoch_loss"] > 0
        assert len(result["step_logs"]) == len(loader)

    def test_mse_training(self):
        """MSE training should work with 1-output model."""
        L = 10
        model = make_small_model(L=L, method="mse")
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_supervised_epoch(
            model, loader, optimizer, scheduler, "mse", L,
            device=get_device(), epoch=1,
        )
        assert result["epoch_loss"] > 0

    def test_loss_decreases(self):
        """Loss should generally decrease over multiple epochs."""
        L = 10
        model = make_small_model(L=L, method="ordinal_ce")
        loader = make_dummy_loader(B=8, L=L, n_batches=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        total_steps = len(loader) * 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        losses = []
        for epoch in range(5):
            result = train_supervised_epoch(
                model, loader, optimizer, scheduler, "ordinal_ce", L,
                device=get_device(), epoch=epoch + 1,
            )
            losses.append(result["epoch_loss"])

        # Loss should decrease over 5 epochs (not necessarily monotonically)
        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_step_logs_format(self):
        """Step logs should have required keys."""
        L = 10
        model = make_small_model(L=L, method="ordinal_ce")
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_supervised_epoch(
            model, loader, optimizer, scheduler, "ordinal_ce", L,
            device=get_device(), epoch=1,
        )

        for log in result["step_logs"]:
            assert "train/loss" in log
            assert "train/lr" in log
            assert "train/grad_norm_global" in log


class TestRLTraining:
    """Test RL training loop."""

    @pytest.mark.parametrize("method", ["cont_maxrl", "binary_maxrl", "grpo", "reinforce"])
    def test_rl_runs(self, method):
        """RL training should complete without error."""
        L = 10
        N = 8
        model = make_small_model(L=L, method=method)
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_rl_epoch(
            model, loader, optimizer, scheduler, method, N, L,
            device=get_device(), epoch=1,
        )

        assert "epoch_loss" in result
        assert "rl_stats" in result
        assert "step_logs" in result

    def test_rl_stats_format(self):
        """RL stats should have expected keys."""
        L = 10
        model = make_small_model(L=L)
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_rl_epoch(
            model, loader, optimizer, scheduler, "cont_maxrl", 16, L,
            device=get_device(), epoch=1,
        )

        rl = result["rl_stats"]
        expected_keys = [
            "train_rl/advantage_mean",
            "train_rl/advantage_std",
            "train_rl/advantage_max",
            "train_rl/advantage_min",
            "train_rl/reward_mean",
            "train_rl/reward_std",
        ]
        for key in expected_keys:
            assert key in rl, f"Missing RL stat: {key}"

    def test_advantage_mean_near_zero(self):
        """Advantage mean should be near zero (mean-centering)."""
        L = 10
        model = make_small_model(L=L)
        loader = make_dummy_loader(B=8, L=L, n_batches=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_rl_epoch(
            model, loader, optimizer, scheduler, "cont_maxrl", 32, L,
            device=get_device(), epoch=1,
        )
        assert abs(result["rl_stats"]["train_rl/advantage_mean"]) < 0.01

    @pytest.mark.parametrize("N", [4, 16, 64])
    def test_different_N_values(self, N):
        """Training should work with different N values."""
        L = 10
        model = make_small_model(L=L)
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_rl_epoch(
            model, loader, optimizer, scheduler, "cont_maxrl", N, L,
            device=get_device(), epoch=1,
        )
        assert result["epoch_loss"] is not None

    def test_maxrl_specific_stats(self):
        """MaxRL methods should log weight_on_top and effective_sample_size."""
        L = 10
        model = make_small_model(L=L)
        loader = make_dummy_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader)
        )

        result = train_rl_epoch(
            model, loader, optimizer, scheduler, "cont_maxrl", 16, L,
            device=get_device(), epoch=1,
        )

        rl = result["rl_stats"]
        assert "train_rl/weight_on_top_sample" in rl
        assert "train_rl/effective_sample_size" in rl
        assert "train_rl/frac_unique_rewards" in rl
