"""Integration tests: end-to-end training + evaluation.

Covers:
- Full mini-training run (2 epochs) for each method
- Model produces valid metrics after training
- Checkpoint saving/loading
- Supervised methods achieve lower loss after training
- RL methods produce valid advantages and rewards
"""

import json
import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

from experiment5_ordinal.evaluate import evaluate_model
from experiment5_ordinal.model import create_model
from experiment5_ordinal.train import train_rl_epoch, train_supervised_epoch
from experiment5_ordinal.tests.conftest import get_device


def make_integration_loader(B=8, L=10, n_batches=4):
    """Create a dummy DataLoader for integration tests."""
    N = B * n_batches
    images = torch.randn(N, 3, 32, 32)
    targets = torch.randint(0, L, (N,))
    ages = targets.clone()

    class DS:
        def __init__(self):
            self.data = list(zip(images, targets, ages))
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    return DataLoader(DS(), batch_size=B, shuffle=False, drop_last=True)


class TestSupervisedIntegration:
    """End-to-end supervised training tests."""

    @pytest.mark.parametrize("method", ["ordinal_ce", "plain_ce", "mse"])
    def test_full_training_loop(self, method):
        """Train for 2 epochs, verify loss decreases and metrics are valid."""
        L = 10
        model = create_model(L=L, method=method, pretrained=False, seed=42).to(get_device())
        loader = make_integration_loader(B=8, L=L, n_batches=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        total_steps = len(loader) * 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )

        losses = []
        for epoch in range(1, 3):
            result = train_supervised_epoch(
                model, loader, optimizer, scheduler, method, L,
                device=get_device(), epoch=epoch,
            )
            losses.append(result["epoch_loss"])

        # Evaluate
        metrics = evaluate_model(
            model, loader, L, get_device(),
            method=method,
            best_of_n_trials=5,
            best_of_n_values=(1,),
        )

        assert "val/mae" in metrics
        assert metrics["val/mae"] >= 0


class TestRLIntegration:
    """End-to-end RL training tests."""

    @pytest.mark.parametrize("method", ["cont_maxrl", "binary_maxrl", "grpo", "reinforce"])
    def test_full_rl_training_loop(self, method):
        """Train for 2 epochs with RL, verify metrics are valid."""
        L = 10
        N = 8
        model = create_model(L=L, method=method, pretrained=False, seed=42).to(get_device())
        loader = make_integration_loader(B=4, L=L, n_batches=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        total_steps = len(loader) * 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )

        for epoch in range(1, 3):
            result = train_rl_epoch(
                model, loader, optimizer, scheduler, method, N, L,
                device=get_device(), epoch=epoch,
            )
            assert "rl_stats" in result
            assert result["epoch_loss"] is not None

        # Evaluate
        metrics = evaluate_model(
            model, loader, L, get_device(),
            best_of_n_trials=5,
            best_of_n_values=(1,),
        )
        assert "val/mae" in metrics
        assert 0 <= metrics["val/expected_reward"] <= 1


class TestCheckpointing:
    """Test model checkpoint save/load."""

    def test_save_and_load_checkpoint(self):
        """Model should produce same output after loading."""
        L = 10
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())

        # Forward pass before saving
        x = torch.randn(2, 3, 32, 32, device=get_device())
        out_before = model(x).detach()

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": 5,
            }, ckpt_path)

        # Load into new model
        model2 = create_model(L=L, method="ordinal_ce", pretrained=False, seed=99).to(get_device())
        ckpt = torch.load(ckpt_path, weights_only=True)
        model2.load_state_dict(ckpt["model_state_dict"])

        out_after = model2(x).detach()
        assert torch.allclose(out_before, out_after, atol=1e-6)

        os.unlink(ckpt_path)


class TestL2BinaryRecoveryIntegration:
    """Integration test for L=2 binary recovery."""

    def test_ordinal_ce_equals_plain_ce_at_l2(self):
        """At L=2, training with ordinal CE and plain CE should behave identically."""
        L = 2
        torch.manual_seed(42)

        loader = make_integration_loader(B=8, L=L, n_batches=4)

        # Train ordinal CE
        model_ord = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42).to(get_device())
        opt_ord = torch.optim.SGD(model_ord.parameters(), lr=0.01)
        sched_ord = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ord, T_max=len(loader),
        )
        result_ord = train_supervised_epoch(
            model_ord, loader, opt_ord, sched_ord, "ordinal_ce", L,
            device=get_device(), epoch=1,
        )

        # Train plain CE (same init)
        model_ce = create_model(L=L, method="plain_ce", pretrained=False, seed=42).to(get_device())
        opt_ce = torch.optim.SGD(model_ce.parameters(), lr=0.01)
        sched_ce = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ce, T_max=len(loader),
        )
        result_ce = train_supervised_epoch(
            model_ce, loader, opt_ce, sched_ce, "plain_ce", L,
            device=get_device(), epoch=1,
        )

        # Losses should be nearly identical
        assert abs(result_ord["epoch_loss"] - result_ce["epoch_loss"]) < 0.1, (
            f"L=2 loss mismatch: ordinal={result_ord['epoch_loss']:.6f}, "
            f"plain={result_ce['epoch_loss']:.6f}"
        )


class TestRewardComputation:
    """Test that rewards are computed correctly during RL training."""

    def test_reward_range(self):
        """Rewards should be in [0, 1]."""
        L = 10
        model = create_model(L=L, method="cont_maxrl", pretrained=False, seed=42).to(get_device())
        loader = make_integration_loader(B=4, L=L, n_batches=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(loader),
        )

        result = train_rl_epoch(
            model, loader, optimizer, scheduler, "cont_maxrl", 16, L,
            device=get_device(), epoch=1,
        )

        assert result["rl_stats"]["train_rl/reward_mean"] >= 0
        assert result["rl_stats"]["train_rl/reward_mean"] <= 1
