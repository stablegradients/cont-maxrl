"""Tests for dense loss wrappers: dense_mse_loss, edge cases."""

import torch
import torch.nn.functional as F
import pytest
from experiment7_depth.losses import dense_mse_loss, dense_plain_ce_loss, dense_ordinal_ce_loss


def test_dense_mse_loss_correctness():
    """dense_mse_loss matches F.mse_loss on valid pixels."""
    torch.manual_seed(0)
    B, H, W = 2, 8, 8
    pred_m = torch.rand(B, 1, H, W) * 9.5 + 0.5
    target_m = torch.rand(B, H, W) * 9.5 + 0.5
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    valid_mask[:, :2, :] = False

    actual = dense_mse_loss(pred_m, target_m, valid_mask)

    pred_flat = pred_m.squeeze(1).reshape(-1)
    target_flat = target_m.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    expected = F.mse_loss(pred_flat[valid_flat], target_flat[valid_flat])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_dense_mse_empty_mask():
    pred_m = torch.rand(2, 1, 4, 4)
    target_m = torch.rand(2, 4, 4)
    valid_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    loss = dense_mse_loss(pred_m, target_m, valid_mask)
    assert loss.item() == 0.0


def test_dense_plain_ce_empty_mask():
    logits = torch.randn(2, 10, 4, 4)
    target_bin = torch.zeros(2, 4, 4, dtype=torch.long)
    valid_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    loss = dense_plain_ce_loss(logits, target_bin, valid_mask)
    assert loss.item() == 0.0


def test_dense_ordinal_ce_gradient_flows():
    """Ensure gradient flows through dense_ordinal_ce_loss."""
    torch.manual_seed(0)
    logits = torch.randn(2, 50, 8, 8, requires_grad=True)
    target_bin = torch.randint(0, 50, (2, 8, 8))
    valid_mask = torch.ones(2, 8, 8, dtype=torch.bool)
    loss = dense_ordinal_ce_loss(logits, target_bin, valid_mask, K=50)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
