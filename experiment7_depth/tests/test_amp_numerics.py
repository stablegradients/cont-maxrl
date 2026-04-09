"""Tests for AMP bf16 numerical stability."""

import torch
import pytest
from experiment7_depth.losses import ordinal_ce_loss_per_example
from experiment7_depth.tests.conftest import requires_gpu


@requires_gpu
def test_ordinal_loss_amp_close_to_fp32():
    """bf16 ordinal loss should be close to fp32 within tolerance."""
    K = 100
    B = 8
    logits = torch.randn(B, K, device="cuda")
    targets = torch.randint(0, K, (B,), device="cuda")

    loss_fp32 = ordinal_ce_loss_per_example(logits.float(), targets, K)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss_bf16 = ordinal_ce_loss_per_example(logits, targets, K)

    assert torch.allclose(loss_fp32, loss_bf16.float(), atol=0.01, rtol=0.05)


@requires_gpu
def test_ordinal_loss_amp_at_K1000():
    """Verify bf16 doesn't blow up at K=1000."""
    K = 1000
    B = 4
    logits = torch.randn(B, K, device="cuda")
    targets = torch.randint(0, K, (B,), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = ordinal_ce_loss_per_example(logits, targets, K)

    assert torch.isfinite(loss).all()
