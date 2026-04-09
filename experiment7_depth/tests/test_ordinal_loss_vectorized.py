"""Tests for vectorized ordinal CE loss and dense wrappers."""

import torch
import torch.nn.functional as F
import pytest
from experiment7_depth.losses import (
    ordinal_ce_loss_per_example,
    dense_ordinal_ce_loss,
    dense_plain_ce_loss,
)


def naive_ordinal_ce_loss_per_example(logits, targets, K):
    """Loop-based reference for testing."""
    B = logits.size(0)
    probs = F.softmax(logits, dim=-1)
    cum_probs = probs.cumsum(dim=-1)
    total = torch.zeros(B, device=logits.device, dtype=logits.dtype)
    for d in range(K - 1):
        low = (targets - d).clamp(min=0)
        high = (targets + d).clamp(max=K - 1)
        upper = cum_probs.gather(1, high.unsqueeze(1)).squeeze(1)
        lower = torch.where(
            low > 0,
            cum_probs.gather(1, (low - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros(B, device=logits.device, dtype=logits.dtype),
        )
        W_d = (upper - lower).clamp(min=1e-12)
        total += -torch.log(W_d)
    return total / (K - 1)


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_vectorized_matches_naive(K):
    torch.manual_seed(42)
    B = 8
    logits = torch.randn(B, K, dtype=torch.float64)
    targets = torch.randint(0, K, (B,))
    naive = naive_ordinal_ce_loss_per_example(logits, targets, K)
    vec = ordinal_ce_loss_per_example(logits, targets, K)
    assert torch.allclose(naive, vec, atol=1e-10), f"Max diff: {(naive - vec).abs().max()}"


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_vectorized_gradient_matches_naive(K):
    torch.manual_seed(42)
    B = 8
    logits_a = torch.randn(B, K, dtype=torch.float64, requires_grad=True)
    logits_b = logits_a.clone().detach().requires_grad_(True)
    targets = torch.randint(0, K, (B,))
    naive_ordinal_ce_loss_per_example(logits_a, targets, K).sum().backward()
    ordinal_ce_loss_per_example(logits_b, targets, K).sum().backward()
    assert torch.allclose(logits_a.grad, logits_b.grad, atol=1e-5)


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_extreme_targets(K):
    B = 4
    logits = torch.randn(B, K)
    targets = torch.tensor([0, 0, K-1, K-1])
    loss = ordinal_ce_loss_per_example(logits, targets, K)
    assert torch.isfinite(loss).all()


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_one_hot_target_loss_near_zero(K):
    if K < 10:
        pytest.skip()
    B = 4
    targets = torch.tensor([5, min(10, K-1), min(20, K-1), K-5])
    logits = torch.full((B, K), -20.0)
    for i, t in enumerate(targets):
        logits[i, t] = 20.0
    loss = ordinal_ce_loss_per_example(logits, targets, K).mean()
    assert loss.item() < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_speed_at_K1000():
    """Must run in < 50ms per batch on GPU."""
    import time
    K = 1000
    B = 64
    logits = torch.randn(B, K, device="cuda", requires_grad=True)
    targets = torch.randint(0, K, (B,), device="cuda")
    for _ in range(5):  # warmup
        loss = ordinal_ce_loss_per_example(logits, targets, K).mean()
        loss.backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        loss = ordinal_ce_loss_per_example(logits, targets, K).mean()
        loss.backward()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 100
    assert dt < 0.05, f"K=1000 ordinal loss too slow: {dt*1000:.1f}ms"


def test_dense_ordinal_ce_loss_correctness():
    """dense_ordinal_ce_loss correctly applies the per-example loss to valid pixels."""
    torch.manual_seed(0)
    B, K, H, W = 2, 50, 16, 16
    logits = torch.randn(B, K, H, W)
    target_bin = torch.randint(0, K, (B, H, W))
    valid_mask = torch.rand(B, H, W) > 0.3

    # Reference: manual extraction and loss computation
    logits_bhwk = logits.permute(0, 2, 3, 1).contiguous()
    logits_flat = logits_bhwk.reshape(-1, K)
    target_flat = target_bin.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    logits_valid = logits_flat[valid_flat]
    target_valid = target_flat[valid_flat]
    expected = ordinal_ce_loss_per_example(logits_valid, target_valid, K).mean()

    actual = dense_ordinal_ce_loss(logits, target_bin, valid_mask, K)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_dense_ordinal_ce_handles_empty_valid_mask():
    """Returns zero tensor when no valid pixels."""
    logits = torch.randn(2, 10, 8, 8)
    target_bin = torch.zeros(2, 8, 8, dtype=torch.long)
    valid_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    loss = dense_ordinal_ce_loss(logits, target_bin, valid_mask, K=10)
    assert loss.item() == 0.0


def test_dense_plain_ce_matches_F_cross_entropy():
    """dense_plain_ce_loss matches F.cross_entropy on the masked subset."""
    torch.manual_seed(0)
    B, K, H, W = 2, 20, 8, 8
    logits = torch.randn(B, K, H, W)
    target_bin = torch.randint(0, K, (B, H, W))
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    valid_mask[:, :2, :] = False

    actual = dense_plain_ce_loss(logits, target_bin, valid_mask)

    logits_bhwk = logits.permute(0, 2, 3, 1).contiguous()
    logits_flat = logits_bhwk.reshape(-1, K)
    target_flat = target_bin.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    expected = F.cross_entropy(logits_flat[valid_flat], target_flat[valid_flat])
    assert torch.allclose(actual, expected, atol=1e-6)
