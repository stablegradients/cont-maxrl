"""Tests for RL training step consistency."""

import torch
import torch.nn.functional as F
import pytest
from experiment7_depth.model import create_model
from experiment7_depth.losses import dense_ordinal_ce_loss
from experiment7_depth.advantages import continuous_maxrl_advantages
from experiment7_depth.tests.conftest import create_synthetic_batch, requires_gpu


def test_rl_loss_zero_advantage_gives_zero_loss():
    """If all advantages are zero, PG loss should be zero."""
    torch.manual_seed(0)
    B, K, H, W, G = 1, 10, 8, 8, 4
    logits = torch.randn(B, K, H, W)
    log_probs = F.log_softmax(logits.permute(0, 2, 3, 1), dim=-1)  # (B, H, W, K)
    samples = torch.randint(0, K, (B, H, W, G))
    sample_log_probs = log_probs.gather(-1, samples)  # (B, H, W, G)
    advantages = torch.zeros(B, H, W, G)
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    pg_per_pixel = -(advantages * sample_log_probs).mean(dim=-1)
    loss = (pg_per_pixel * valid_mask.float()).sum() / valid_mask.float().sum()
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@requires_gpu
def test_gos_rl_step_runs():
    """Verify a full RL training step completes for cont_maxrl at K=10, G=4."""
    from experiment7_depth.train import train_rl_step
    model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0).cuda()
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch = create_synthetic_batch(B=2, H=32, W=32, K=10)
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()
    result = train_rl_step(model, batch, opt, None, "cont_maxrl", G=4, K=10)
    assert "loss" in result
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert result["reward_mean"] >= 0
    assert result["reward_mean"] <= 1


@requires_gpu
def test_supervised_step_runs():
    """Verify a supervised training step completes."""
    from experiment7_depth.train import train_supervised_step
    model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0).cuda()
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    batch = create_synthetic_batch(B=2, H=32, W=32, K=10)
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()
    result = train_supervised_step(model, batch, opt, None, "ordinal_ce", K=10)
    assert "loss" in result
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert result["grad_norm"] > 0


@requires_gpu
def test_all_rl_methods_run():
    """Verify all RL methods complete a step without error."""
    from experiment7_depth.train import train_rl_step
    for method in ["cont_maxrl", "cont_maxrl_no_bl", "binary_maxrl", "grpo", "reinforce"]:
        model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0).cuda()
        model.train()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        batch = create_synthetic_batch(B=2, H=16, W=16, K=10)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
        result = train_rl_step(model, batch, opt, None, method, G=4, K=10)
        assert torch.isfinite(torch.tensor(result["loss"])), f"Non-finite loss for {method}"


@requires_gpu
def test_all_supervised_methods_run():
    """Verify all supervised methods complete a step without error."""
    from experiment7_depth.train import train_supervised_step
    for method in ["ordinal_ce", "plain_ce", "mse"]:
        K = 10
        m = "mse" if method == "mse" else "ordinal_ce"
        model = create_model(K=K, method=m, pretrained=False, seed=0).cuda()
        model.train()
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        batch = create_synthetic_batch(B=2, H=16, W=16, K=K)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
        result = train_supervised_step(model, batch, opt, None, method, K=K)
        assert torch.isfinite(torch.tensor(result["loss"])), f"Non-finite loss for {method}"


def test_compute_grad_norm():
    """Verify grad norm computation."""
    from experiment7_depth.train import compute_grad_norm
    model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0)
    model.train()
    out = model(torch.randn(1, 3, 32, 32))
    out.sum().backward()
    gn = compute_grad_norm(model)
    assert gn > 0


def test_poly_lr_schedule():
    """Verify polynomial LR schedule properties."""
    from experiment7_depth.train import get_poly_lr
    base_lr = 2e-4
    total = 1000
    # At step 0, should be base_lr
    assert get_poly_lr(0, total, base_lr) == pytest.approx(base_lr)
    # Monotonically decreasing
    prev = base_lr
    for s in range(1, total):
        lr = get_poly_lr(s, total, base_lr)
        assert lr < prev
        prev = lr
    # At final step, should be near zero but positive
    final_lr = get_poly_lr(total - 1, total, base_lr)
    assert final_lr > 0
    assert final_lr < base_lr * 0.1
