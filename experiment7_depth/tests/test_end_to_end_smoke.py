"""End-to-end smoke tests on synthetic data."""

import torch
import pytest
from experiment7_depth.model import create_model
from experiment7_depth.losses import dense_ordinal_ce_loss
from experiment7_depth.train import train_supervised_step, train_rl_step
from experiment7_depth.tests.conftest import create_synthetic_batch, requires_gpu


@requires_gpu
def test_supervised_ordinal_ce_smoke():
    """Run 20 training steps on synthetic data, verify loss decreases."""
    model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0).cuda()
    model.train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2, momentum=0.9, weight_decay=5e-4
    )
    losses = []
    for step in range(20):
        batch = create_synthetic_batch(B=2, H=32, W=32, K=10)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
        result = train_supervised_step(model, batch, opt, None, "ordinal_ce", K=10)
        losses.append(result["loss"])
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@requires_gpu
def test_rl_cont_maxrl_smoke():
    """Run 10 RL training steps, verify loss is finite."""
    model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0).cuda()
    model.train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, momentum=0.9
    )
    for step in range(10):
        batch = create_synthetic_batch(B=2, H=16, W=16, K=10)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
        result = train_rl_step(model, batch, opt, None, "cont_maxrl", G=4, K=10)
        assert torch.isfinite(torch.tensor(result["loss"])), f"Non-finite loss at step {step}"


@requires_gpu
def test_supervised_plain_ce_smoke():
    """Verify plain_ce also trains."""
    model = create_model(K=10, method="ordinal_ce", pretrained=False, seed=0).cuda()
    model.train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2, momentum=0.9
    )
    losses = []
    for step in range(15):
        batch = create_synthetic_batch(B=2, H=32, W=32, K=10)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
        result = train_supervised_step(model, batch, opt, None, "plain_ce", K=10)
        losses.append(result["loss"])
    assert losses[-1] < losses[0]


@requires_gpu
def test_supervised_mse_smoke():
    """Verify MSE also trains."""
    model = create_model(K=10, method="mse", pretrained=False, seed=0).cuda()
    model.train()
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-2, momentum=0.9
    )
    losses = []
    for step in range(15):
        batch = create_synthetic_batch(B=2, H=32, W=32, K=10)
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
        result = train_supervised_step(model, batch, opt, None, "mse", K=10)
        losses.append(result["loss"])
    assert losses[-1] < losses[0]
