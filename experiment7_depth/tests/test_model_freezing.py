"""Tests for DORN-style model freezing."""

import torch
import torch.nn as nn
import pytest
from experiment7_depth.model import create_model, convert_decoder_bns_to_sync


def test_conv1_frozen():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    for p in model.enc_conv1.parameters():
        assert not p.requires_grad
    for p in model.enc_bn1.parameters():
        assert not p.requires_grad


def test_layer1_frozen():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    for p in model.enc_layer1.parameters():
        assert not p.requires_grad


def test_encoder_bn_frozen_in_layer234():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    for layer in [model.enc_layer2, model.enc_layer3, model.enc_layer4]:
        for m in layer.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    assert not p.requires_grad


def test_layer234_convs_trainable():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    for layer in [model.enc_layer2, model.enc_layer3, model.enc_layer4]:
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                for p in m.parameters():
                    assert p.requires_grad


def test_decoder_all_trainable():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    for dec in [model.dec1, model.dec2, model.dec3, model.dec4, model.dec5, model.head]:
        for p in dec.parameters():
            assert p.requires_grad


def test_frozen_bn_stays_in_eval_mode_during_train():
    """After model.train(), frozen BNs must still be in eval mode."""
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    model.train()
    assert not model.enc_bn1.training
    for layer in [model.enc_layer1, model.enc_layer2, model.enc_layer3, model.enc_layer4]:
        for m in layer.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert not m.training


def test_running_stats_unchanged_after_forward():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    model.train()
    initial_mean = model.enc_bn1.running_mean.clone()
    initial_var = model.enc_bn1.running_var.clone()
    for _ in range(3):
        _ = model(torch.randn(2, 3, 352, 256))
    assert torch.equal(model.enc_bn1.running_mean, initial_mean)
    assert torch.equal(model.enc_bn1.running_var, initial_var)


def test_frozen_params_zero_grad_after_backward():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    model.train()
    out = model(torch.randn(2, 3, 352, 256))
    out.sum().backward()
    for p in model.enc_conv1.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0
    for p in model.enc_layer1.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_trainable_params_nonzero_grad():
    model = create_model(K=100, method="ordinal_ce", pretrained=True, seed=0)
    model.train()
    out = model(torch.randn(2, 3, 352, 256))
    out.sum().backward()
    grad_sum = 0.0
    for p in model.dec1.parameters():
        if p.grad is not None:
            grad_sum += p.grad.abs().sum().item()
    assert grad_sum > 0


def test_syncbn_conversion():
    """Verify decoder BNs are converted to SyncBatchNorm and encoder BNs are not."""
    model = create_model(K=100, method="ordinal_ce", pretrained=False, seed=0)
    model = convert_decoder_bns_to_sync(model)

    for name in ["dec1", "dec2", "dec3", "dec4", "dec5"]:
        dec = getattr(model, name)
        for m in dec.modules():
            if "BatchNorm" in type(m).__name__:
                assert "SyncBatchNorm" in type(m).__name__, \
                    f"Decoder {name} BN not converted: {type(m).__name__}"

    for layer in [model.enc_layer2, model.enc_layer3, model.enc_layer4]:
        for m in layer.modules():
            if "BatchNorm" in type(m).__name__:
                assert type(m).__name__ == "BatchNorm2d", \
                    f"Encoder BN should not be SyncBN: {type(m).__name__}"
