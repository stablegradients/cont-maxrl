"""Tests for model forward pass and output shapes."""

import torch
import pytest
from experiment7_depth.model import create_model


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_output_shapes(K):
    model = create_model(K=K, method="ordinal_ce", pretrained=False, seed=0)
    images = torch.randn(2, 3, 352, 256)
    out = model(images)
    assert out.shape == (2, K, 352, 256)


def test_mse_output_shape():
    model = create_model(K=100, method="mse", pretrained=False, seed=0)
    out = model(torch.randn(2, 3, 352, 256))
    assert out.shape == (2, 1, 352, 256)


def test_decoder_upsampling_correctness():
    """Verify encoder-decoder spatial dims work for 352x256 input."""
    model = create_model(K=100, method="ordinal_ce", pretrained=False, seed=0)
    out = model(torch.randn(1, 3, 352, 256))
    assert out.shape[-2:] == (352, 256), "Output spatial dims do not match input"


def test_different_methods_same_decoder():
    """All non-mse methods produce K-channel output."""
    for method in ["ordinal_ce", "plain_ce", "cont_maxrl", "grpo"]:
        model = create_model(K=50, method=method, pretrained=False, seed=0)
        out = model(torch.randn(1, 3, 352, 256))
        assert out.shape == (1, 50, 352, 256), f"Failed for method={method}"


def test_param_count_approximate():
    """Verify total params are in expected range (~14M)."""
    model = create_model(K=100, method="ordinal_ce", pretrained=False, seed=0)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    assert 13_000_000 < total < 16_000_000, f"Total params {total} outside expected range"
    assert frozen > 100_000, f"Too few frozen params: {frozen}"
    assert trainable > 13_000_000, f"Too few trainable params: {trainable}"
