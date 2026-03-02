"""Tests for model creation.

Covers:
- Correct output dimensions for softmax and regression heads
- Pretrained weight loading
- Seed reproducibility for FC layer initialization
- Forward pass shape correctness
- Parameter count
"""

import pytest
import torch

from experiment5_ordinal.model import create_model


class TestModelCreation:
    """Basic model creation tests."""

    @pytest.mark.parametrize("L", [2, 10, 100])
    def test_softmax_head_output_shape(self, L):
        """Softmax head should produce (B, L) output."""
        model = create_model(L=L, method="ordinal_ce", pretrained=False, seed=42)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, L)

    def test_mse_head_output_shape(self):
        """MSE head should produce (B, 1) output."""
        model = create_model(L=100, method="mse", pretrained=False, seed=42)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1)

    def test_pretrained_weights(self):
        """Pretrained model should have non-random conv weights."""
        model_pt = create_model(L=100, method="ordinal_ce", pretrained=True, seed=42)
        model_rnd = create_model(L=100, method="ordinal_ce", pretrained=False, seed=42)
        # Conv1 weights should differ between pretrained and random
        w_pt = model_pt.conv1.weight.data.flatten()[:10]
        w_rnd = model_rnd.conv1.weight.data.flatten()[:10]
        assert not torch.allclose(w_pt, w_rnd, atol=1e-3)


class TestSeedReproducibility:
    """FC layer initialization should be reproducible."""

    def test_same_seed_same_fc(self):
        """Same seed should produce identical FC weights."""
        m1 = create_model(L=100, method="ordinal_ce", pretrained=True, seed=42)
        m2 = create_model(L=100, method="ordinal_ce", pretrained=True, seed=42)
        assert torch.allclose(m1.fc.weight, m2.fc.weight)
        assert torch.allclose(m1.fc.bias, m2.fc.bias)

    def test_different_seed_different_fc(self):
        """Different seeds should produce different FC weights."""
        m1 = create_model(L=100, method="ordinal_ce", pretrained=True, seed=42)
        m2 = create_model(L=100, method="ordinal_ce", pretrained=True, seed=43)
        assert not torch.allclose(m1.fc.weight, m2.fc.weight)

    def test_same_backbone_across_seeds(self):
        """Pretrained backbone should be identical across seeds."""
        m1 = create_model(L=100, method="ordinal_ce", pretrained=True, seed=42)
        m2 = create_model(L=100, method="ordinal_ce", pretrained=True, seed=99)
        # Conv1 is part of pretrained backbone, should be same
        assert torch.allclose(m1.conv1.weight, m2.conv1.weight)


class TestModelMethods:
    """Test model variants for different methods."""

    @pytest.mark.parametrize("method", [
        "ordinal_ce", "plain_ce", "cont_maxrl", "binary_maxrl", "grpo", "reinforce"
    ])
    def test_softmax_methods_share_architecture(self, method):
        """All non-MSE methods should use Linear(512, L)."""
        L = 100
        model = create_model(L=L, method=method, pretrained=False, seed=42)
        assert model.fc.in_features == 512
        assert model.fc.out_features == L

    def test_mse_different_architecture(self):
        """MSE should use Linear(512, 1)."""
        model = create_model(L=100, method="mse", pretrained=False, seed=42)
        assert model.fc.in_features == 512
        assert model.fc.out_features == 1


class TestForwardPass:
    """Test forward pass properties."""

    def test_gradient_flows(self):
        """Gradients should flow through the entire model."""
        model = create_model(L=10, method="ordinal_ce", pretrained=False, seed=42)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        out.sum().backward()

        # Check gradient exists for first conv and last fc
        assert model.conv1.weight.grad is not None
        assert model.fc.weight.grad is not None

    def test_different_input_sizes(self):
        """AdaptiveAvgPool should handle various input sizes."""
        model = create_model(L=10, method="ordinal_ce", pretrained=False, seed=42)
        for size in [112, 224, 256]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 10)

    def test_parameter_count(self):
        """ResNet-18 should have ~11.2M params with L=100 head."""
        model = create_model(L=100, method="ordinal_ce", pretrained=False, seed=42)
        total = sum(p.numel() for p in model.parameters())
        # ResNet-18 backbone ≈ 11.17M, fc(512,100) ≈ 51K
        assert 11_000_000 < total < 12_000_000, f"Unexpected param count: {total:,}"
