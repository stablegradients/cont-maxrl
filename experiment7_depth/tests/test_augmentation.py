"""Tests for data augmentation atomicity."""

import torch
import pytest
from experiment7_depth.data import apply_horizontal_flip


def test_flip_is_atomic():
    H, W = 32, 32
    image = torch.randn(3, H, W)
    target_bin = torch.randint(0, 100, (H, W))
    target_m = torch.rand(H, W) * 9.5 + 0.5
    valid = torch.ones(H, W, dtype=torch.bool)
    valid[:, 0] = False  # mark first column invalid

    f_image, f_bin, f_m, f_valid = apply_horizontal_flip(image, target_bin, target_m, valid)

    assert torch.equal(f_image, torch.flip(image, dims=[-1]))
    assert torch.equal(f_bin, torch.flip(target_bin, dims=[-1]))
    assert torch.equal(f_m, torch.flip(target_m, dims=[-1]))
    assert torch.equal(f_valid, torch.flip(valid, dims=[-1]))
    # After flip, the invalid column should be the last, not the first
    assert not f_valid[:, -1].any()
    assert f_valid[:, 0].all()


def test_flip_preserves_dtype():
    image = torch.randn(3, 16, 16)
    target_bin = torch.randint(0, 10, (16, 16))
    target_m = torch.rand(16, 16)
    valid = torch.ones(16, 16, dtype=torch.bool)
    f_image, f_bin, f_m, f_valid = apply_horizontal_flip(image, target_bin, target_m, valid)
    assert f_image.dtype == torch.float32
    assert f_bin.dtype == torch.long
    assert f_valid.dtype == torch.bool
