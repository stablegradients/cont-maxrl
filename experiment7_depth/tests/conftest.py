"""Shared fixtures for experiment7_depth tests."""

import os
import pytest
import torch
import numpy as np
import random


DATA_DIR = "/zfsauton/scratch/shrinivr/cont-maxrl/data/nyu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def device():
    return DEVICE


@pytest.fixture(params=[10, 100, 1000])
def K(request):
    return request.param


@pytest.fixture(params=[4, 16, 64])
def G(request):
    return request.param


def create_synthetic_batch(B=2, H=32, W=32, K=100):
    """Tiny synthetic batch for unit tests. Don't use NYU in unit tests."""
    image = torch.randn(B, 3, H, W)
    target_m = torch.empty(B, H, W).uniform_(0.5, 10.0)
    bin_edges = torch.linspace(0.5, 10.0, K + 1)
    target_bin = (torch.bucketize(target_m, bin_edges) - 1).clamp(0, K - 1)
    valid_mask = torch.ones(B, H, W, dtype=torch.bool)
    # Mark some pixels invalid to exercise masking paths
    valid_mask[:, :2, :2] = False
    return {
        "image": image,
        "target_bin": target_bin,
        "target_m": target_m,
        "valid_mask": valid_mask,
    }


def has_nyu_data():
    """Check if NYU data is available."""
    return os.path.isdir(DATA_DIR)


def dataset_cache_exists():
    """Check if NYU HF cache exists."""
    return has_nyu_data()


requires_data = pytest.mark.skipif(
    not has_nyu_data(), reason="NYU dataset not available"
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
