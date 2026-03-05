"""Shared fixtures for experiment5_ordinal tests."""

import os
import pytest
import torch

DATA_DIR = "/projects/bgmo/sramasubramanian/cont-maxrl/data/utkface"

# Auto-detect device once at import time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def device():
    return DEVICE


@pytest.fixture
def data_dir():
    return DATA_DIR


def get_device():
    """Get the test device (GPU if available, else CPU)."""
    return DEVICE


def has_data():
    """Check if UTKFace data is available."""
    return os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 100


# Skip marker for tests that require the dataset
requires_data = pytest.mark.skipif(
    not has_data(), reason="UTKFace dataset not available"
)

# Skip marker for tests that require GPU
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
