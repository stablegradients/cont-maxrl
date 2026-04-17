"""Shared fixtures for experiment9_imagenet_localization tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (run with -m slow)")

import os
import random

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = "/zfsauton/datasets/imagenet"

# Specific validation annotation file required by the localization task.
_LOC_VAL_CSV = os.path.join(DATA_DIR, "LOC_val_solution.csv")

# Auto-detect device once at import time.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Autouse seed fixture — runs before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all RNGs for reproducibility before each test."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)


# ---------------------------------------------------------------------------
# Device / data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Pytest fixture exposing the auto-detected compute device."""
    return DEVICE


@pytest.fixture
def data_dir():
    """Pytest fixture exposing the ImageNet data directory path."""
    return DATA_DIR


# ---------------------------------------------------------------------------
# Helper predicates (used by skip markers below)
# ---------------------------------------------------------------------------


def _has_imagenet_data() -> bool:
    """Return True only when the ImageNet directory and val CSV both exist."""
    return os.path.isdir(DATA_DIR) and os.path.isfile(_LOC_VAL_CSV)


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

# Tests that stream from the ImageNet dataset or parse the val CSV.
requires_data = pytest.mark.skipif(
    not _has_imagenet_data(),
    reason=(
        f"ImageNet data not available: expected directory {DATA_DIR!r} "
        f"and file {_LOC_VAL_CSV!r}"
    ),
)

# Tests that exercise GPU-specific code paths.
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
