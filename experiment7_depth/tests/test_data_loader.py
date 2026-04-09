"""Tests for NYU data loader and synthetic batch helper."""

import torch
import pytest
from experiment7_depth.tests.conftest import create_synthetic_batch, requires_data


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_synthetic_batch_shapes(K):
    batch = create_synthetic_batch(B=2, H=32, W=32, K=K)
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["target_bin"].shape == (2, 32, 32)
    assert batch["target_m"].shape == (2, 32, 32)
    assert batch["valid_mask"].shape == (2, 32, 32)
    assert batch["target_bin"].dtype == torch.long
    assert batch["valid_mask"].dtype == torch.bool


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_target_bin_within_range(K):
    batch = create_synthetic_batch(K=K)
    tb = batch["target_bin"]
    assert tb.min() >= 0 and tb.max() < K


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_target_m_bin_consistency(K):
    batch = create_synthetic_batch(K=K)
    edges = torch.linspace(0.5, 10.0, K + 1)
    expected_bin = (torch.bucketize(batch["target_m"], edges) - 1).clamp(0, K - 1)
    assert torch.equal(expected_bin, batch["target_bin"])


def test_valid_mask_excludes_invalid():
    batch = create_synthetic_batch(K=100)
    assert (~batch["valid_mask"]).any()


def test_bin_depth_helper():
    from experiment7_depth.data import bin_depth
    edges = torch.linspace(0.5, 10.0, 101)
    depth = torch.tensor([0.5, 5.0, 10.0])
    bins = bin_depth(depth, edges, 100)
    assert bins[0].item() == 0
    assert bins[-1].item() == 99


def test_bin_to_meters_helper():
    from experiment7_depth.data import bin_to_meters
    edges = torch.linspace(0.5, 10.0, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = torch.tensor([0, 50, 99])
    meters = bin_to_meters(idx, centers)
    assert meters[0].item() == pytest.approx(centers[0].item())
    assert meters[2].item() == pytest.approx(centers[99].item())


@requires_data
def test_nyu_smoke():
    """Load one batch from real NYU data, verify shapes and ranges."""
    from experiment7_depth.data import get_dataloaders
    train_loader, _, _ = get_dataloaders(
        data_dir="/zfsauton/scratch/shrinivr/cont-maxrl/data/nyu",
        batch_size=2, K=100, num_workers=0,
    )
    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 3, 352, 256)
    assert batch["target_m"].min() >= 0.5
    assert batch["target_m"].max() <= 10.0
    valid_frac = batch["valid_mask"].float().mean().item()
    assert valid_frac > 0.5
