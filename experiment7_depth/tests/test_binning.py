"""Tests for depth binning math."""

import torch
import pytest


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_bin_edges_count(K):
    edges = torch.linspace(0.5, 10.0, K + 1)
    assert edges.numel() == K + 1
    assert edges[0].item() == pytest.approx(0.5)
    assert edges[-1].item() == pytest.approx(10.0)


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_bucketize_roundtrip(K):
    edges = torch.linspace(0.5, 10.0, K + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = (10.0 - 0.5) / K
    depth = torch.empty(1000).uniform_(0.5, 10.0)
    bin_idx = (torch.bucketize(depth, edges) - 1).clamp(0, K - 1)
    recovered = centers[bin_idx]
    max_err = (depth - recovered).abs().max().item()
    assert max_err < bin_width


@pytest.mark.parametrize("K", [10, 100, 1000])
def test_boundary_values(K):
    edges = torch.linspace(0.5, 10.0, K + 1)
    depth = torch.tensor([0.5, 10.0])
    bin_idx = (torch.bucketize(depth, edges) - 1).clamp(0, K - 1)
    assert bin_idx[0].item() == 0
    assert bin_idx[1].item() == K - 1
