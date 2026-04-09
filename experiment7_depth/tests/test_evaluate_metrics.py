"""Tests for dense evaluation metrics."""

import torch
import pytest


def test_mae_meters_perfect_prediction():
    """Perfect predictions should give zero MAE."""
    pred_m = torch.tensor([1.0, 5.0, 9.0])
    target_m = torch.tensor([1.0, 5.0, 9.0])
    mae = (pred_m - target_m).abs().mean().item()
    assert mae == 0.0


def test_mae_bins_perfect_prediction():
    pred_bin = torch.tensor([0, 5, 99])
    target_bin = torch.tensor([0, 5, 99])
    mae = (pred_bin - target_bin).abs().float().mean().item()
    assert mae == 0.0


def test_delta_threshold():
    """All ratios < 1.25 should give delta_1.25 = 1.0."""
    pred_m = torch.tensor([1.0, 2.0, 3.0])
    target_m = torch.tensor([1.1, 2.1, 3.1])
    ratio = torch.maximum(pred_m / target_m, target_m / pred_m)
    assert (ratio < 1.25).all()


def test_cumulative_accuracy_exact_match():
    """Exact match should give acc_within_0 = 1.0."""
    pred_bin = torch.tensor([5, 10, 20])
    target_bin = torch.tensor([5, 10, 20])
    err = (pred_bin - target_bin).abs()
    assert (err <= 0).all()


def test_cumulative_accuracy_off_by_one():
    """Off-by-one should give acc_within_0 = 0, acc_within_1 = 1."""
    pred_bin = torch.tensor([6, 11, 21])
    target_bin = torch.tensor([5, 10, 20])
    err = (pred_bin - target_bin).abs()
    assert not (err <= 0).all()
    assert (err <= 1).all()


def test_abs_rel_formula():
    pred_m = torch.tensor([2.0])
    target_m = torch.tensor([4.0])
    abs_rel = ((pred_m - target_m).abs() / target_m).item()
    assert abs_rel == pytest.approx(0.5)


def test_rmse_formula():
    pred_m = torch.tensor([1.0, 3.0])
    target_m = torch.tensor([2.0, 4.0])
    rmse = ((pred_m - target_m) ** 2).mean().sqrt().item()
    assert rmse == pytest.approx(1.0)


def test_accumulators_to_metrics():
    """Verify accumulators_to_metrics produces correct averages."""
    from experiment7_depth.evaluate import accumulators_to_metrics
    accum = {
        "sum_abs_err_m": torch.tensor([10.0]),
        "sum_sq_err_m": torch.tensor([20.0]),
        "sum_abs_rel": torch.tensor([5.0]),
        "sum_delta_125": torch.tensor([80.0]),
        "sum_delta_125_sq": torch.tensor([90.0]),
        "sum_delta_125_cb": torch.tensor([95.0]),
        "sum_err_bins": torch.tensor([50.0]),
        "sum_ordinal_obj": torch.tensor([-30.0]),
        "sum_plain_ce_obj": torch.tensor([-40.0]),
        "total_valid": torch.tensor([100.0]),
    }
    for d in [0, 1, 2, 5, 10, 20]:
        accum[f"sum_acc_{d}"] = torch.tensor([50.0 + d])

    metrics = accumulators_to_metrics(accum, method="ordinal_ce")
    assert metrics["val/mae_meters"] == pytest.approx(0.1)
    assert metrics["val/mae_bins"] == pytest.approx(0.5)
    assert metrics["val/abs_rel"] == pytest.approx(0.05)
    assert metrics["val/delta_1.25"] == pytest.approx(0.8)
    assert metrics["val/ordinal_ce_objective"] == pytest.approx(-0.3)
    assert metrics["val/acc_within_0"] == pytest.approx(0.5)
    assert metrics["val/acc_within_20"] == pytest.approx(0.7)


def test_accumulators_to_metrics_mse():
    """MSE method should not have ordinal/CE metrics."""
    from experiment7_depth.evaluate import accumulators_to_metrics
    accum = {
        "sum_abs_err_m": torch.tensor([10.0]),
        "sum_sq_err_m": torch.tensor([20.0]),
        "sum_abs_rel": torch.tensor([5.0]),
        "sum_delta_125": torch.tensor([80.0]),
        "sum_delta_125_sq": torch.tensor([90.0]),
        "sum_delta_125_cb": torch.tensor([95.0]),
        "sum_err_bins": torch.tensor([50.0]),
        "sum_ordinal_obj": torch.tensor([0.0]),
        "sum_plain_ce_obj": torch.tensor([0.0]),
        "total_valid": torch.tensor([100.0]),
    }
    for d in [0, 1, 2, 5, 10, 20]:
        accum[f"sum_acc_{d}"] = torch.tensor([50.0])

    metrics = accumulators_to_metrics(accum, method="mse")
    assert "val/mae_meters" in metrics
    assert "val/ordinal_ce_objective" not in metrics
    assert "val/acc_within_0" not in metrics
