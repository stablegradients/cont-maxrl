"""Dense evaluation metrics for depth estimation."""

import torch
import torch.nn.functional as F

from .losses import ordinal_ce_objective_per_example


@torch.no_grad()
def evaluate_model(model, dataloader, K, method, device, bin_centers, bin_edges):
    """Evaluate model on a dataloader, computing dense depth metrics.

    Args:
        model: DepthResNet18UNet (possibly DDP-wrapped)
        dataloader: val or test DataLoader
        K: number of bins
        method: training method name
        device: torch device
        bin_centers: (K,) tensor of bin center values in meters
        bin_edges: (K+1,) tensor of bin edge values

    Returns:
        dict of metric_name -> float value (already averaged)
    """
    model.eval()

    bin_centers = bin_centers.to(device)
    bin_edges = bin_edges.to(device)

    # Accumulators
    total_valid = 0
    sum_abs_err_m = 0.0
    sum_sq_err_m = 0.0
    sum_abs_rel = 0.0
    sum_delta_125 = 0
    sum_delta_125_sq = 0
    sum_delta_125_cb = 0
    sum_err_bins = 0
    sum_ordinal_obj = 0.0
    sum_plain_ce_obj = 0.0

    acc_thresholds = [0, 1, 2, 5, 10, 20]
    acc_sums = {d: 0 for d in acc_thresholds}

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        target_bin = batch["target_bin"].to(device)
        target_m = batch["target_m"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)

        if method == "mse":
            pred_m = output.squeeze(1).float().clamp(0.5, 10.0)   # (B, H, W)
            pred_bin = (torch.bucketize(pred_m, bin_edges) - 1).clamp(0, K - 1)
        else:
            logits = output.float()                                 # (B, K, H, W)
            probs = F.softmax(logits, dim=1)
            pred_bin = probs.argmax(dim=1)                          # (B, H, W)
            pred_m = bin_centers[pred_bin]                           # (B, H, W)

            # Ordinal CE and plain CE objectives per valid pixel
            B, _, H, W = logits.shape
            logits_bhwk = logits.permute(0, 2, 3, 1).reshape(B * H * W, K)
            target_flat = target_bin.reshape(B * H * W)
            valid_flat = valid_mask.reshape(B * H * W)
            logits_valid = logits_bhwk[valid_flat]
            target_valid = target_flat[valid_flat]

            if logits_valid.numel() > 0:
                ordinal_obj_valid = ordinal_ce_objective_per_example(
                    logits_valid, target_valid, K
                )
                plain_ce_valid = F.log_softmax(logits_valid, dim=-1).gather(
                    1, target_valid.unsqueeze(1)
                ).squeeze(1)
                sum_ordinal_obj += ordinal_obj_valid.sum().item()
                sum_plain_ce_obj += plain_ce_valid.sum().item()

        # Valid pixels
        vm = valid_mask
        target_m_v = target_m[vm]
        pred_m_v = pred_m[vm]
        target_bin_v = target_bin[vm]
        pred_bin_v = pred_bin[vm]

        num = pred_m_v.numel()
        total_valid += num

        # MAE / RMSE in meters
        abs_err_m = (pred_m_v - target_m_v).abs()
        sum_abs_err_m += abs_err_m.sum().item()
        sum_sq_err_m += (abs_err_m ** 2).sum().item()

        # Abs rel
        sum_abs_rel += (abs_err_m / target_m_v.clamp(min=1e-6)).sum().item()

        # Delta thresholds
        ratio = torch.maximum(pred_m_v / target_m_v.clamp(min=1e-6),
                              target_m_v / pred_m_v.clamp(min=1e-6))
        sum_delta_125 += (ratio < 1.25).sum().item()
        sum_delta_125_sq += (ratio < 1.25 ** 2).sum().item()
        sum_delta_125_cb += (ratio < 1.25 ** 3).sum().item()

        # MAE in bins
        err_bins = (pred_bin_v.long() - target_bin_v.long()).abs()
        sum_err_bins += err_bins.sum().item()

        # Cumulative accuracy thresholds
        for d in acc_thresholds:
            acc_sums[d] += (err_bins <= d).sum().item()

    if total_valid == 0:
        return {"val/mae_meters": float("inf"), "val/total_valid": 0}

    metrics = {
        "val/mae_meters": sum_abs_err_m / total_valid,
        "val/rmse_meters": (sum_sq_err_m / total_valid) ** 0.5,
        "val/mae_bins": sum_err_bins / total_valid,
        "val/abs_rel": sum_abs_rel / total_valid,
        "val/delta_1.25": sum_delta_125 / total_valid,
        "val/delta_1.25^2": sum_delta_125_sq / total_valid,
        "val/delta_1.25^3": sum_delta_125_cb / total_valid,
        "val/total_valid": total_valid,
    }

    if method != "mse":
        metrics["val/ordinal_ce_objective"] = sum_ordinal_obj / total_valid
        metrics["val/plain_ce_objective"] = sum_plain_ce_obj / total_valid
        for d in acc_thresholds:
            metrics[f"val/acc_within_{d}"] = acc_sums[d] / total_valid

    return metrics


@torch.no_grad()
def evaluate_accumulators(model, dataloader, K, method, device, bin_centers, bin_edges):
    """Same as evaluate_model but returns raw accumulator tensors for DDP all-reduce.

    Returns:
        dict of accumulator_name -> torch.Tensor (on device, shape (1,))
    """
    model.eval()

    bin_centers = bin_centers.to(device)
    bin_edges = bin_edges.to(device)

    accum = {
        "sum_abs_err_m": torch.zeros(1, device=device),
        "sum_sq_err_m": torch.zeros(1, device=device),
        "sum_abs_rel": torch.zeros(1, device=device),
        "sum_delta_125": torch.zeros(1, device=device),
        "sum_delta_125_sq": torch.zeros(1, device=device),
        "sum_delta_125_cb": torch.zeros(1, device=device),
        "sum_err_bins": torch.zeros(1, device=device),
        "sum_ordinal_obj": torch.zeros(1, device=device),
        "sum_plain_ce_obj": torch.zeros(1, device=device),
        "total_valid": torch.zeros(1, device=device),
    }
    acc_thresholds = [0, 1, 2, 5, 10, 20]
    for d in acc_thresholds:
        accum[f"sum_acc_{d}"] = torch.zeros(1, device=device)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        target_bin = batch["target_bin"].to(device)
        target_m = batch["target_m"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images)

        if method == "mse":
            pred_m = output.squeeze(1).float().clamp(0.5, 10.0)
            pred_bin = (torch.bucketize(pred_m, bin_edges) - 1).clamp(0, K - 1)
        else:
            logits = output.float()
            probs = F.softmax(logits, dim=1)
            pred_bin = probs.argmax(dim=1)
            pred_m = bin_centers[pred_bin]

            B, _, H, W = logits.shape
            logits_bhwk = logits.permute(0, 2, 3, 1).reshape(B * H * W, K)
            target_flat = target_bin.reshape(B * H * W)
            valid_flat = valid_mask.reshape(B * H * W)
            logits_valid = logits_bhwk[valid_flat]
            target_valid = target_flat[valid_flat]

            if logits_valid.numel() > 0:
                ordinal_obj_valid = ordinal_ce_objective_per_example(
                    logits_valid, target_valid, K
                )
                plain_ce_valid = F.log_softmax(logits_valid, dim=-1).gather(
                    1, target_valid.unsqueeze(1)
                ).squeeze(1)
                accum["sum_ordinal_obj"] += ordinal_obj_valid.sum()
                accum["sum_plain_ce_obj"] += plain_ce_valid.sum()

        vm = valid_mask
        target_m_v = target_m[vm]
        pred_m_v = pred_m[vm]
        target_bin_v = target_bin[vm]
        pred_bin_v = pred_bin[vm]

        num = pred_m_v.numel()
        accum["total_valid"] += num

        abs_err_m = (pred_m_v - target_m_v).abs()
        accum["sum_abs_err_m"] += abs_err_m.sum()
        accum["sum_sq_err_m"] += (abs_err_m ** 2).sum()
        accum["sum_abs_rel"] += (abs_err_m / target_m_v.clamp(min=1e-6)).sum()

        ratio = torch.maximum(pred_m_v / target_m_v.clamp(min=1e-6),
                              target_m_v / pred_m_v.clamp(min=1e-6))
        accum["sum_delta_125"] += (ratio < 1.25).sum()
        accum["sum_delta_125_sq"] += (ratio < 1.25 ** 2).sum()
        accum["sum_delta_125_cb"] += (ratio < 1.25 ** 3).sum()

        err_bins = (pred_bin_v.long() - target_bin_v.long()).abs()
        accum["sum_err_bins"] += err_bins.sum()

        for d in acc_thresholds:
            accum[f"sum_acc_{d}"] += (err_bins <= d).sum()

    return accum


def accumulators_to_metrics(accum, method):
    """Convert raw accumulators to final metric values.

    Args:
        accum: dict from evaluate_accumulators (after all-reduce if DDP)
        method: training method name

    Returns:
        dict of metric_name -> float
    """
    total = accum["total_valid"].item()
    if total == 0:
        return {"val/mae_meters": float("inf"), "val/total_valid": 0}

    metrics = {
        "val/mae_meters": accum["sum_abs_err_m"].item() / total,
        "val/rmse_meters": (accum["sum_sq_err_m"].item() / total) ** 0.5,
        "val/mae_bins": accum["sum_err_bins"].item() / total,
        "val/abs_rel": accum["sum_abs_rel"].item() / total,
        "val/delta_1.25": accum["sum_delta_125"].item() / total,
        "val/delta_1.25^2": accum["sum_delta_125_sq"].item() / total,
        "val/delta_1.25^3": accum["sum_delta_125_cb"].item() / total,
        "val/total_valid": total,
    }

    if method != "mse":
        metrics["val/ordinal_ce_objective"] = accum["sum_ordinal_obj"].item() / total
        metrics["val/plain_ce_objective"] = accum["sum_plain_ce_obj"].item() / total
        for d in [0, 1, 2, 5, 10, 20]:
            metrics[f"val/acc_within_{d}"] = accum[f"sum_acc_{d}"].item() / total

    return metrics
