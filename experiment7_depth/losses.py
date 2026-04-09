import torch
import torch.nn.functional as F


def ordinal_ce_loss_per_example(logits: torch.Tensor, targets: torch.Tensor, K: int) -> torch.Tensor:
    """Vectorized per-example ordinal cross-entropy loss.

    Args:
        logits:  (M, K) raw logits — M is number of valid pixels in the batch
        targets: (M,) long in [0, K-1]
        K: number of bins

    Returns:
        (M,) per-example loss, no reduction.
    """
    M = logits.size(0)
    probs = F.softmax(logits, dim=-1)                               # (M, K)
    zero_col = torch.zeros(M, 1, device=logits.device, dtype=logits.dtype)
    cum = torch.cat([zero_col, probs.cumsum(dim=-1)], dim=-1)       # (M, K+1)

    d = torch.arange(K - 1, device=logits.device)                   # (K-1,)
    targets_2d = targets.unsqueeze(1)                               # (M, 1)
    low_idx  = (targets_2d - d.unsqueeze(0)).clamp(min=0)           # (M, K-1)
    high_idx = (targets_2d + d.unsqueeze(0) + 1).clamp(max=K)       # (M, K-1)

    W = cum.gather(1, high_idx) - cum.gather(1, low_idx)            # (M, K-1)
    W = W.clamp(min=1e-12)

    log_W = torch.log(W)                                             # (M, K-1)
    return -log_W.mean(dim=-1)                                       # (M,)


def ordinal_ce_loss(logits: torch.Tensor, targets: torch.Tensor, K: int) -> torch.Tensor:
    """Batch-mean version. Used in unit tests."""
    return ordinal_ce_loss_per_example(logits, targets, K).mean()


def ordinal_ce_objective_per_example(logits: torch.Tensor, targets: torch.Tensor, K: int) -> torch.Tensor:
    """Same as loss but signed positive (higher is better). For evaluation."""
    return -ordinal_ce_loss_per_example(logits, targets, K)


def compute_window_probs(
    probs: torch.Tensor, targets: torch.Tensor, K: int, d_values: list[int],
) -> dict[int, torch.Tensor]:
    """Vectorized W_d computation for evaluation metrics (cumulative accuracy)."""
    M = probs.size(0)
    zero_col = torch.zeros(M, 1, device=probs.device, dtype=probs.dtype)
    cum = torch.cat([zero_col, probs.cumsum(dim=-1)], dim=-1)
    result = {}
    for d in d_values:
        low  = (targets - d).clamp(min=0)
        high = (targets + d + 1).clamp(max=K)
        upper = cum.gather(1, high.unsqueeze(1)).squeeze(1)
        lower = cum.gather(1, low.unsqueeze(1)).squeeze(1)
        result[d] = (upper - lower).clamp(min=1e-12)
    return result


def dense_ordinal_ce_loss(
    logits: torch.Tensor,         # (B, K, H, W)
    target_bin: torch.Tensor,     # (B, H, W) long
    valid_mask: torch.Tensor,     # (B, H, W) bool
    K: int,
) -> torch.Tensor:
    """Dense ordinal CE over valid pixels only."""
    B, _, H, W = logits.shape
    logits_bhwk = logits.permute(0, 2, 3, 1).contiguous()           # (B, H, W, K)
    logits_flat = logits_bhwk.reshape(B * H * W, K)
    target_flat = target_bin.reshape(B * H * W)
    valid_flat  = valid_mask.reshape(B * H * W)

    logits_valid = logits_flat[valid_flat]
    target_valid = target_flat[valid_flat]

    if logits_valid.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

    return ordinal_ce_loss_per_example(logits_valid, target_valid, K).mean()


def dense_plain_ce_loss(
    logits: torch.Tensor,         # (B, K, H, W)
    target_bin: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Dense plain softmax CE over valid pixels only."""
    B, K, H, W = logits.shape
    logits_bhwk = logits.permute(0, 2, 3, 1).contiguous()
    logits_flat = logits_bhwk.reshape(B * H * W, K)
    target_flat = target_bin.reshape(B * H * W)
    valid_flat  = valid_mask.reshape(B * H * W)
    logits_valid = logits_flat[valid_flat]
    target_valid = target_flat[valid_flat]
    if logits_valid.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits_valid, target_valid)


def dense_mse_loss(
    pred_m: torch.Tensor,         # (B, 1, H, W) raw output in meters
    target_m: torch.Tensor,       # (B, H, W) ground truth meters
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Dense MSE loss in meters over valid pixels."""
    pred_flat = pred_m.squeeze(1).reshape(-1)
    target_flat = target_m.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    pred_valid = pred_flat[valid_flat]
    target_valid = target_flat[valid_flat]
    if pred_valid.numel() == 0:
        return torch.zeros((), device=pred_m.device, dtype=pred_m.dtype)
    return F.mse_loss(pred_valid, target_valid)
