"""Loss functions for ordinal regression experiment.

- ordinal_ce_loss: Ordinal cross-entropy (the supervised gold standard)
- plain CE: F.cross_entropy (standard classification)
- MSE: F.mse_loss with sigmoid output (regression baseline)
"""

import torch
import torch.nn.functional as F


def ordinal_ce_loss(logits: torch.Tensor, targets: torch.Tensor, L: int) -> torch.Tensor:
    """Supervised ordinal cross-entropy loss (to MINIMIZE).

    Computes: (1/(L-1)) * sum_{d=0}^{L-2} [-log W_d(x)]
    where W_d(x) = P(|b_hat - y*| <= d | x) = sum of softmax probs in window.

    Args:
        logits: (B, L) raw logits from the model
        targets: (B,) integer age bins in {0, ..., L-1}
        L: number of age bins (e.g., 100)

    Returns:
        scalar loss, averaged over batch. Positive value.
    """
    B = logits.size(0)
    probs = F.softmax(logits, dim=-1)           # (B, L)
    cum_probs = probs.cumsum(dim=-1)            # (B, L)

    total_neg_log_survival = torch.zeros(B, device=logits.device, dtype=logits.dtype)

    for d in range(L - 1):  # d = 0, 1, ..., L-2
        # Window bounds: [max(0, y*-d), min(L-1, y*+d)]
        low = (targets - d).clamp(min=0)        # (B,)
        high = (targets + d).clamp(max=L - 1)   # (B,)

        # W_d = P(low <= b <= high) via cumulative sum
        upper = cum_probs.gather(1, high.unsqueeze(1)).squeeze(1)   # (B,)
        lower = torch.where(
            low > 0,
            cum_probs.gather(1, (low - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros(B, device=logits.device, dtype=logits.dtype),
        )
        W_d = (upper - lower).clamp(min=1e-12)  # (B,)
        total_neg_log_survival += -torch.log(W_d)

    # Average over (L-1) thresholds, then average over batch
    loss = total_neg_log_survival / (L - 1)     # (B,)
    return loss.mean()


def compute_window_probs(
    probs: torch.Tensor, targets: torch.Tensor, L: int, d_values: list[int]
) -> dict[int, torch.Tensor]:
    """Compute W_d(x) for specified distance values.

    Args:
        probs: (B, L) softmax probabilities
        targets: (B,) integer targets
        L: number of bins
        d_values: list of distance values to compute

    Returns:
        dict mapping d -> (B,) tensor of W_d values
    """
    cum_probs = probs.cumsum(dim=-1)
    B = probs.size(0)
    result = {}

    for d in d_values:
        low = (targets - d).clamp(min=0)
        high = (targets + d).clamp(max=L - 1)
        upper = cum_probs.gather(1, high.unsqueeze(1)).squeeze(1)
        lower = torch.where(
            low > 0,
            cum_probs.gather(1, (low - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros(B, device=probs.device, dtype=probs.dtype),
        )
        result[d] = (upper - lower).clamp(min=1e-12)

    return result


def mse_loss(output: torch.Tensor, targets: torch.Tensor, L: int) -> torch.Tensor:
    """MSE loss for regression baseline.

    Args:
        output: (B, 1) raw model output (pre-sigmoid)
        targets: (B,) integer age bins
        L: number of age bins

    Returns:
        scalar MSE loss
    """
    pred = torch.sigmoid(output.squeeze(-1))  # (B,) in [0, 1]
    normalized_targets = targets.float() / (L - 1)
    return F.mse_loss(pred, normalized_targets)
