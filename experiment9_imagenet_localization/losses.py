"""Loss functions for experiment9_imagenet_localization.

- factored_sample_log_prob: sum of per-head log-probs at sampled indices
- ordinal_ce_single_head: ordinal cross-entropy (window-survival form, per head)
- localization_ordinal_ce_loss: mean of per-head ordinal CE over 4 coordinate heads
- localization_plain_ce_loss: mean of per-head F.cross_entropy over 4 heads
- localization_mse_loss: MSE against the primary GT box (regression baseline)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")


# ---------------------------------------------------------------------------
# 1. Factored sample log-prob (the #1 trap for newcomers)
# ---------------------------------------------------------------------------

def factored_sample_log_prob(
    log_probs: dict[str, torch.Tensor],   # each (B, K)
    samples:   dict[str, torch.Tensor],   # each (B, N) long
) -> torch.Tensor:
    """Sum of per-head log-probs at sampled bin indices.

    log π(sample | I) = Σ_h log π_h(sample_h | I)

    Args:
        log_probs: dict keyed by HEAD_NAMES, each value shape (B, K)
                   — log-probabilities (log_softmax output) from one head.
        samples:   dict keyed by HEAD_NAMES, each value shape (B, N) long
                   — sampled bin indices per head.

    Returns:
        (B, N) tensor of summed log-probs. NOT (B, N, 4): summed across heads.

    The #1 trap: gathering across heads without summing leaves a (B, N, 4)
    tensor that silently broadcasts-wrong in downstream losses.
    """
    # For each head h: log_probs[h] has shape (B, K); samples[h] has shape (B, N).
    # gather(1, samples[h]) -> (B, N).
    # Sum across heads.
    out = None
    for h in HEAD_NAMES:
        per_head = log_probs[h].gather(1, samples[h])    # (B, N)
        out = per_head if out is None else out + per_head
    return out  # (B, N)


# ---------------------------------------------------------------------------
# 2. Single-head ordinal CE (window-survival form — from experiment5_ordinal)
# ---------------------------------------------------------------------------

def ordinal_ce_single_head(
    logits: torch.Tensor,     # (B, K)
    targets: torch.Tensor,    # (B,) long in {0, ..., K-1}
    K: int,
) -> torch.Tensor:
    """Single-head ordinal cross-entropy (window-survival form).

    L = mean_b [ (1/(K-1)) * sum_{d=0}^{K-2} -log W_d(b, target_b) ]
    where W_d(b, y) = P( |bin - y| <= d | b ) = sum of softmax probs in the window.

    At K=2, this reduces to standard cross-entropy — which is the ordinal_ce
    ≡ plain_ce binary-recovery identity Task 6 has a test for.

    Args:
        logits:   (B, K) raw logits
        targets:  (B,)    long bin indices
        K:        number of bins

    Returns:
        scalar loss, positive value, averaged over batch.
    """
    # Adapt from experiment5_ordinal/losses.py::ordinal_ce_loss.
    B = logits.size(0)
    probs = F.softmax(logits, dim=-1)           # (B, K)
    cum_probs = probs.cumsum(dim=-1)            # (B, K)

    total_neg_log_survival = torch.zeros(B, device=logits.device, dtype=logits.dtype)

    for d in range(K - 1):  # d = 0, 1, ..., K-2
        # Window bounds: [max(0, y*-d), min(K-1, y*+d)]
        low = (targets - d).clamp(min=0)        # (B,)
        high = (targets + d).clamp(max=K - 1)   # (B,)

        # W_d = P(low <= b <= high) via cumulative sum
        upper = cum_probs.gather(1, high.unsqueeze(1)).squeeze(1)   # (B,)
        lower = torch.where(
            low > 0,
            cum_probs.gather(1, (low - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.zeros(B, device=logits.device, dtype=logits.dtype),
        )
        W_d = (upper - lower).clamp(min=1e-12)  # (B,)
        total_neg_log_survival += -torch.log(W_d)

    # Average over (K-1) thresholds, then average over batch
    loss = total_neg_log_survival / (K - 1)     # (B,)
    return loss.mean()


# ---------------------------------------------------------------------------
# 3. Factored supervised losses (sum over 4 heads, divided by 4)
# ---------------------------------------------------------------------------

def localization_ordinal_ce_loss(
    logits: dict[str, torch.Tensor],    # each (B, K)
    targets: dict[str, torch.Tensor],   # each (B,) long
    K: int,
) -> torch.Tensor:
    """Mean of per-head ordinal CE losses over the 4 coordinate heads.

    Args:
        logits:  dict keyed by HEAD_NAMES, each (B, K)
        targets: dict keyed by HEAD_NAMES, each (B,) long
        K:       number of bins

    Returns:
        scalar loss — mean over the 4 heads.
    """
    losses = [
        ordinal_ce_single_head(logits[h], targets[h], K)
        for h in HEAD_NAMES
    ]
    return sum(losses) / 4.0


def localization_plain_ce_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean of per-head `F.cross_entropy` losses over 4 coordinate heads."""
    losses = [
        F.cross_entropy(logits[h], targets[h])
        for h in HEAD_NAMES
    ]
    return sum(losses) / 4.0


def localization_mse_loss(
    pred: torch.Tensor,        # (B, 4) in [0, 1]
    gt_primary_box: torch.Tensor,  # (B, 4) in [0, 1]
) -> torch.Tensor:
    """MSE loss against the PRIMARY GT box (legacy — kept for reproducibility).

    Uses `gt_boxes[:, 0, :]` (first slot in CSV). This is biased on multi-GT
    images — use `localization_mse_iou_match_loss` instead for new runs.
    """
    return F.mse_loss(pred, gt_primary_box)


def _centroid_match_target(
    pred: torch.Tensor,       # (B, 4) xywh
    gt_boxes: torch.Tensor,   # (B, M, 4) xywh
    gt_mask: torch.Tensor,    # (B, M) bool
) -> torch.Tensor:
    """Select each image's GT target by minimum centroid L2 distance.

    For each image, picks j* = argmin_j ||pred_center - GT_j_center||_2 among
    valid GTs. Returns the (B, 4) target box.

    Motivation: max-IoU matching breaks down when the prediction overlaps no
    GT (IoU==0 for all j). In that regime argmax-IoU is undefined up to ties
    and silently falls back to the first valid GT. Centroid L2 matching gives
    a meaningful signal even when boxes don't overlap, making it a more
    stable matcher early in training or for hard (small/off-center) images.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        pred_c = pred[:, None, :2]             # (B, 1, 2)
        gt_c   = gt_boxes[..., :2]             # (B, M, 2)
        d2 = ((pred_c - gt_c) ** 2).sum(dim=-1)   # (B, M) squared L2
        # Mask padded rows so they lose the argmin
        d2 = torch.where(gt_mask, d2, torch.full_like(d2, float("inf")))
        j_star = d2.argmin(dim=-1)             # (B,)

    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)
    return target


def localization_mse_centroid_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss against the GT with closest centroid to the prediction.

    Matcher: min-L2-distance of (x_c, y_c) — see `_centroid_match_target`.
    Loss: F.mse_loss(pred, target).

    Use this when max-IoU matching is unstable (e.g., early training when
    predictions overlap no GT, so IoU matching falls back to first-GT).
    """
    target = _centroid_match_target(pred, gt_boxes, gt_mask)
    return F.mse_loss(pred, target)


def localization_l1_centroid_match_loss(
    pred: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 loss against the GT with closest centroid to the prediction.

    Matcher: min-L2-distance of (x_c, y_c) — see `_centroid_match_target`.
    Loss: F.l1_loss(pred, target).
    """
    target = _centroid_match_target(pred, gt_boxes, gt_mask)
    return F.l1_loss(pred, target)


def localization_l1_iou_match_loss(
    pred: torch.Tensor,       # (B, 4) in [0, 1]  — xywh
    gt_boxes: torch.Tensor,   # (B, M, 4)         — xywh, padded
    gt_mask: torch.Tensor,    # (B, M) bool       — valid GT rows
    eps: float = 1e-10,
) -> torch.Tensor:
    """L1 (mean absolute error) against the GT with highest IoU vs the prediction.

    Same matching rule as `localization_mse_iou_match_loss` (argmax-IoU over
    valid GTs, recomputed each batch, no gradient through argmax), but applies
    F.l1_loss instead of F.mse_loss on the residual. L1 is more robust to
    outliers and generally yields slightly tighter localization on small boxes
    since large coordinate errors aren't squared.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)
        pc1 = pred_exp[..., :2] - pred_exp[..., 2:] / 2
        pc2 = pred_exp[..., :2] + pred_exp[..., 2:] / 2
        gc1 = gt_boxes[..., :2] - gt_boxes[..., 2:] / 2
        gc2 = gt_boxes[..., :2] + gt_boxes[..., 2:] / 2
        i1 = torch.maximum(pc1, gc1)
        i2 = torch.minimum(pc2, gc2)
        inter_wh = (i2 - i1).clamp(min=0.0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]
        pa = pred_exp[..., 2] * pred_exp[..., 3]
        ga = gt_boxes[..., 2] * gt_boxes[..., 3]
        union = (pa + ga - inter).clamp(min=eps)
        iou = inter / union
        iou = torch.where(gt_mask, iou, torch.full_like(iou, -1.0))
        j_star = iou.argmax(dim=-1)

    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)

    return F.l1_loss(pred, target)


def localization_mse_iou_match_loss(
    pred: torch.Tensor,       # (B, 4) in [0, 1]  — xywh
    gt_boxes: torch.Tensor,   # (B, M, 4)         — xywh, padded
    gt_mask: torch.Tensor,    # (B, M) bool       — valid GT rows
    eps: float = 1e-10,
) -> torch.Tensor:
    """MSE against the GT with highest IoU vs the prediction.

    For each image, picks j* = argmax_j IoU(pred, GT_j) among valid GTs
    (masked by gt_mask), then applies F.mse_loss against GT_{j*}.

    Why: a deterministic regressor outputs one box per image. The val metric
    takes max-IoU over all GTs. Using max-IoU-match as the training target
    aligns the loss with the eval metric. Matching is recomputed every batch
    under `torch.no_grad()`, so no gradient flows through argmax — only
    through the L2 residual against the selected target.

    Args:
        pred:     (B, 4) xywh in [0, 1] (sigmoid output)
        gt_boxes: (B, M, 4) xywh in [0, 1], padded
        gt_mask:  (B, M) bool — True for valid GT slots

    Returns:
        scalar F.mse_loss(pred, target) where target is the max-IoU GT box.
    """
    B, M, _ = gt_boxes.shape

    with torch.no_grad():
        # Per-GT pairwise IoU (B, M) via xywh → corners
        pred_exp = pred.unsqueeze(1).expand(-1, M, -1)            # (B, M, 4)
        pc1 = pred_exp[..., :2] - pred_exp[..., 2:] / 2           # (B, M, 2)
        pc2 = pred_exp[..., :2] + pred_exp[..., 2:] / 2
        gc1 = gt_boxes[..., :2] - gt_boxes[..., 2:] / 2
        gc2 = gt_boxes[..., :2] + gt_boxes[..., 2:] / 2
        i1 = torch.maximum(pc1, gc1)
        i2 = torch.minimum(pc2, gc2)
        inter_wh = (i2 - i1).clamp(min=0.0)
        inter = inter_wh[..., 0] * inter_wh[..., 1]               # (B, M)
        pa = pred_exp[..., 2] * pred_exp[..., 3]
        ga = gt_boxes[..., 2] * gt_boxes[..., 3]
        union = (pa + ga - inter).clamp(min=eps)
        iou = inter / union                                        # (B, M)
        # Mask out padded rows → they lose the argmax
        iou = torch.where(gt_mask, iou, torch.full_like(iou, -1.0))
        j_star = iou.argmax(dim=-1)                                # (B,)

    # Gather per-image target (B, 4)
    target = gt_boxes.gather(
        1, j_star.view(B, 1, 1).expand(-1, 1, 4)
    ).squeeze(1)

    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# 4. Expected-IoU ordinal CE — exact supervised analog of GOS for IoU reward
# ---------------------------------------------------------------------------

def localization_exp_iou_ord_ce_loss(
    logits: dict[str, torch.Tensor],     # each (B, K)
    gt_boxes: torch.Tensor,              # (B, M, 4) xywh in [0, 1]
    gt_mask: torch.Tensor,               # (B, M) bool — valid GT rows
    K: int,
    n_thresholds: int = 100,
    eps: float = 1e-4,
    clamp_pred: bool = False,
) -> torch.Tensor:
    """Expected-IoU ordinal CE loss — exact infinite-N limit of GOS for IoU.

    L = -∫_0^1 log P_π(max_j IoU(b_hat, GT_j) > τ) dτ
      ≈ -Σ_{j=1}^{T-1} (1/T) log P_π(max-over-M IoU > j/T)   [left-Riemann]

    Derivation: for the reward r = max_j IoU, the score-function policy-gradient
    identity gives ∇_θ E[w(r)] = 0 (cumulative-hazard weight has policy-
    invariant expectation). Decomposing r as an integral of threshold
    indicators and using the binary-MaxRL Prop 3.3 per threshold, the GOS
    gradient at N→∞ equals -∇_θ L_ord-IoU.

    Implementation:
      1. Compute max-over-M-GT IoU tensor (B, K, K, K, K) via factored int_x,
         int_y per GT slot; elementwise-max across slots.
      2. Bucketize IoU into T = n_thresholds buckets.
      3. Materialize joint probability tensor (B, K, K, K, K) via outer product
         of the 4 softmax vectors.
      4. Scatter-add joint probs into per-bucket histogram.
      5. Cumulative sum → P(IoU > τ_j) = 1 - cum[j-1] for j=1..T-1.
      6. L = -(1/T) Σ log P(τ_j).

    Memory: O(B K^4) peak (IoU tensor + joint tensor + bucket indices).
    Compute: O(B K^4) per step. At B=128, K=50, this is ~800M ops.

    Args:
        logits:        dict keyed by HEAD_NAMES, each (B, K)
        gt_boxes:      (B, M, 4) normalized xywh coordinates
        gt_mask:       (B, M) bool — which rows are real GT (vs padding)
        K:             number of bins per coordinate
        n_thresholds:  T; number of grid points for the τ integral
        eps:           clamp for log(P) when P ≈ 0 and for zero-area union.
                       Must match ``GOS_SURVIVAL_EPS`` in ``advantages.py``
                       for the GOS → exp_iou_ord_ce identity (N → ∞) to hold.
        clamp_pred:    if True, clamp predicted box corners to [0, 1] before
                       computing intersection/union, matching the RL path's
                       clamp_boxes_to_image. The predicted area becomes 4-D
                       over (xc, yc, w, h) because the clamped width depends
                       on xc (and clamped height on yc). Default False
                       (baseline behavior — bit-identical to the pre-flag version).

    Returns:
        scalar loss, positive value, averaged over batch.
    """
    device = logits['x_c'].device
    dtype = logits['x_c'].dtype
    B = logits['x_c'].size(0)
    M = gt_boxes.size(1)
    T = n_thresholds

    # 1. Softmax per head
    p_xc = F.softmax(logits['x_c'], dim=-1)   # (B, K)
    p_yc = F.softmax(logits['y_c'], dim=-1)
    p_w  = F.softmax(logits['w'],   dim=-1)
    p_h  = F.softmax(logits['h'],   dim=-1)

    # 2. Bin centers
    bc = (torch.arange(K, device=device, dtype=dtype) + 0.5) / K   # (K,)

    # 3. Predicted corner intervals per (k1, k3) / (k2, k4)
    px1 = bc.view(K, 1) - bc.view(1, K) / 2    # (K, K)
    px2 = bc.view(K, 1) + bc.view(1, K) / 2
    py1 = bc.view(K, 1) - bc.view(1, K) / 2
    py2 = bc.view(K, 1) + bc.view(1, K) / 2

    if clamp_pred:
        # Mirror clamp_boxes_to_image in the RL path: corners pinned to [0, 1].
        # After clamping, the predicted width depends on (xc, w) jointly, so
        # pred_area becomes 4-D over (xc, yc, w, h).
        px1 = px1.clamp(0.0, 1.0)
        px2 = px2.clamp(0.0, 1.0)
        py1 = py1.clamp(0.0, 1.0)
        py2 = py2.clamp(0.0, 1.0)
        pred_w = (px2 - px1).clamp(min=0.0)                              # (K_xc, K_w)
        pred_h = (py2 - py1).clamp(min=0.0)                              # (K_yc, K_h)
        # axis order matches iou_m: (xc, yc, w, h)
        pred_area = pred_w[:, None, :, None] * pred_h[None, :, None, :]  # (K_xc, K_yc, K_w, K_h)
    else:
        # Predicted area (K, K) over (k3, k4) = w × h (full, unclamped)
        pred_area = bc[:, None] * bc[None, :]                            # (K, K)

    # 4. Max-over-M-GT IoU tensor (B, K, K, K, K)
    iou_max = torch.full(
        (B, K, K, K, K), -1.0, device=device, dtype=dtype
    )
    for m in range(M):
        gt_m    = gt_boxes[:, m]                 # (B, 4)
        valid   = gt_mask[:, m]                  # (B,) bool
        gt_xc, gt_yc, gt_w, gt_h = gt_m.unbind(-1)
        gt_x1 = gt_xc - gt_w / 2
        gt_x2 = gt_xc + gt_w / 2
        gt_y1 = gt_yc - gt_h / 2
        gt_y2 = gt_yc + gt_h / 2
        gt_area = (gt_w * gt_h).clamp(min=eps)

        # int_x: (B, K, K) [k1, k3]
        int_x_l = torch.maximum(px1[None], gt_x1[:, None, None])
        int_x_r = torch.minimum(px2[None], gt_x2[:, None, None])
        int_x = (int_x_r - int_x_l).clamp(min=0.0)

        # int_y: (B, K, K) [k2, k4]
        int_y_t = torch.maximum(py1[None], gt_y1[:, None, None])
        int_y_b = torch.minimum(py2[None], gt_y2[:, None, None])
        int_y = (int_y_b - int_y_t).clamp(min=0.0)

        # IoU (B, K, K, K, K) = inter / (pred_area + gt_area - inter)
        inter = int_x[:, :, None, :, None] * int_y[:, None, :, None, :]
        if clamp_pred:
            # pred_area shape (K_xc, K_yc, K_w, K_h) → broadcast as (1, ...)
            union = (
                pred_area[None]
                + gt_area[:, None, None, None, None]
                - inter
            )
        else:
            # pred_area shape (K_w, K_h) → broadcast as (1, 1, 1, K_w, K_h)
            union = (
                pred_area[None, None, None, :, :]
                + gt_area[:, None, None, None, None]
                - inter
            )
        iou_m = inter / union.clamp(min=eps)

        # Zero-out (actually set to -1) for invalid GT rows
        iou_m = torch.where(
            valid[:, None, None, None, None],
            iou_m,
            torch.full_like(iou_m, -1.0),
        )

        iou_max = torch.maximum(iou_max, iou_m)

        # Explicit free of per-slot tensors
        del int_x, int_y, inter, union, iou_m

    # If a whole image had zero valid GTs, iou_max stays at -1; clip to 0.
    iou_max = iou_max.clamp(min=0.0, max=1.0)
    iou_max = iou_max.detach()   # no gradient flows through IoU values

    # 5. Bucketize into T buckets
    bucket_idx = (iou_max * T).clamp(max=T - 1).long()   # (B, K, K, K, K)

    # 6. Joint probability (B, K, K, K, K) via outer product
    joint = (
        p_xc[:, :, None, None, None] *
        p_yc[:, None, :, None, None] *
        p_w [:, None, None, :, None] *
        p_h [:, None, None, None, :]
    )

    # 7. Scatter-add joint into per-bucket histogram
    joint_flat = joint.reshape(B, -1)                    # (B, K^4)
    bucket_flat = bucket_idx.reshape(B, -1)              # (B, K^4)
    hist = torch.zeros(B, T, device=device, dtype=dtype)
    hist.scatter_add_(1, bucket_flat, joint_flat)        # (B, T)

    # 8. Cumulative sum — cum[b, j] = P(IoU <= j/T)
    cum = hist.cumsum(dim=-1)                            # (B, T)

    # 9. P(τ_j) = P(IoU > τ_j) for τ_j = j/T, j=1..T-1:
    #    P(τ_j) = 1 - cum[j-1]
    # For j=0, P = 1 so log(P)=0 (no contribution).
    P_tau = (1.0 - cum[:, :-1]).clamp(min=eps)           # (B, T-1)

    # 10. Left-Riemann sum with dtau = 1/T
    loss = -(1.0 / T) * torch.log(P_tau).sum(dim=-1).mean()

    return loss
