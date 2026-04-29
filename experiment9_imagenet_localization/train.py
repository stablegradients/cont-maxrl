from __future__ import annotations

from typing import Optional, Callable

import os
import torch
import torch.nn.functional as F

from experiment9_imagenet_localization.advantages import ADVANTAGE_FNS
from experiment9_imagenet_localization.iou import batched_max_iou, clamp_boxes_to_image
from experiment9_imagenet_localization.losses import (
    factored_sample_log_prob,
    localization_exp_iou_ord_ce_loss,
    localization_ordinal_ce_loss,
    localization_plain_ce_loss,
    localization_mse_loss,
    localization_mse_iou_match_loss,
    localization_mse_centroid_match_loss,
    localization_l1_iou_match_loss,
    localization_l1_centroid_match_loss,
)
from experiment9_imagenet_localization.model import LocalizationPolicy  # type hint

HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_boxes_from_policy(
    logits: dict[str, torch.Tensor],    # each (B, K)
    N: int,
    K: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Sample N boxes per image via multinomial over each head.

    Returns:
        samples: dict keyed by HEAD_NAMES, each (B, N) long.
        sampled_boxes: (B, N, 4) float, xywh in [0, 1], already clamp_boxes_to_image-ed.
    """
    log_probs = {h: F.log_softmax(logits[h], dim=-1) for h in HEAD_NAMES}
    probs     = {h: log_probs[h].exp()               for h in HEAD_NAMES}

    samples: dict[str, torch.Tensor] = {}
    for h in HEAD_NAMES:
        # probs[h]: (B, K); torch.multinomial on (B, K) with N samples -> (B, N)
        samples[h] = torch.multinomial(probs[h], N, replacement=True)

    # Convert bin indices to box coordinates (bin centers).
    coords = []
    for h in HEAD_NAMES:
        coords.append((samples[h].float() + 0.5) / K)   # (B, N)
    sampled_boxes = torch.stack(coords, dim=-1)         # (B, N, 4)
    sampled_boxes = clamp_boxes_to_image(sampled_boxes) # safe-guard (already in [0,1])

    return samples, sampled_boxes


# ---------------------------------------------------------------------------
# One RL training step (per-batch) — exposed so tests can call it directly.
# ---------------------------------------------------------------------------

def rl_training_step(
    model: LocalizationPolicy,
    batch: dict,
    method: str,
    N: int,
    K: int,
    device: torch.device,
) -> dict:
    """Run a single RL training step (no optimizer.step — test helper).

    Args:
        model: the policy. Must be on `device` already.
        batch: dict from the collate fn, keys 'images', 'gt_boxes', 'gt_mask'.
        method: one of the ADVANTAGE_FNS keys.
        N: number of rollouts per image.
        K: number of bins per coordinate.
        device: torch device.

    Returns:
        dict with keys:
        - 'loss': scalar tensor (with grad).
        - 'rewards': (B, N) float tensor (no grad).
        - 'advantages': (B, N) float tensor (no grad).
        - 'sample_log_probs': (B, N) float tensor (has grad).
    """
    advantage_fn = ADVANTAGE_FNS[method]

    images   = batch['images'].to(device)               # (B, 3, 224, 224)
    gt_boxes = batch['gt_boxes'].to(device)             # (B, MAX_M, 4)
    gt_mask  = batch['gt_mask'].to(device)              # (B, MAX_M) bool

    # Forward pass: optionally bf16 autocast (toggle via BF16=1 env var).
    # Default off because earlier comparisons showed ~5% train reward drop
    # at end of epoch 1 vs fp32 — likely from the gradient on logits being
    # downcast to bf16 at the autocast boundary.
    use_bf16 = bool(int(os.environ.get("BF16", "0"))) and device.type == 'cuda'
    if use_bf16:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(images)
        log_probs = {h: F.log_softmax(logits[h].float(), dim=-1) for h in HEAD_NAMES}
    else:
        logits = model(images)                           # dict of (B, K)
        log_probs = {h: F.log_softmax(logits[h], dim=-1) for h in HEAD_NAMES}

    # Stack the 4 head probabilities into one (4·B, K) tensor so we can
    # do a single multinomial / single gather instead of 4 separate kernel
    # launches per step. Saves a few % at small N, more at large N where
    # kernel-launch overhead matters less than the memory traffic.
    B = log_probs['x_c'].size(0)
    log_probs_stacked = torch.stack([log_probs[h] for h in HEAD_NAMES], dim=0)  # (4, B, K)

    with torch.no_grad():
        probs_flat = log_probs_stacked.exp().reshape(4 * B, K)              # (4·B, K)
        # One fused multinomial draws N samples for every (head, image).
        samples_flat = torch.multinomial(probs_flat, N, replacement=True)   # (4·B, N)
        samples_stacked = samples_flat.view(4, B, N)                         # (4, B, N)

        # coords: (B, N, 4) — note we permute (4, B, N) → (B, N, 4)
        coords = ((samples_stacked.float() + 0.5) / K).permute(1, 2, 0)
        sampled_boxes = clamp_boxes_to_image(coords)
        rewards = batched_max_iou(sampled_boxes, gt_boxes, gt_mask)         # (B, N)

        # advantage_fn operates on the last dim — (B, N) in → (B, N) out.
        advantages = advantage_fn(rewards)                                  # (B, N)

    # Fused gather across heads: per_head_lp[h] = log_probs[h].gather(1, samples[h])
    # → stacked: (4, B, N) gathered from (4, B, K) at indices (4, B, N)
    sample_log_probs = log_probs_stacked.gather(2, samples_stacked).sum(dim=0)  # (B, N)

    # PG loss: -mean_b mean_n (advantages[b, n] * log_prob[b, n])
    per_image = -(advantages.detach() * sample_log_probs).mean(dim=1)  # (B,)
    loss = per_image.mean()

    return {
        'loss': loss,
        'rewards': rewards.detach(),
        'advantages': advantages.detach(),
        'sample_log_probs': sample_log_probs.detach(),
    }


# ---------------------------------------------------------------------------
# Epoch loops — the shapes used by run.py
# ---------------------------------------------------------------------------

def train_rl_epoch_localization(
    model: LocalizationPolicy,
    dataloader,
    optimizer,
    scheduler,
    method: str,
    N: int,
    K: int,
    device: torch.device,
    epoch: int,
    grad_clip: float = 10.0,
    wandb_run=None,
    log_every: int = 50,
) -> dict:
    """Run one RL training epoch.

    Returns a dict of running metrics (aggregated over the epoch):
      - loss_mean, reward_mean, reward_std, reward_at_0_5, reward_at_0_75,
        frac_zero_reward, advantage_abs_mean, grad_norm_mean.
    Optionally logs per-step metrics to `wandb_run`.
    """
    model.train()
    metrics = _running_metrics_init()
    step = 0
    for batch in dataloader:
        optimizer.zero_grad()
        out = rl_training_step(model, batch, method, N, K, device)
        loss = out['loss']
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip
        ).item()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _running_metrics_update(metrics, loss.item(), out, grad_norm)

        if wandb_run is not None and step % log_every == 0:
            r_step = out['rewards']
            wandb_run.log({
                'train/loss': loss.item(),
                'train/grad_norm_global': grad_norm,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train_rl/reward_mean': r_step.mean().item(),
                'train_rl/reward_std':  r_step.std().item(),
                'train_rl/advantage_abs_mean': out['advantages'].abs().mean().item(),
                'train_rl/frac_iou_band_02_08': (
                    ((r_step > 0.2) & (r_step < 0.8)).float().mean().item()
                ),
                'epoch': epoch,
                'step': step,
            })
        step += 1

    return _running_metrics_finalize(metrics)


def train_supervised_epoch_localization(
    model,  # LocalizationPolicy (for ordinal/plain CE) OR LocalizationRegressor (for MSE)
    dataloader,
    optimizer,
    scheduler,
    method: str,   # "ordinal_ce" | "plain_ce" | "mse"
    K: int,
    device: torch.device,
    epoch: int,
    grad_clip: float = 10.0,
    wandb_run=None,
    log_every: int = 50,
) -> dict:
    """Run one supervised training epoch for the chosen baseline method.

    Methods:
    - "ordinal_ce": uses LocalizationPolicy (4 heads, K bins each) +
       localization_ordinal_ce_loss.
    - "plain_ce": LocalizationPolicy + localization_plain_ce_loss.
    - "mse":       LocalizationRegressor + localization_mse_loss.
    """
    model.train()
    losses = []
    step = 0
    for batch in dataloader:
        images = batch['images'].to(device)
        optimizer.zero_grad()

        if method == "mse":
            # Legacy primary-GT MSE (kept for reproducibility; biased on multi-GT).
            pred = model(images)
            gt_primary = batch['gt_boxes'][:, 0, :].to(device)
            loss = localization_mse_loss(pred, gt_primary)
        elif method == "mse_iou_match":
            # MSE against the GT with highest IoU vs the prediction.
            pred = model(images)                                # (B, 4)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_mse_iou_match_loss(pred, gt_boxes, gt_mask)
        elif method == "l1_iou_match":
            # L1 against the GT with highest IoU vs the prediction.
            pred = model(images)                                # (B, 4)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_l1_iou_match_loss(pred, gt_boxes, gt_mask)
        elif method == "mse_centroid_match":
            pred = model(images)
            gt_boxes = batch['gt_boxes'].to(device)
            gt_mask  = batch['gt_mask'].to(device)
            loss = localization_mse_centroid_match_loss(pred, gt_boxes, gt_mask)
        elif method == "l1_centroid_match":
            pred = model(images)
            gt_boxes = batch['gt_boxes'].to(device)
            gt_mask  = batch['gt_mask'].to(device)
            loss = localization_l1_centroid_match_loss(pred, gt_boxes, gt_mask)
        elif method == "exp_iou_ord_ce":
            logits = model(images)                             # dict of (B, K)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_exp_iou_ord_ce_loss(
                logits, gt_boxes, gt_mask, K,
                n_thresholds=100,
            )
        else:
            logits = model(images)                             # dict of (B, K)
            targets = {h: batch['target_bins'][h].to(device) for h in HEAD_NAMES}
            if method == "ordinal_ce":
                loss = localization_ordinal_ce_loss(logits, targets, K)
            elif method == "plain_ce":
                loss = localization_plain_ce_loss(logits, targets)
            else:
                raise ValueError(f"unknown supervised method: {method}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=grad_clip
        ).item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        if wandb_run is not None and step % log_every == 0:
            wandb_run.log({
                'train/loss': loss.item(),
                'train/grad_norm_global': grad_norm,
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'step': step,
            })
        step += 1

    import statistics
    return {'loss_mean': statistics.fmean(losses) if losses else 0.0}


# ---------------------------------------------------------------------------
# Running-metrics helpers (kept here rather than in their own file)
# ---------------------------------------------------------------------------

def _running_metrics_init() -> dict:
    return {
        'losses': [], 'rewards': [], 'advantages': [], 'grad_norms': [],
    }

def _running_metrics_update(m: dict, loss: float, out: dict, grad_norm: float) -> None:
    m['losses'].append(loss)
    m['rewards'].append(out['rewards'].flatten().cpu())
    m['advantages'].append(out['advantages'].flatten().cpu())
    m['grad_norms'].append(grad_norm)

def _running_metrics_finalize(m: dict) -> dict:
    if not m['losses']:
        return {'loss_mean': 0.0}
    r = torch.cat(m['rewards'])
    a = torch.cat(m['advantages'])
    import statistics
    return {
        'loss_mean':         statistics.fmean(m['losses']),
        'reward_mean':       r.mean().item(),
        'reward_std':        r.std().item(),
        'reward_at_0_5':     (r > 0.5).float().mean().item(),
        'reward_at_0_75':    (r > 0.75).float().mean().item(),
        'frac_zero_reward':  (r == 0).float().mean().item(),
        # Day-2 go/no-go gate (spec §9.1): fraction of rollouts with IoU strictly
        # in (0.2, 0.8). Must be >= 0.10 after epoch 1 for GOS vs binary_maxrl to
        # be separable; otherwise the reward distribution is effectively binary.
        'frac_iou_band_02_08': ((r > 0.2) & (r < 0.8)).float().mean().item(),
        'advantage_abs_mean':a.abs().mean().item(),
        'grad_norm_mean':    statistics.fmean(m['grad_norms']),
    }
