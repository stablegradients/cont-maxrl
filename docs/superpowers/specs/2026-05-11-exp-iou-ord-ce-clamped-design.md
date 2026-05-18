# `exp_iou_ord_ce_clamped` — design

**Date:** 2026-05-11
**Scope:** `experiment9_imagenet_localization`

## Motivation

The current `exp_iou_ord_ce` supervised loss scores predicted-box corners
that can extend outside `[0, 1]` (e.g. for `xc=0.01, w=0.99`, `px1=-0.485`
and `px2=0.505`). This is asymmetric with the RL training path
(`train.py:128`), which applies `clamp_boxes_to_image` to sampled boxes
before computing IoU. The mismatch breaks the "exp_iou_ord_ce is the
N→∞ limit of GOS" comparison at finite K, because the two are scoring
different geometric objects (unclipped vs clipped boxes).

This spec adds a clamped variant of the supervised loss that mirrors the
RL clamping, so that supervised vs RL is a controlled comparison.

## Change set

### 1. `experiment9_imagenet_localization/losses.py`

Add `clamp_pred: bool = False` parameter to
`localization_exp_iou_ord_ce_loss`. When `True`:

- Clamp each of `px1, px2, py1, py2` to `[0, 1]`.
- Recompute predicted width and height from clamped corners:
  - `pred_w = (px2 - px1).clamp(min=0.0)`, shape `(K_xc, K_w)`.
  - `pred_h = (py2 - py1).clamp(min=0.0)`, shape `(K_yc, K_h)`.
- Build a 4-D `pred_area_4d` of shape `(K_xc, K_yc, K_w, K_h)` via
  outer product and broadcast it into the union computation as
  `pred_area_4d[None]` (shape `(1, K_xc, K_yc, K_w, K_h)`), matching
  the `(B, xc, yc, w, h)` axis order of `iou_m`.
- Intersection computation already uses `(int_x_r − int_x_l).clamp(min=0)`,
  which works correctly with clamped corners (no change).

When `clamp_pred=False`: bit-identical to the current loss (no
behavioral change to the baseline).

**Memory:** at K=50, the new 4-D `pred_area` is `50⁴ × 4 B = 25 MB`,
negligible against the existing 5-D union tensor (~3 GB at batch=128).

**Numerics:** `eps=1e-4` and `n_thresholds=100` unchanged; the
clamp is exact, no new epsilons introduced.

### 2. `experiment9_imagenet_localization/run.py`

- Add `"exp_iou_ord_ce_clamped"` to `SUPERVISED_METHODS` at line ~50.
- Model selection at lines 508–515 unchanged: the new method is **not**
  in `is_mse`, so it falls through to `LocalizationPolicy` as required.

### 3. `experiment9_imagenet_localization/train.py`

In `train_supervised_epoch_localization` (line 269), extend the existing
`elif method == "exp_iou_ord_ce":` branch to also accept
`"exp_iou_ord_ce_clamped"`, passing
`clamp_pred=(method == "exp_iou_ord_ce_clamped")` to the loss.

### 4. New sweep scripts

Three SLURM files, templated off
`sweep_scripts/exp_iou_ord_ce_K50_N64_seed43.sh`:

```
sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed42.sh
sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed43.sh
sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed44.sh
```

Each differs from the baseline only in:

| Field | Value |
|---|---|
| `#SBATCH --job-name` | `j_clamped_seed{seed}` |
| `WANDB_RUN_GROUP` | `exp_iou_ord_ce_clamped` |
| `--method` | `exp_iou_ord_ce_clamped` |
| `--seed` | `{42,43,44}` |
| `--output_dir` | `results/exp_iou_ord_ce_clamped_K50_N64_seed{seed}` |
| log path | `sweep_scripts/logs/j_clamped_seed{seed}_%j.out` |

All other knobs (Adam, `--lr 5e-4`, `--K 50`, `--N 64`,
`--epochs 30`, `--batch_size 128`, 1× a6000, 32 GB, 48 h,
`num_workers=8`, `--wandb`) match the baseline exactly.

### 5. Submission

Three `sbatch` calls from the repo root. Capture job IDs, run
`squeue -u $USER` to confirm queueing.

## Non-goals

- Not changing the baseline `exp_iou_ord_ce` runs.
- Not changing the RL path or `clamp_boxes_to_image`.
- Not adding new seeds beyond the requested 3.
- Not running the SGD optimizer variant (Adam only this round).

## Verification

Pre-submit local check (5 min, runs on a single GPU node):

```
python -c "
import torch
from experiment9_imagenet_localization.losses import localization_exp_iou_ord_ce_loss
torch.manual_seed(0)
B, K = 4, 50
logits = {h: torch.randn(B, K, requires_grad=True) for h in ('x_c','y_c','w','h')}
gt_boxes = torch.rand(B, 1, 4)
gt_mask = torch.ones(B, 1, dtype=torch.bool)
loss_a = localization_exp_iou_ord_ce_loss(logits, gt_boxes, gt_mask, K, clamp_pred=False)
loss_b = localization_exp_iou_ord_ce_loss(logits, gt_boxes, gt_mask, K, clamp_pred=True)
print(loss_a.item(), loss_b.item())  # finite, different
loss_b.backward()                     # gradients flow
"
```

Then post-submit: monitor wandb for the new group; verify the first
epoch's train loss is finite and that train/val IoU curves rise.

## Rollback

Trivial: remove the new method string from `run.py`, revert the
`losses.py` arg, delete the three scripts. The baseline
`exp_iou_ord_ce` runs are untouched, so no data loss.
