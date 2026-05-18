"""One-shot cosine-sim test: GOS vs exp_iou_ord_ce on a single fresh-init batch.

Usage (from repo root, on a GPU node):
    python -m experiment9_imagenet_localization.scripts.cosine_quick \
        --n_images 8 --Gs 1024,16384,262144

No checkpoint needed — builds a fresh LocalizationPolicy with the given seed.
The asymptotic identity claim is model-agnostic, so a random-init policy is
sufficient for "does GOS gradient at large N point along the supervised
gradient?" diagnostics.

Outputs:
    For each G value, print cosine similarity between GOS's mean-gradient and
    each of (exp_iou_ord_ce, exp_iou_ord_ce_clamped). Magnitude ratio also.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path("/zfsauton/scratch/shrinivr/cont-maxrl")
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from experiment9_imagenet_localization.advantages import ADVANTAGE_FNS  # noqa: E402
from experiment9_imagenet_localization.gradient_analysis import (  # noqa: E402
    _accumulate_ref_gradient,
    _accumulate_rl_gradient,
    _load_val_subset,
)
from experiment9_imagenet_localization.model import LocalizationPolicy  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_images", type=int, default=8,
                   help="Images in the single batch")
    p.add_argument("--Gs", default="1024,16384,262144",
                   help="Comma-separated rollout sizes")
    p.add_argument("--data_dir", default="/zfsauton/datasets/imagenet")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    G_values = [int(g) for g in args.Gs.split(",") if g.strip()]
    print(f"G values: {G_values}")

    # Fresh-init policy (no checkpoint).
    model = LocalizationPolicy(K=args.K, pretrained=False, seed=args.seed).to(device)
    model.eval()

    print(f"Loading {args.n_images} val images...")
    t0 = time.time()
    images, gt_boxes, gt_mask, target_bins = _load_val_subset(
        args.data_dir, args.K, args.seed, args.n_images,
        batch_size=args.n_images, num_workers=args.num_workers,
    )
    print(f"  loaded in {time.time() - t0:.1f}s — images shape {tuple(images.shape)}")
    images = images.to(device)
    gt_boxes = gt_boxes.to(device)
    gt_mask = gt_mask.to(device)
    target_bins = {h: v.to(device) for h, v in target_bins.items()}

    # --- Reference gradients (once each) ---
    refs: dict[str, torch.Tensor] = {}
    for ref_method in ("exp_iou_ord_ce", "exp_iou_ord_ce_clamped"):
        t0 = time.time()
        refs[ref_method] = _accumulate_ref_gradient(
            model, images, gt_boxes, gt_mask, target_bins, args.K, ref_method,
        )
        print(f"  ref={ref_method}  norm={refs[ref_method].norm().item():.4e}  "
              f"({time.time() - t0:.1f}s)")

    gos_fn = ADVANTAGE_FNS["gos"]

    # --- RL gradients at each G, cosine vs each ref ---
    print()
    print(f"{'G':>10} | "
          f"{'cos vs exp_iou_ord_ce':>22} | "
          f"{'cos vs exp_iou_ord_ce_clamped':>30} | "
          f"{'GOS grad norm':>14} | "
          f"{'mag ratio (clamped)':>20} | "
          f"{'time (s)':>8}")
    print("-" * 130)
    torch.manual_seed(args.seed)
    for G in G_values:
        t0 = time.time()
        rl_grad = _accumulate_rl_gradient(
            model, images, gt_boxes, gt_mask, args.K, G, gos_fn,
        )
        dt = time.time() - t0
        rl_norm = rl_grad.norm().item()

        cos_unclamped = torch.nn.functional.cosine_similarity(
            refs["exp_iou_ord_ce"].unsqueeze(0), rl_grad.unsqueeze(0),
        ).item()
        cos_clamped = torch.nn.functional.cosine_similarity(
            refs["exp_iou_ord_ce_clamped"].unsqueeze(0), rl_grad.unsqueeze(0),
        ).item()
        mag_ratio_clamped = rl_norm / max(refs["exp_iou_ord_ce_clamped"].norm().item(), 1e-12)
        print(f"{G:>10d} | "
              f"{cos_unclamped:>22.4f} | "
              f"{cos_clamped:>30.4f} | "
              f"{rl_norm:>14.4e} | "
              f"{mag_ratio_clamped:>20.4f} | "
              f"{dt:>8.1f}")


if __name__ == "__main__":
    main()
