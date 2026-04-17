"""Gradient cosine analysis: RL methods vs supervised ordinal CE.

For each checkpoint, each method, each K, each G ∈ {4, 16, 64, 256, 1024, 4096}:
1. Load a fixed set of val images (100 by default).
2. Per image: compute supervised ordinal-CE gradient (exact) + RL gradient (G rollouts).
3. Flatten gradients on the 4 classification heads only (exclude backbone — confounds).
4. Per-image cosine similarity; then mean across images.

Output: JSON file with {method: {G: mean_cosine}}.

CLI:
  python -m experiment9_imagenet_localization.gradient_analysis \
      --checkpoint PATH \
      --data_dir /zfsauton/datasets/imagenet \
      --K 50 \
      --methods gos,binary_maxrl,grpo,rloo,reinforce,dg \
      --Gs 4,16,64,256,1024,4096 \
      --n_images 100 \
      --output_json PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment9_imagenet_localization.advantages import ADVANTAGE_FNS
from experiment9_imagenet_localization.data import (
    ImageNetLocDataset, build_collate_fn,
)
from experiment9_imagenet_localization.iou import batched_max_iou, clamp_boxes_to_image
from experiment9_imagenet_localization.losses import (
    factored_sample_log_prob,
    localization_ordinal_ce_loss,
)
from experiment9_imagenet_localization.model import LocalizationPolicy

HEAD_NAMES = ("x_c", "y_c", "w", "h")


def head_params(model: LocalizationPolicy) -> list[torch.nn.Parameter]:
    """Return the list of trainable parameters in the 4 classification heads."""
    return [p for p in model.heads.parameters() if p.requires_grad]


def flatten_head_grads(params) -> torch.Tensor:
    """Flatten `p.grad` for all passed params into a single 1D tensor. Each must have .grad set."""
    return torch.cat([p.grad.detach().flatten() for p in params])


def compute_supervised_head_gradient(
    model: LocalizationPolicy,
    image: torch.Tensor,        # (1, 3, 224, 224)
    target_bins: dict[str, torch.Tensor],  # each (1,) long
    K: int,
) -> torch.Tensor:
    """Compute ∇_heads L_ordinal_ce(logits, target_bins) for a single image. Returns flat head grads."""
    model.zero_grad()
    # Do not freeze backbone — we just don't read backbone grads. Forward must be fresh.
    logits = model(image)                                       # dict of (1, K)
    loss = localization_ordinal_ce_loss(logits, target_bins, K)
    loss.backward()
    return flatten_head_grads(head_params(model))


def compute_rl_head_gradient(
    model: LocalizationPolicy,
    image: torch.Tensor,
    gt_boxes: torch.Tensor,     # (1, MAX_M, 4)
    gt_mask: torch.Tensor,      # (1, MAX_M)
    K: int,
    G: int,
    advantage_fn,
) -> torch.Tensor:
    """Compute ∇_heads L_pg for a single image with G rollouts."""
    model.zero_grad()
    logits = model(image)
    log_probs = {h: F.log_softmax(logits[h], -1) for h in HEAD_NAMES}

    with torch.no_grad():
        probs = {h: log_probs[h].exp() for h in HEAD_NAMES}
        samples = {h: torch.multinomial(probs[h], G, replacement=True) for h in HEAD_NAMES}
        coords = torch.stack(
            [(samples[h].float() + 0.5) / K for h in HEAD_NAMES], dim=-1,
        )  # (1, G, 4)
        sampled_boxes = clamp_boxes_to_image(coords)
        rewards = batched_max_iou(sampled_boxes, gt_boxes, gt_mask)  # (1, G)
        advantages = advantage_fn(rewards[0]).unsqueeze(0)            # (1, G)

    sample_log_probs = factored_sample_log_prob(log_probs, samples)   # (1, G)
    loss = -(advantages.detach() * sample_log_probs).mean()
    loss.backward()
    return flatten_head_grads(head_params(model))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scalar cosine similarity between two 1D tensors. Returns 0.0 if either is zero-norm."""
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return (a @ b / (norm_a * norm_b)).item()


def run_analysis(
    model: LocalizationPolicy,
    images: torch.Tensor,        # (n_images, 3, 224, 224) on device
    gt_boxes: torch.Tensor,      # (n_images, MAX_M, 4)
    gt_mask: torch.Tensor,       # (n_images, MAX_M)
    target_bins: dict[str, torch.Tensor],   # each (n_images,) long
    K: int,
    methods: list[str],
    Gs: list[int],
) -> dict[str, dict[int, float]]:
    """Return dict[method][G] = mean cosine similarity over images."""
    results: dict[str, dict[int, float]] = {m: {} for m in methods}
    n_images = images.shape[0]

    # For each image: compute supervised gradient once (reused across methods / Gs).
    for method in methods:
        advantage_fn = ADVANTAGE_FNS[method]
        for G in Gs:
            cosines = []
            for i in range(n_images):
                im = images[i:i+1]
                gb = gt_boxes[i:i+1]
                gm = gt_mask[i:i+1]
                tb = {h: target_bins[h][i:i+1] for h in HEAD_NAMES}

                g_sup = compute_supervised_head_gradient(model, im, tb, K)
                g_rl  = compute_rl_head_gradient(model, im, gb, gm, K, G, advantage_fn)
                cosines.append(cosine(g_sup, g_rl))

            mean_cos = sum(cosines) / max(1, len(cosines))
            results[method][G] = float(mean_cos)
            print(f"  method={method:>12s}  G={G:5d}  mean_cos={mean_cos:.4f}")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Path to model checkpoint (.pt). If omitted, use a fresh model with pretrained=False.")
    p.add_argument("--data_dir", default="/zfsauton/datasets/imagenet")
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--methods", default="gos,binary_maxrl,grpo,rloo,reinforce,dg",
                   help="Comma-separated method names")
    p.add_argument("--Gs", default="4,16,64,256,1024,4096",
                   help="Comma-separated rollout counts")
    p.add_argument("--n_images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Images loaded per batch (for DataLoader)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_json", default=None, required=False,
                   help="Where to write the JSON result. Default: print to stdout.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Gs = [int(g) for g in args.Gs.split(",") if g.strip()]

    # --- Load images ---
    val_dataset = ImageNetLocDataset(
        root_dir=args.data_dir, split="val", K=args.K,
        subsample=args.n_images, seed=args.seed, train_aug=False,
    )
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=build_collate_fn())
    images_list, gt_boxes_list, gt_mask_list = [], [], []
    target_bins_list = {h: [] for h in HEAD_NAMES}
    for batch in loader:
        images_list.append(batch["images"])
        gt_boxes_list.append(batch["gt_boxes"])
        gt_mask_list.append(batch["gt_mask"])
        for h in HEAD_NAMES:
            target_bins_list[h].append(batch["target_bins"][h])
    images      = torch.cat(images_list).to(device)
    gt_boxes    = torch.cat(gt_boxes_list).to(device)
    gt_mask     = torch.cat(gt_mask_list).to(device)
    target_bins = {h: torch.cat(target_bins_list[h]).to(device) for h in HEAD_NAMES}

    # --- Load model ---
    model = LocalizationPolicy(K=args.K, pretrained=(args.checkpoint is None), seed=args.seed).to(device)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)

    results = run_analysis(model, images, gt_boxes, gt_mask, target_bins,
                           args.K, methods, Gs)

    # --- Write JSON ---
    output = {
        "config": {
            "K": args.K, "methods": methods, "Gs": Gs,
            "n_images": images.shape[0], "seed": args.seed,
            "checkpoint": args.checkpoint,
        },
        "results": results,
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {args.output_json}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
