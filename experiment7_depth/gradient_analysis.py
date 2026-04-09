"""Gradient cosine alignment analysis.

Computes cosine similarity between RL method gradients and supervised ordinal CE
gradient at saved checkpoints. Produces data for Figure 7.

Usage:
    python -m experiment7_depth.gradient_analysis \
        --results_dir ./experiment7_depth/results \
        --manifest ./experiment7_depth/scripts/runs/manifest.csv \
        --data_dir ./data/nyu \
        --output_dir ./experiment7_depth/figures \
        --K_values 10 100 1000 \
        --epochs 5 15 25
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import create_model
from .losses import dense_ordinal_ce_loss
from .advantages import ADVANTAGE_FNS
from .data import NYUDepthDataset


RL_METHODS = ["cont_maxrl", "grpo", "reinforce", "binary_maxrl"]
SUPERVISED_REF = "ordinal_ce"
G_FOR_ANALYSIS = 64


def get_trainable_grad_vector(model):
    """Flatten gradients of all trainable params into one vector."""
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return torch.zeros(1)
    return torch.cat(grads)


def compute_cosine_similarity(grad_a, grad_b):
    """Cosine similarity between two gradient vectors."""
    if grad_a.numel() == 0 or grad_b.numel() == 0:
        return 0.0
    cos = F.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0)).item()
    return cos


def compute_supervised_gradient(model, batch, K, device):
    """Compute gradient from supervised ordinal CE loss. Returns flattened grad vector."""
    model.train()
    model.zero_grad()

    images = batch["image"].to(device)
    target_bin = batch["target_bin"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(images)
        loss = dense_ordinal_ce_loss(logits, target_bin, valid_mask, K)

    loss.backward()
    return get_trainable_grad_vector(model)


def compute_rl_gradient(model, batch, method, K, G, device):
    """Compute gradient from RL policy gradient loss. Returns flattened grad vector."""
    model.train()
    model.zero_grad()

    advantage_fn = ADVANTAGE_FNS[method]

    images = batch["image"].to(device)
    target_bin = batch["target_bin"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(images)
        B, _, H, W = logits.shape

        logits_bhwk = logits.permute(0, 2, 3, 1).contiguous()
        log_probs = F.log_softmax(logits_bhwk, dim=-1)
        probs = F.softmax(logits_bhwk, dim=-1)

        probs_flat = probs.reshape(B * H * W, K)
        samples_flat = torch.multinomial(probs_flat, G, replacement=True)
        samples = samples_flat.reshape(B, H, W, G)

        target_expanded = target_bin.unsqueeze(-1)
        errors = (samples - target_expanded).abs().float()
        rewards = 1.0 - errors / (K - 1)

    advantages = advantage_fn(rewards.float())

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        sample_log_probs = log_probs.gather(-1, samples)
        pg_per_pixel = -(advantages.detach() * sample_log_probs).mean(dim=-1)
        valid_float = valid_mask.float()
        loss = (pg_per_pixel * valid_float).sum() / valid_float.sum().clamp(min=1.0)

    loss.backward()
    return get_trainable_grad_vector(model)


def analyze_checkpoint(ckpt_path, val_dataset, K, device, batch_size=8, num_batches=5):
    """Analyze one checkpoint: compute cosine similarity for all RL methods vs supervised.

    Args:
        ckpt_path: path to checkpoint .pt file
        val_dataset: NYUDepthDataset for validation
        K: number of bins
        device: torch device
        batch_size: images per batch for gradient computation
        num_batches: average over this many batches

    Returns:
        dict: method -> average cosine similarity
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # Create dataloader for a few batches
    from torch.utils.data import DataLoader
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=True)

    results = {method: [] for method in RL_METHODS}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        # Compute supervised reference gradient
        model = create_model(K=K, method="ordinal_ce", pretrained=False, seed=42)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        sup_grad = compute_supervised_gradient(model, batch, K, device)

        # Compute each RL method gradient (from the same model state)
        for method in RL_METHODS:
            model = create_model(K=K, method="ordinal_ce", pretrained=False, seed=42)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)

            torch.manual_seed(batch_idx * 1000)  # deterministic sampling
            rl_grad = compute_rl_gradient(model, batch, method, K, G_FOR_ANALYSIS, device)

            cos = compute_cosine_similarity(sup_grad, rl_grad)
            results[method].append(cos)

    # Average over batches
    avg_results = {}
    for method in RL_METHODS:
        if results[method]:
            avg_results[method] = float(sum(results[method]) / len(results[method]))
        else:
            avg_results[method] = float("nan")

    return avg_results


def run_analysis(results_dir, manifest_path, data_dir, output_dir,
                 K_values, epochs, device_str="cuda"):
    """Run full gradient analysis across K values and epochs.

    Saves results as JSON for plot.py Figure 7.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load manifest to find ordinal_ce runs (we use their checkpoints)
    with open(manifest_path) as f:
        manifest = list(csv.DictReader(f))

    # Load val dataset
    from datasets import load_from_disk
    full_train = load_from_disk(os.path.join(data_dir, "train"))
    split = full_train.train_test_split(test_size=0.1, seed=42)
    val_hf = split["test"]

    all_results = {}  # K -> epoch -> method -> cosine

    for K in K_values:
        all_results[K] = {}

        # Find ordinal_ce run for this K (any seed, we just need the checkpoint)
        target_runs = [r for r in manifest
                       if r["method"] == "ordinal_ce" and int(r["K"]) == K
                       and int(r["seed"]) == 42]

        if not target_runs:
            print(f"  No ordinal_ce run found for K={K}, skipping")
            continue

        run_dir = Path(results_dir) / target_runs[0]["job_id"]
        val_dataset = NYUDepthDataset(val_hf, K=K, train=False)

        for epoch in epochs:
            ckpt_path = run_dir / "checkpoints" / f"epoch_{epoch}.pt"
            if not ckpt_path.exists():
                print(f"  Checkpoint not found: {ckpt_path}")
                continue

            print(f"  Analyzing K={K}, epoch={epoch}...")
            epoch_results = analyze_checkpoint(ckpt_path, val_dataset, K, device)
            all_results[K][epoch] = epoch_results

            for method, cos in epoch_results.items():
                print(f"    {method}: cos = {cos:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gradient_cosine_results.json")

    # Convert keys to strings for JSON serialization
    json_results = {}
    for K, epochs_dict in all_results.items():
        json_results[str(K)] = {}
        for epoch, methods_dict in epochs_dict.items():
            json_results[str(K)][str(epoch)] = methods_dict

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Gradient cosine alignment analysis")
    parser.add_argument("--results_dir", type=str,
                        default="./experiment7_depth/results")
    parser.add_argument("--manifest", type=str,
                        default="./experiment7_depth/scripts/runs/manifest.csv")
    parser.add_argument("--data_dir", type=str, default="./data/nyu")
    parser.add_argument("--output_dir", type=str,
                        default="./experiment7_depth/figures")
    parser.add_argument("--K_values", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--epochs", nargs="+", type=int, default=[5, 15, 25])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_analysis(args.results_dir, args.manifest, args.data_dir, args.output_dir,
                 args.K_values, args.epochs, args.device)


if __name__ == "__main__":
    main()
