"""Gradient analysis for ordinal regression experiment.

Metric 7: Per-image gradient L2 norm (scatter vs P(exact match))
Metric 8: Per-threshold gradient norm decomposition

Computed at checkpoint epochs (1, 10, 25, 50) on 500 random val images.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import get_dataloaders, get_difficulty_tier
from .losses import ordinal_ce_loss, compute_window_probs
from .model import create_model


def compute_per_image_gradients(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    L: int,
    device: torch.device,
    N_rollouts: int = 10000,
) -> list[dict]:
    """Compute per-image gradient norms.

    For supervised methods: deterministic gradient from single image.
    For RL methods: use N_rollouts to estimate population gradient.

    Args:
        model: Trained model
        images: (K, C, H, W) batch of images
        targets: (K,) integer targets
        method: Method name
        L: Number of bins
        device: Torch device
        N_rollouts: Rollouts for RL gradient estimation

    Returns:
        List of dicts with per-image gradient info
    """
    model.eval()
    results = []

    for i in range(images.size(0)):
        img = images[i:i+1].to(device)  # (1, C, H, W)
        tgt = targets[i:i+1].to(device, dtype=torch.long)  # (1,)

        model.zero_grad()

        if method in ("ordinal_ce", "plain_ce"):
            logits = model(img)
            if method == "ordinal_ce":
                loss = ordinal_ce_loss(logits, tgt, L)
            else:
                loss = F.cross_entropy(logits, tgt)
            loss.backward()

        elif method == "mse":
            output = model(img)
            pred = torch.sigmoid(output.squeeze(-1))
            loss = F.mse_loss(pred, tgt.float() / (L - 1))
            loss.backward()

        else:
            # RL methods: estimate population gradient with many rollouts
            logits = model(img)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            samples = torch.multinomial(probs.expand(1, -1).squeeze(0), N_rollouts, replacement=True)  # (N,)
            errors = (samples - tgt.item()).abs().float()
            rewards = 1.0 - errors / (L - 1)

            from .advantages import ADVANTAGE_FNS
            advantages = ADVANTAGE_FNS[method](rewards)

            sample_log_probs = log_probs[0, samples]  # (N,)
            loss = -(advantages * sample_log_probs).mean()
            loss.backward()

        # Collect gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        # Compute analytical quantities
        with torch.no_grad():
            if method != "mse":
                logits = model(img)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                # W_0 = P(exact match)
                p_exact = probs[0, tgt.item()].item()
                # W_5
                w_probs = compute_window_probs(probs, tgt, L, [0, 5])
                p_within_5 = w_probs[5][0].item()

                # Ordinal CE value
                cum_probs = probs.cumsum(dim=-1)
                jml = 0.0
                for d in range(L - 1):
                    low = max(0, tgt.item() - d)
                    high = min(L - 1, tgt.item() + d)
                    upper = cum_probs[0, high].item()
                    lower = cum_probs[0, low - 1].item() if low > 0 else 0.0
                    w_d = max(upper - lower, 1e-12)
                    jml += np.log(w_d)
                jml /= (L - 1)

                # Expected reward
                bins = torch.arange(L, device=device).float()
                er = (probs[0] * (1.0 - (bins - tgt.item()).abs() / (L - 1))).sum().item()

                # Entropy
                entropy = -(probs[0] * log_probs[0]).sum().item()
            else:
                p_exact = 0.0
                p_within_5 = 0.0
                jml = 0.0
                er = 0.0
                entropy = 0.0

        results.append({
            "image_idx": i,
            "true_age": tgt.item(),
            "p_exact_match": p_exact,
            "p_within_5": p_within_5,
            "gradient_l2_norm": grad_norm,
            "ordinal_ce_value": jml,
            "expected_reward": er,
            "entropy": entropy,
        })

        model.zero_grad()

    return results


def compute_per_threshold_gradients(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    L: int,
    device: torch.device,
) -> list[dict]:
    """Compute per-threshold gradient norms for ordinal CE decomposition.

    For each image and each threshold d, compute ||grad_theta [-log W_d(x)]||_2.
    Only used for ordinal CE and continuous MaxRL.

    Args:
        model: Trained model
        images: (K, C, H, W) images (typically 50)
        targets: (K,) integer targets
        L: Number of bins
        device: Torch device

    Returns:
        List of dicts with per-threshold info
    """
    model.eval()
    results = []

    for i in range(images.size(0)):
        img = images[i:i+1].to(device)
        tgt = targets[i:i+1].to(device, dtype=torch.long)

        for d in range(L - 1):
            model.zero_grad()

            logits = model(img)
            probs = F.softmax(logits, dim=-1)
            cum_probs = probs.cumsum(dim=-1)

            low = max(0, tgt.item() - d)
            high = min(L - 1, tgt.item() + d)
            upper = cum_probs[0, high]
            lower = cum_probs[0, low - 1] if low > 0 else torch.tensor(0.0, device=device)
            W_d = (upper - lower).clamp(min=1e-12)

            neg_log_W = -torch.log(W_d)
            neg_log_W.backward()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            results.append({
                "image_idx": i,
                "threshold_d": d,
                "survival_prob": W_d.item(),
                "gradient_norm_at_threshold": grad_norm,
                "log_survival": np.log(max(W_d.item(), 1e-12)),
                "inverse_survival_weight": 1.0 / max(W_d.item(), 1e-12),
            })

            model.zero_grad()

    return results


def run_gradient_analysis(
    checkpoint_path: str,
    data_dir: str,
    method: str,
    L: int,
    seed: int,
    output_dir: str,
    n_images: int = 500,
    n_threshold_images: int = 50,
    use_wandb: bool = False,
):
    """Run full gradient analysis on a checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: UTKFace data directory
        method: Method name
        L: Number of bins
        seed: Random seed (for reproducible image selection)
        output_dir: Where to save results
        n_images: Number of images for per-image analysis (default 500)
        n_threshold_images: Number of images for per-threshold analysis (default 50)
        use_wandb: Log to W&B
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = ckpt["epoch"]
    print(f"Gradient analysis: method={method}, epoch={epoch}, L={L}")

    # Create model and load weights
    model = create_model(L=L, method=method, pretrained=True, seed=seed)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Load val data
    _, val_loader, _, _ = get_dataloaders(
        data_dir=data_dir, batch_size=256, L=L, num_workers=4,
    )

    # Collect all val images
    all_images, all_targets, all_ages = [], [], []
    for images, targets, ages in val_loader:
        all_images.append(images)
        all_targets.append(targets)
        all_ages.extend(ages.tolist() if torch.is_tensor(ages) else ages)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Random subset
    rng = np.random.RandomState(seed)
    n_val = all_images.size(0)
    perm = rng.permutation(n_val)

    # Per-image gradient analysis
    idx = perm[:n_images]
    print(f"  Computing per-image gradients for {len(idx)} images...")
    per_image_results = compute_per_image_gradients(
        model, all_images[idx], all_targets[idx], method, L, device,
    )

    # Add difficulty tier
    for j, r in enumerate(per_image_results):
        r["difficulty_tier"] = get_difficulty_tier(all_ages[idx[j]])

    # Per-threshold analysis (only for ordinal CE and cont_maxrl)
    per_threshold_results = None
    if method in ("ordinal_ce", "cont_maxrl"):
        t_idx = perm[:n_threshold_images]
        print(f"  Computing per-threshold gradients for {len(t_idx)} images...")
        per_threshold_results = compute_per_threshold_gradients(
            model, all_images[t_idx], all_targets[t_idx], L, device,
        )

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / f"gradient_per_image_epoch{epoch}.json", "w") as f:
        json.dump(per_image_results, f, indent=2)

    if per_threshold_results:
        with open(out_path / f"gradient_per_threshold_epoch{epoch}.json", "w") as f:
            json.dump(per_threshold_results, f, indent=2)

    # W&B logging
    if use_wandb:
        import wandb

        # Per-image table
        columns = ["image_idx", "true_age", "difficulty_tier",
                    "p_exact_match", "p_within_5", "gradient_l2_norm",
                    "ordinal_ce_value", "expected_reward", "entropy"]
        table = wandb.Table(columns=columns)
        for r in per_image_results:
            table.add_data(*[r[c] for c in columns])

        wandb.log({f"gradient_analysis/epoch_{epoch}": table})
        wandb.log({
            f"gradient_analysis/scatter_epoch_{epoch}": wandb.plot.scatter(
                table, "p_exact_match", "gradient_l2_norm",
                title=f"Gradient Norm vs P(exact) — Epoch {epoch}",
            )
        })

        # Per-threshold table
        if per_threshold_results:
            t_columns = ["image_idx", "threshold_d", "survival_prob",
                         "gradient_norm_at_threshold", "log_survival",
                         "inverse_survival_weight"]
            t_table = wandb.Table(columns=t_columns)
            for r in per_threshold_results:
                t_table.add_data(*[r[c] for c in t_columns])
            wandb.log({f"per_threshold/epoch_{epoch}": t_table})

    print(f"  Gradient analysis complete. Results saved to {out_path}")
    return per_image_results, per_threshold_results


def compute_cosine_vs_G(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    L: int,
    device: torch.device,
    G_values: list[int] = None,
    n_trials: int = 10,
    methods: list[str] = None,
    ref_method: str = "ordinal_ce",
) -> dict:
    """Compute cosine similarity between RL and supervised gradients at various G.

    For each method and each G (number of rollouts per image), estimates the
    batch policy gradient and measures cosine similarity with the supervised
    reference gradient. Repeats n_trials times for variance estimation.

    Args:
        model: Trained model (eval mode, on device)
        images: (K, C, H, W) batch of images (already on CPU)
        targets: (K,) integer targets (already on CPU)
        L: Number of bins
        device: Torch device
        G_values: List of rollout counts to evaluate
        n_trials: Repetitions per (method, G) for variance
        methods: RL methods to evaluate
        ref_method: Supervised reference ('ordinal_ce' or 'plain_ce')

    Returns:
        Dict: {method: {G: {"mean": float, "std": float, "values": [...]}}}
    """
    from .advantages import ADVANTAGE_FNS

    if G_values is None:
        G_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
    if methods is None:
        methods = ["cont_maxrl", "cont_maxrl_no_bl", "binary_maxrl", "grpo", "reinforce"]

    n_images = images.size(0)
    model.eval()

    # 1. Compute reference supervised gradient (deterministic)
    model.zero_grad()
    for i in range(n_images):
        img = images[i:i+1].to(device)
        tgt = targets[i:i+1].to(device, dtype=torch.long)
        logits = model(img)
        if ref_method == "ordinal_ce":
            loss = ordinal_ce_loss(logits, tgt, L)
        elif ref_method == "mse":
            # MSE on expected value from softmax distribution
            probs = F.softmax(logits, dim=-1)
            bins = torch.arange(L, device=device).float()
            pred = (probs * bins).sum(dim=-1) / (L - 1)
            loss = F.mse_loss(pred, tgt.float() / (L - 1))
        else:
            loss = F.cross_entropy(logits, tgt)
        (loss / n_images).backward()

    ref_grad = torch.cat([p.grad.flatten() for p in model.parameters()
                          if p.grad is not None]).clone()
    ref_norm = ref_grad.norm().item()
    model.zero_grad()
    print(f"  Reference ({ref_method}) gradient norm: {ref_norm:.4f}")

    # 2. For each method and G, compute RL gradient, cosine similarity, and magnitude ratio
    results = {}
    for method in methods:
        results[method] = {}
        adv_fn = ADVANTAGE_FNS[method]

        for G in G_values:
            cosines = []
            mag_ratios = []
            for trial in range(n_trials):
                model.zero_grad()
                for i in range(n_images):
                    img = images[i:i+1].to(device)
                    tgt = targets[i:i+1].to(device, dtype=torch.long)

                    logits = model(img)
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)

                    samples = torch.multinomial(probs[0], G, replacement=True)
                    errors = (samples - tgt.item()).abs().float()
                    rewards = 1.0 - errors / (L - 1)

                    advantages = adv_fn(rewards)
                    sample_log_probs = log_probs[0, samples]
                    img_loss = -(advantages * sample_log_probs).mean()
                    (img_loss / n_images).backward()

                rl_grad = torch.cat([p.grad.flatten() for p in model.parameters()
                                     if p.grad is not None])
                cos_sim = F.cosine_similarity(
                    ref_grad.unsqueeze(0), rl_grad.unsqueeze(0)
                ).item()
                rl_norm = rl_grad.norm().item()
                mag_ratio = rl_norm / ref_norm if ref_norm > 1e-12 else 0.0

                cosines.append(cos_sim)
                mag_ratios.append(mag_ratio)

            mean_cos = float(np.mean(cosines))
            std_cos = float(np.std(cosines))
            mean_mag = float(np.mean(mag_ratios))
            std_mag = float(np.std(mag_ratios))
            results[method][str(G)] = {
                "mean": mean_cos,
                "std": std_cos,
                "values": [float(c) for c in cosines],
                "mag_ratio_mean": mean_mag,
                "mag_ratio_std": std_mag,
                "mag_ratio_values": [float(m) for m in mag_ratios],
            }
            print(f"  {method} G={G}: cos={mean_cos:.4f}±{std_cos:.4f}  mag={mean_mag:.4f}±{std_mag:.4f}")

    return results


def run_cosine_analysis(
    checkpoint_path: str,
    data_dir: str,
    L: int,
    seed: int,
    output_dir: str,
    n_images: int = 200,
    n_trials: int = 10,
    ref_method: str = "ordinal_ce",
    methods: list[str] = None,
):
    """Run cosine similarity analysis: G vs cos(RL gradient, supervised gradient).

    Args:
        ref_method: Supervised reference loss ('ordinal_ce', 'plain_ce', or 'mse').
                    For 'mse', uses MSE on expected value from softmax distribution.
        methods: RL methods to evaluate. Default: all 5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = ckpt["epoch"]
    print(f"Cosine analysis: ref={ref_method}, epoch={epoch}, L={L}, n_images={n_images}")

    # MSE reference uses L-output model (ordinal_ce architecture) since
    # MSE model has 1-output regression head incompatible with RL sampling.
    model_method = "ordinal_ce" if ref_method == "mse" else ref_method
    model = create_model(L=L, method=model_method, pretrained=True, seed=seed)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    _, val_loader, _, _ = get_dataloaders(
        data_dir=data_dir, batch_size=256, L=L, num_workers=4,
    )

    all_images, all_targets = [], []
    for images, targets, _ages in val_loader:
        all_images.append(images)
        all_targets.append(targets)
    all_images = torch.cat(all_images, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(all_images.size(0))[:n_images]

    results = compute_cosine_vs_G(
        model, all_images[idx], all_targets[idx], L, device,
        n_trials=n_trials, methods=methods, ref_method=ref_method,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_ref_{ref_method}" if ref_method != "ordinal_ce" else ""
    out_file = out_path / f"cosine_vs_G_epoch{epoch}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump({"epoch": epoch, "n_images": n_images, "n_trials": n_trials,
                    "ref_method": ref_method, "results": results}, f, indent=2)
    print(f"Cosine analysis saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/utkface")
    parser.add_argument("--method", type=str, default=None,
                        help="Method for per-image analysis (omit for cosine analysis)")
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results/gradient_analysis")
    parser.add_argument("--n_images", type=int, default=500)
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Trials per (method, G) for cosine analysis")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mode", type=str, default="per_image",
                        choices=["per_image", "cosine"],
                        help="per_image: gradient norms; cosine: G vs cosine similarity")
    parser.add_argument("--ref_method", type=str, default="ordinal_ce",
                        choices=["ordinal_ce", "plain_ce", "mse"],
                        help="Supervised reference for cosine analysis")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="RL methods to evaluate (cosine mode)")
    args = parser.parse_args()

    if args.mode == "cosine":
        run_cosine_analysis(
            args.checkpoint, args.data_dir, args.L,
            args.seed, args.output_dir, args.n_images, args.n_trials,
            ref_method=args.ref_method, methods=args.methods,
        )
    else:
        if args.method is None:
            parser.error("--method is required for per_image mode")
        run_gradient_analysis(
            args.checkpoint, args.data_dir, args.method, args.L,
            args.seed, args.output_dir, args.n_images, use_wandb=args.wandb,
        )
