"""Evaluation metrics for ordinal regression experiment.

All metrics computed on the validation set. Unless noted, computed
analytically from softmax distribution (no sampling at eval).

Metrics:
1. MAE (greedy argmax prediction)
2. Expected reward (analytical)
3. Best-of-N reward (sampled)
4. Ordinal CE objective (analytical)
5. Plain CE objective (analytical)
6. Cumulative accuracy at multiple thresholds
7. Policy entropy
8. Per-difficulty tier metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .data import get_difficulty_tier
from .losses import compute_window_probs


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    L: int,
    device: torch.device,
    best_of_n_trials: int = 100,
    best_of_n_values: tuple[int, ...] = (1, 8, 64, 256),
    method: str = "ordinal_ce",
) -> dict:
    """Compute all evaluation metrics.

    Args:
        model: Trained model
        dataloader: Validation DataLoader (yields image, target, raw_age)
        L: Number of age bins
        device: Torch device
        best_of_n_trials: Number of independent trials per image for best-of-N
        best_of_n_values: N values for best-of-N computation
        method: Method name (affects output type for MSE)

    Returns:
        Dict with all metrics
    """
    model.eval()

    # Accumulators
    all_maes = []
    all_expected_rewards = []
    all_ordinal_ce = []
    all_plain_ce = []
    all_mse_bins = []
    all_entropies = []
    all_top1_probs = []
    all_top5_probs = []
    all_acc = {d: [] for d in [0, 1, 2, 5, 10, 20]}
    all_survival = {}  # tau -> list
    all_best_of_n = {n: [] for n in best_of_n_values}
    all_reward_samples = []

    # Per-difficulty accumulators
    tier_metrics = {
        tier: {
            "maes": [], "expected_rewards": [], "ordinal_ce": [],
            "acc_exact": [], "acc_within_5": [],
            "best_of_64": [], "entropies": [],
        }
        for tier in ["easy", "medium", "hard"]
    }

    for batch in dataloader:
        images, targets, raw_ages = batch
        images = images.to(device)
        targets = targets.to(device, dtype=torch.long)

        if method == "mse":
            output = model(images)
            _compute_mse_metrics(
                output, targets, raw_ages, L,
                all_maes, all_expected_rewards, tier_metrics,
            )
            continue

        logits = model(images)  # (B, L)
        probs = F.softmax(logits, dim=-1)  # (B, L)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L)
        B = logits.size(0)

        # 1. MAE (greedy argmax)
        greedy_preds = probs.argmax(dim=-1)  # (B,)
        mae_per_image = (greedy_preds - targets).abs().float()  # (B,)
        all_maes.extend(mae_per_image.cpu().tolist())

        # 2. Expected reward (analytical)
        # E[r|x] = sum_b pi(b|x) * (1 - |b - y*|/(L-1))
        bin_indices = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        abs_errors = (bin_indices - targets.unsqueeze(1)).abs().float()  # (B, L)
        rewards_matrix = 1.0 - abs_errors / (L - 1)  # (B, L)
        expected_rewards = (probs * rewards_matrix).sum(dim=-1)  # (B,)
        all_expected_rewards.extend(expected_rewards.cpu().tolist())

        # 3. MSE in bins squared
        mse_per_image = ((greedy_preds - targets).float() ** 2)
        all_mse_bins.extend(mse_per_image.cpu().tolist())

        # 4. Ordinal CE objective (analytical, larger = better)
        # J_ML = (1/(L-1)) sum_{d=0}^{L-2} log W_d(x)
        cum_probs = probs.cumsum(dim=-1)  # (B, L)
        ordinal_ce_per_image = torch.zeros(B, device=device)
        for d in range(L - 1):
            low = (targets - d).clamp(min=0)
            high = (targets + d).clamp(max=L - 1)
            upper = cum_probs.gather(1, high.unsqueeze(1)).squeeze(1)
            lower = torch.where(
                low > 0,
                cum_probs.gather(1, (low - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                torch.zeros(B, device=device),
            )
            W_d = (upper - lower).clamp(min=1e-12)
            ordinal_ce_per_image += torch.log(W_d)
        ordinal_ce_per_image /= (L - 1)
        all_ordinal_ce.extend(ordinal_ce_per_image.cpu().tolist())

        # 5. Plain CE objective (analytical)
        plain_ce_per_image = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        all_plain_ce.extend(plain_ce_per_image.cpu().tolist())

        # 6. Cumulative accuracy at thresholds
        window_probs = compute_window_probs(probs, targets, L, list(all_acc.keys()))
        for d, w_d in window_probs.items():
            all_acc[d].extend(w_d.cpu().tolist())

        # 7. Policy entropy
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)
        all_entropies.extend(entropy.cpu().tolist())

        # Top-1 and top-5 probs
        top1_prob = probs.max(dim=-1).values  # (B,)
        all_top1_probs.extend(top1_prob.cpu().tolist())
        top5_prob = probs.topk(min(5, L), dim=-1).values.sum(dim=-1)  # (B,)
        all_top5_probs.extend(top5_prob.cpu().tolist())

        # Survival at representative thresholds
        # P(r > tau) = P(|error| <= d) where d = floor((L-1)*(1-tau) - eps)
        survival_thresholds = {
            0.50: 49, 0.80: 19, 0.90: 9, 0.95: 4, 0.99: 0,
        }
        for tau, d in survival_thresholds.items():
            if d not in window_probs:
                w_d = compute_window_probs(probs, targets, L, [d])[d]
            else:
                w_d = window_probs[d]
            key = f"{tau:.2f}"
            if key not in all_survival:
                all_survival[key] = []
            all_survival[key].extend(w_d.cpu().tolist())

        # 3. Best-of-N reward (sampled)
        for n_eval in best_of_n_values:
            # Sample n_eval predictions, repeat best_of_n_trials times
            # Shape: (B, trials, n_eval)
            samples = torch.multinomial(
                probs.unsqueeze(1).expand(-1, best_of_n_trials, -1).reshape(-1, L),
                n_eval, replacement=True,
            ).reshape(B, best_of_n_trials, n_eval)

            # Compute rewards
            sample_errors = (samples - targets.unsqueeze(1).unsqueeze(2)).abs().float()
            sample_rewards = 1.0 - sample_errors / (L - 1)

            # Best reward per trial, then average over trials
            best_per_trial = sample_rewards.max(dim=-1).values  # (B, trials)
            best_of_n_mean = best_per_trial.mean(dim=-1)  # (B,)
            all_best_of_n[n_eval].extend(best_of_n_mean.cpu().tolist())

        # Reward distribution (sample 256 per image)
        reward_samples = torch.multinomial(probs, 256, replacement=True)  # (B, 256)
        sample_errors = (reward_samples - targets.unsqueeze(1)).abs().float()
        sample_rewards = 1.0 - sample_errors / (L - 1)
        all_reward_samples.append(sample_rewards.cpu())

        # Per-difficulty metrics
        for i in range(B):
            age = raw_ages[i].item() if torch.is_tensor(raw_ages[i]) else raw_ages[i]
            tier = get_difficulty_tier(age)
            tier_metrics[tier]["maes"].append(mae_per_image[i].item())
            tier_metrics[tier]["expected_rewards"].append(expected_rewards[i].item())
            tier_metrics[tier]["ordinal_ce"].append(ordinal_ce_per_image[i].item())
            tier_metrics[tier]["acc_exact"].append(window_probs[0][i].item())
            tier_metrics[tier]["acc_within_5"].append(window_probs[5][i].item())
            tier_metrics[tier]["entropies"].append(entropy[i].item())
            if 64 in all_best_of_n and len(all_best_of_n[64]) > 0:
                tier_metrics[tier]["best_of_64"].append(all_best_of_n[64][-B + i])

    # Aggregate metrics
    import numpy as np

    metrics = {}

    if method == "mse":
        metrics["val/mae"] = np.mean(all_maes) if all_maes else 0.0
        metrics["val/expected_reward"] = np.mean(all_expected_rewards) if all_expected_rewards else 0.0
        # Per-tier
        for tier in ["easy", "medium", "hard"]:
            tm = tier_metrics[tier]
            if tm["maes"]:
                metrics[f"val_{tier}/mae"] = np.mean(tm["maes"])
                metrics[f"val_{tier}/expected_reward"] = np.mean(tm["expected_rewards"])
        return metrics

    metrics["val/mae"] = np.mean(all_maes)
    metrics["val/expected_reward"] = np.mean(all_expected_rewards)
    metrics["val/ordinal_ce_objective"] = np.mean(all_ordinal_ce)
    metrics["val/plain_ce_objective"] = np.mean(all_plain_ce)
    metrics["val/mse_in_bins_squared"] = np.mean(all_mse_bins)

    # Cumulative accuracy
    metrics["val/acc_exact"] = np.mean(all_acc[0])
    metrics["val/acc_within_1"] = np.mean(all_acc[1])
    metrics["val/acc_within_2"] = np.mean(all_acc[2])
    metrics["val/acc_within_5"] = np.mean(all_acc[5])
    metrics["val/acc_within_10"] = np.mean(all_acc[10])
    metrics["val/acc_within_20"] = np.mean(all_acc[20])

    # Best-of-N
    for n_eval in best_of_n_values:
        metrics[f"val/best_of_{n_eval}_reward"] = np.mean(all_best_of_n[n_eval])

    # Entropy stats
    entropies = np.array(all_entropies)
    metrics["val/entropy_mean"] = entropies.mean()
    metrics["val/entropy_std"] = entropies.std()
    metrics["val/entropy_min"] = entropies.min()
    metrics["val/entropy_max"] = entropies.max()

    # Top probs
    top1 = np.array(all_top1_probs)
    metrics["val/top1_prob_mean"] = top1.mean()
    metrics["val/top1_prob_std"] = top1.std()
    metrics["val/top5_prob_mean"] = np.mean(all_top5_probs)

    # Reward distribution
    all_r = torch.cat(all_reward_samples, dim=0)  # (total_images, 256)
    metrics["val/reward_mean"] = all_r.mean().item()
    metrics["val/reward_std"] = all_r.std().item()
    metrics["val/reward_median"] = all_r.median().item()
    metrics["val/reward_max_mean"] = all_r.max(dim=-1).values.mean().item()
    metrics["val/reward_min_mean"] = all_r.min(dim=-1).values.mean().item()

    # Survival at thresholds
    for tau_str, vals in all_survival.items():
        metrics[f"val/survival_tau_{tau_str}"] = np.mean(vals)

    # Per-difficulty metrics
    for tier in ["easy", "medium", "hard"]:
        tm = tier_metrics[tier]
        if not tm["maes"]:
            continue
        metrics[f"val_{tier}/mae"] = np.mean(tm["maes"])
        metrics[f"val_{tier}/expected_reward"] = np.mean(tm["expected_rewards"])
        metrics[f"val_{tier}/ordinal_ce_objective"] = np.mean(tm["ordinal_ce"])
        metrics[f"val_{tier}/acc_exact"] = np.mean(tm["acc_exact"])
        metrics[f"val_{tier}/acc_within_5"] = np.mean(tm["acc_within_5"])
        if tm["best_of_64"]:
            metrics[f"val_{tier}/best_of_64_reward"] = np.mean(tm["best_of_64"])
        metrics[f"val_{tier}/entropy_mean"] = np.mean(tm["entropies"])

    return metrics


def _compute_mse_metrics(output, targets, raw_ages, L, all_maes, all_expected_rewards, tier_metrics):
    """Compute metrics for MSE regression method."""
    pred_normalized = torch.sigmoid(output.squeeze(-1))  # (B,) in [0, 1]
    pred_bins = (pred_normalized * (L - 1)).round().long().clamp(0, L - 1)  # (B,)

    mae_per_image = (pred_bins - targets).abs().float()
    all_maes.extend(mae_per_image.cpu().tolist())

    # Expected reward for MSE: just the single prediction's reward
    rewards = 1.0 - mae_per_image / (L - 1)
    all_expected_rewards.extend(rewards.cpu().tolist())

    B = output.size(0)
    for i in range(B):
        age = raw_ages[i].item() if torch.is_tensor(raw_ages[i]) else raw_ages[i]
        tier = get_difficulty_tier(age)
        tier_metrics[tier]["maes"].append(mae_per_image[i].item())
        tier_metrics[tier]["expected_rewards"].append(rewards[i].item())
