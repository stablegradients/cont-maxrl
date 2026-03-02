"""Training loops for ordinal regression experiment.

Two training loops:
- train_supervised(): For ordinal CE, plain CE, MSE regression
- train_rl(): For continuous MaxRL, binary MaxRL, GRPO, REINFORCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .advantages import ADVANTAGE_FNS
from .losses import ordinal_ce_loss, mse_loss


def train_supervised_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    method: str,
    L: int,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train one epoch with supervised loss.

    Args:
        model: ResNet-18 model
        dataloader: Training DataLoader
        optimizer: SGD optimizer
        scheduler: Cosine LR scheduler
        method: "ordinal_ce", "plain_ce", or "mse"
        L: Number of age bins
        device: Torch device
        epoch: Current epoch number

    Returns:
        Dict with training stats for the epoch
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_steps = 0

    step_logs = []

    for batch_idx, (images, targets, _raw_ages) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()

        if method == "mse":
            output = model(images)  # (B, 1)
            loss = mse_loss(output, targets, L)
        elif method == "plain_ce":
            logits = model(images)  # (B, L)
            loss = F.cross_entropy(logits, targets)
        elif method == "ordinal_ce":
            logits = model(images)  # (B, L)
            loss = ordinal_ce_loss(logits, targets, L)
        else:
            raise ValueError(f"Unknown supervised method: {method}")

        loss.backward()

        # Compute gradient norm
        grad_norm = _compute_grad_norm(model)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_grad_norm += grad_norm
        num_steps += 1

        step_logs.append({
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/grad_norm_global": grad_norm,
        })

    return {
        "epoch_loss": total_loss / max(num_steps, 1),
        "epoch_grad_norm": total_grad_norm / max(num_steps, 1),
        "step_logs": step_logs,
    }


def train_rl_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    method: str,
    N: int,
    L: int,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train one epoch with RL policy gradient.

    For each image:
    1. Forward pass to get softmax distribution
    2. Sample N age bin predictions (free)
    3. Compute rewards (free)
    4. Compute advantages (method-dependent)
    5. Policy gradient: loss = -(1/N) * sum(A_i * log pi(b_hat_i | x))

    Args:
        model: ResNet-18 model
        dataloader: Training DataLoader
        optimizer: SGD optimizer
        scheduler: Cosine LR scheduler
        method: "cont_maxrl", "binary_maxrl", "grpo", or "reinforce"
        N: Number of rollouts per image
        L: Number of age bins
        device: Torch device
        epoch: Current epoch number

    Returns:
        Dict with training stats for the epoch
    """
    model.train()
    advantage_fn = ADVANTAGE_FNS[method]

    total_loss = 0.0
    total_grad_norm = 0.0
    num_steps = 0

    # RL-specific accumulators
    all_advantages = []
    all_rewards = []
    all_exact_matches = 0
    all_weight_on_top = []
    all_ess = []
    all_frac_unique = []

    step_logs = []

    for batch_idx, (images, targets, _raw_ages) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device, dtype=torch.long)
        B = images.size(0)

        optimizer.zero_grad()

        # 1. Forward pass (1 per batch)
        logits = model(images)  # (B, L)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L)
        probs = F.softmax(logits, dim=-1)  # (B, L)

        # 2. Sample N predictions per image (free — categorical sampling)
        samples = torch.multinomial(probs, N, replacement=True)  # (B, N)

        # 3. Compute rewards: r = 1 - |b_hat - y*| / (L-1)
        errors = (samples - targets.unsqueeze(1)).abs().float()  # (B, N)
        rewards = 1.0 - errors / (L - 1)  # (B, N)

        # 4. Compute advantages per image
        advantages = torch.zeros(B, N, device=device)
        for i in range(B):
            advantages[i] = advantage_fn(rewards[i])

        # 5. Policy gradient loss
        # log pi(b_hat_i | x) for each sample
        sample_log_probs = log_probs.gather(1, samples)  # (B, N)

        # loss = -(1/N) * sum(A_i * log pi(b_hat_i | x)), averaged over batch
        loss_per_image = -(advantages * sample_log_probs).mean(dim=1)  # (B,)
        loss = loss_per_image.mean()

        loss.backward()
        grad_norm = _compute_grad_norm(model)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_grad_norm += grad_norm
        num_steps += 1

        # Collect RL stats
        all_advantages.append(advantages.detach().cpu())
        all_rewards.append(rewards.detach().cpu())
        all_exact_matches += (rewards == 1.0).sum().item()

        # MaxRL-specific stats (before mean-centering)
        if method in ("cont_maxrl", "cont_maxrl_no_bl", "binary_maxrl"):
            for i in range(B):
                r = rewards[i]
                sorted_r, sort_idx = torch.sort(r)
                prev = torch.cat([torch.zeros(1, device=r.device), sorted_r[:-1]])
                gaps = sorted_r - prev
                survivors = torch.arange(N, 0, -1, device=r.device).float()
                increments = gaps / survivors
                weights = torch.cumsum(increments, dim=0)
                # Weight on top sample (before centering)
                w_abs_sum = weights.abs().sum().item()
                if w_abs_sum > 0:
                    all_weight_on_top.append(weights[-1].item() / w_abs_sum)
                # Effective sample size
                w_sq_sum = (weights ** 2).sum().item()
                if w_sq_sum > 0:
                    all_ess.append(w_abs_sum ** 2 / w_sq_sum)
                # Fraction unique rewards
                all_frac_unique.append(r.unique().numel() / N)

        # Reward stats for training
        step_logs.append({
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/grad_norm_global": grad_norm,
        })

    # Aggregate RL stats
    import numpy as np

    all_adv = torch.cat(all_advantages, dim=0)  # (total_images, N)
    all_r = torch.cat(all_rewards, dim=0)

    rl_stats = {
        "train_rl/advantage_mean": all_adv.mean().item(),
        "train_rl/advantage_std": all_adv.std().item(),
        "train_rl/advantage_max": all_adv.max().item(),
        "train_rl/advantage_min": all_adv.min().item(),
        "train_rl/advantage_abs_mean": all_adv.abs().mean().item(),
        "train_rl/frac_zero_advantage": (all_adv == 0).float().mean().item(),
        "train_rl/reward_mean": all_r.mean().item(),
        "train_rl/reward_std": all_r.std().item(),
        "train_rl/reward_max_mean": all_r.max(dim=-1).values.mean().item(),
        "train_rl/num_exact_matches": all_exact_matches,
    }

    if all_weight_on_top:
        rl_stats["train_rl/weight_on_top_sample"] = np.mean(all_weight_on_top)
        rl_stats["train_rl/effective_sample_size"] = np.mean(all_ess)
        rl_stats["train_rl/frac_unique_rewards"] = np.mean(all_frac_unique)

    return {
        "epoch_loss": total_loss / max(num_steps, 1),
        "epoch_grad_norm": total_grad_norm / max(num_steps, 1),
        "step_logs": step_logs,
        "rl_stats": rl_stats,
    }


def _compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5
