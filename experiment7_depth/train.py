"""Training step functions for dense depth estimation."""

import torch
import torch.nn.functional as F

from .losses import dense_ordinal_ce_loss, dense_plain_ce_loss, dense_mse_loss
from .advantages import ADVANTAGE_FNS


def compute_grad_norm(model):
    """Global L2 norm over all trainable parameters."""
    # Unwrap DDP if needed
    m = model.module if hasattr(model, "module") else model
    total = 0.0
    for p in m.parameters():
        if p.grad is not None and p.requires_grad:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def get_poly_lr(step: int, total_steps: int, base_lr: float, power: float = 0.9) -> float:
    """Polynomial LR decay."""
    return base_lr * (1 - step / total_steps) ** power


def train_supervised_step(model, batch, optimizer, scheduler, method, K):
    """One supervised training step.

    Args:
        model: DepthResNet18UNet (possibly DDP-wrapped)
        batch: dict with image, target_bin, target_m, valid_mask tensors (already on device)
        optimizer: SGD optimizer
        scheduler: LR scheduler (callable(step) -> lr), or None
        method: one of 'ordinal_ce', 'plain_ce', 'mse'
        K: number of bins

    Returns:
        dict with 'loss' and 'grad_norm'
    """
    images = batch["image"]
    target_bin = batch["target_bin"]
    target_m = batch["target_m"]
    valid_mask = batch["valid_mask"]

    optimizer.zero_grad(set_to_none=True)

    device_type = "cuda" if images.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=images.is_cuda):
        output = model(images)
        if method == "ordinal_ce":
            loss = dense_ordinal_ce_loss(output, target_bin, valid_mask, K)
        elif method == "plain_ce":
            loss = dense_plain_ce_loss(output, target_bin, valid_mask)
        elif method == "mse":
            loss = dense_mse_loss(output, target_m, valid_mask)
        else:
            raise ValueError(f"Unknown supervised method: {method}")

    loss.backward()
    grad_norm = compute_grad_norm(model)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {"loss": loss.item(), "grad_norm": grad_norm}


def train_rl_step(model, batch, optimizer, scheduler, method, G, K):
    """One RL training step with dense per-pixel rollouts.

    Args:
        model: DepthResNet18UNet (possibly DDP-wrapped)
        batch: dict with image, target_bin, valid_mask tensors (already on device)
        optimizer: SGD optimizer
        scheduler: LR scheduler, or None
        method: one of 'cont_maxrl', 'cont_maxrl_no_bl', 'binary_maxrl', 'grpo', 'reinforce'
        G: number of rollouts per pixel
        K: number of bins

    Returns:
        dict with loss, grad_norm, reward_mean, reward_std, advantage_abs_mean,
        num_exact_matches, exact_match_rate
    """
    images = batch["image"]
    target_bin = batch["target_bin"]
    valid_mask = batch["valid_mask"]

    advantage_fn = ADVANTAGE_FNS[method]

    optimizer.zero_grad(set_to_none=True)

    device_type = "cuda" if images.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=images.is_cuda):
        logits = model(images)                                    # (B, K, H, W)
        B, _, H, W = logits.shape

        # Permute to (B, H, W, K) for per-pixel categorical ops
        logits_bhwk = logits.permute(0, 2, 3, 1).contiguous()    # (B, H, W, K)
        log_probs = F.log_softmax(logits_bhwk, dim=-1)            # (B, H, W, K)
        probs = F.softmax(logits_bhwk, dim=-1)                    # (B, H, W, K)

        # Sample G rollouts per pixel
        probs_flat = probs.reshape(B * H * W, K)                   # (BHW, K)
        samples_flat = torch.multinomial(probs_flat, G, replacement=True)  # (BHW, G)
        samples = samples_flat.reshape(B, H, W, G)                # (B, H, W, G)

        # Compute rewards: r = 1 - |b_hat - y| / (K - 1)
        target_expanded = target_bin.unsqueeze(-1)                # (B, H, W, 1)
        errors = (samples - target_expanded).abs().float()        # (B, H, W, G)
        rewards = 1.0 - errors / (K - 1)                          # (B, H, W, G)

    # Compute advantages per-pixel, vectorized over (B, H, W) — outside autocast for fp32
    advantages = advantage_fn(rewards.float())                     # (B, H, W, G)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=images.is_cuda):
        # Gather log-probs of sampled actions
        sample_log_probs = log_probs.gather(-1, samples)          # (B, H, W, G)

        # Policy gradient loss: -(1/G) * sum_g A_g * log pi(a_g), averaged over valid pixels
        pg_per_pixel = -(advantages.detach() * sample_log_probs).mean(dim=-1)  # (B, H, W)

        valid_float = valid_mask.float()
        num_valid = valid_float.sum().clamp(min=1.0)
        loss = (pg_per_pixel * valid_float).sum() / num_valid

    loss.backward()
    grad_norm = compute_grad_norm(model)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # RL stats (detached for logging)
    with torch.no_grad():
        # Expand valid_mask to match rewards shape for masking
        valid_expanded = valid_mask.unsqueeze(-1).expand_as(rewards)  # (B, H, W, G)
        valid_rewards = rewards[valid_expanded]
        valid_advantages = advantages[valid_expanded]
        num_exact_matches = (valid_rewards == 1.0).sum().item()
        total_valid_rollouts = max(valid_rewards.numel(), 1)
        exact_match_rate = num_exact_matches / total_valid_rollouts

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm,
        "reward_mean": valid_rewards.mean().item(),
        "reward_std": valid_rewards.std().item(),
        "advantage_abs_mean": valid_advantages.abs().mean().item(),
        "num_exact_matches": num_exact_matches,
        "exact_match_rate": exact_match_rate,
    }
