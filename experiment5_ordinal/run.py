"""Main entry point for ordinal regression experiment.

Single run: trains one model with one method, one N, one L, one seed.
Handles W&B logging, checkpointing, and per-epoch evaluation.

Usage:
    python -m experiment5_ordinal.run \
        --method cont_maxrl --N 64 --L 100 --seed 42 \
        --epochs 50 --batch_size 64 --lr 1e-4 \
        --data_dir ./data/utkface --output_dir ./results --wandb
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import get_dataloaders
from .evaluate import evaluate_model
from .model import create_model
from .train import train_supervised_epoch, train_rl_epoch

SUPERVISED_METHODS = {"ordinal_ce", "plain_ce", "mse"}
RL_METHODS = {"cont_maxrl", "binary_maxrl", "grpo", "reinforce"}
ALL_METHODS = SUPERVISED_METHODS | RL_METHODS

# Epochs at which to save checkpoints for gradient analysis
GRADIENT_CHECKPOINT_EPOCHS = {1, 10, 25, 50}


def parse_args():
    parser = argparse.ArgumentParser(description="Ordinal Regression Experiment")
    parser.add_argument("--method", type=str, required=True, choices=sorted(ALL_METHODS))
    parser.add_argument("--N", type=int, default=64, help="Rollouts per image (RL methods)")
    parser.add_argument("--L", type=int, default=100, choices=[2, 100], help="Number of age bins")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate. Default: 1e-3 for supervised, 1e-4 for RL.")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--data_dir", type=str, default="./data/utkface")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Set default LR based on method type
    if args.lr is None:
        args.lr = 1e-3 if args.method in SUPERVISED_METHODS else 1e-4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run name and config
    run_name = f"{args.method}_L{args.L}_N{args.N}_seed{args.seed}"
    run_group = f"{args.method}_L{args.L}_N{args.N}"

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Method: {args.method} | N: {args.N} | L: {args.L} | Seed: {args.seed}")
    print(f"LR: {args.lr} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # W&B init
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="cont-maxrl-ordinal",
            group=run_group,
            config={
                "method": args.method,
                "N": args.N if args.method in RL_METHODS else None,
                "L": args.L,
                "seed": args.seed,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "optimizer": "sgd",
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "architecture": "resnet18_pretrained",
                "dataset": "utkface",
                "input_size": args.input_size,
            },
            tags=[args.method, f"L={args.L}", f"N={args.N}", f"seed={args.seed}"],
            name=run_name,
        )

    # Data
    train_loader, val_loader, test_loader, full_dataset = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        split_seed=42,  # always 42 for split
        L=args.L,
    )

    if wandb_run:
        wandb_run.config.update({
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        })

    # Model
    model = create_model(L=args.L, method=args.method, pretrained=True, seed=args.seed)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,
    )

    # LR scheduler: cosine decay over all steps
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    # Output directories
    run_dir = Path(args.output_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    start_time = time.time()
    best_mae = float("inf")
    best_ordinal_ce = float("-inf")

    # Local metrics log
    all_epoch_metrics = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        if args.method in SUPERVISED_METHODS:
            train_result = train_supervised_epoch(
                model, train_loader, optimizer, scheduler,
                args.method, args.L, device, epoch,
            )
        else:
            train_result = train_rl_epoch(
                model, train_loader, optimizer, scheduler,
                args.method, args.N, args.L, device, epoch,
            )

        # Log per-step metrics
        for step_log in train_result["step_logs"]:
            global_step += 1
            step_log["step"] = global_step
            step_log["epoch"] = epoch
            if wandb_run:
                import wandb
                wandb.log(step_log, step=global_step)

        # Evaluate
        eval_metrics = evaluate_model(
            model, val_loader, args.L, device,
            method=args.method,
        )
        eval_metrics["epoch"] = epoch

        # Track bests
        if "val/mae" in eval_metrics:
            if eval_metrics["val/mae"] < best_mae:
                best_mae = eval_metrics["val/mae"]
        if "val/ordinal_ce_objective" in eval_metrics:
            if eval_metrics["val/ordinal_ce_objective"] > best_ordinal_ce:
                best_ordinal_ce = eval_metrics["val/ordinal_ce_objective"]

        # Log RL stats
        if args.method in RL_METHODS and "rl_stats" in train_result:
            eval_metrics.update(train_result["rl_stats"])

        # W&B epoch logging
        if wandb_run:
            import wandb
            wandb.log(eval_metrics, step=global_step)

        # Print summary
        epoch_time = time.time() - epoch_start
        mae_str = f"MAE={eval_metrics.get('val/mae', 0):.2f}"
        jml_str = f"J_ML={eval_metrics.get('val/ordinal_ce_objective', 0):.4f}" if "val/ordinal_ce_objective" in eval_metrics else ""
        loss_str = f"loss={train_result['epoch_loss']:.4f}"
        print(f"Epoch {epoch:3d}/{args.epochs} | {loss_str} | {mae_str} | {jml_str} | {epoch_time:.1f}s")

        # Save checkpoint for gradient analysis
        if epoch in GRADIENT_CHECKPOINT_EPOCHS:
            ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "metrics": eval_metrics,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        all_epoch_metrics.append(eval_metrics)

    # Final checkpoint
    final_ckpt = ckpt_dir / "final.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_ckpt)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Best MAE: {best_mae:.2f}")

    # W&B summary
    if wandb_run:
        import wandb
        final = all_epoch_metrics[-1]
        wandb.run.summary.update({
            "final/mae": final.get("val/mae", 0),
            "final/ordinal_ce_objective": final.get("val/ordinal_ce_objective", 0),
            "final/plain_ce_objective": final.get("val/plain_ce_objective", 0),
            "final/expected_reward": final.get("val/expected_reward", 0),
            "final/best_of_64_reward": final.get("val/best_of_64_reward", 0),
            "final/entropy_mean": final.get("val/entropy_mean", 0),
            "final/acc_exact": final.get("val/acc_exact", 0),
            "final/acc_within_5": final.get("val/acc_within_5", 0),
            "final/total_train_time_seconds": total_time,
            "final/best_epoch_mae": best_mae,
            "final/best_epoch_ordinal_ce": best_ordinal_ce,
        })
        wandb.finish()

    # Save local metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_epoch_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
