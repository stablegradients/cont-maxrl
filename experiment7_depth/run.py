"""Main entry point for dense depth estimation experiment.

Single run: trains one model with one method, K, G, seed.
Supports 4-GPU DDP via torchrun or single-GPU fallback.

Usage (DDP):
    torchrun --standalone --nproc_per_node=4 -m experiment7_depth.run \
        --method ordinal_ce --K 100 --seed 42 --global_batch_size 32 \
        --output_dir ./results/test --amp

Usage (single-GPU):
    python -m experiment7_depth.run \
        --method ordinal_ce --K 100 --seed 42 --global_batch_size 8 \
        --output_dir ./results/test --amp
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import create_model, convert_decoder_bns_to_sync
from .data import get_dataloaders, NYUDepthDataset
from .train import train_supervised_step, train_rl_step, get_poly_lr
from .evaluate import evaluate_model, evaluate_accumulators, accumulators_to_metrics

SUPERVISED_METHODS = {"ordinal_ce", "plain_ce", "mse"}
RL_METHODS = {"cont_maxrl", "cont_maxrl_no_bl", "binary_maxrl", "grpo", "reinforce"}
ALL_METHODS = SUPERVISED_METHODS | RL_METHODS

GRADIENT_CHECKPOINT_EPOCHS = {5, 15, 25}


def setup_ddp():
    """Initialize DDP from torchrun env vars."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Dense Depth Estimation Experiment")
    parser.add_argument("--method", type=str, required=True, choices=sorted(ALL_METHODS))
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--G", type=int, default=64, help="Rollouts per pixel (RL only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--epochs_hard_cap", type=int, default=35)
    parser.add_argument("--epochs_min_before_es", type=int, default=18)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_schedule", type=str, default="polynomial")
    parser.add_argument("--lr_power", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data_dir", type=str, default="./data/nyu")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--print_memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # DDP or single-GPU
    use_ddp = "RANK" in os.environ
    if use_ddp:
        local_rank, world_size, rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        world_size = 1
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert args.global_batch_size % world_size == 0
    local_batch_size = args.global_batch_size // world_size

    if is_main_process():
        print(f"Method: {args.method} | K: {args.K} | G: {args.G} | Seed: {args.seed}")
        print(f"DDP: {use_ddp} | World size: {world_size} | Local batch: {local_batch_size}")
        print(f"LR: {args.lr} | Epochs: {args.epochs} (hard cap {args.epochs_hard_cap})")

    # Model
    model_method = "mse" if args.method == "mse" else "ordinal_ce"
    model = create_model(K=args.K, method=model_method, pretrained=True, seed=args.seed)
    if use_ddp:
        model = convert_decoder_bns_to_sync(model)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # Data
    from datasets import load_from_disk
    train_path = os.path.join(args.data_dir, "train")
    val_path = os.path.join(args.data_dir, "validation")
    full_train_hf = load_from_disk(train_path)
    test_hf = load_from_disk(val_path)
    split = full_train_hf.train_test_split(test_size=0.1, seed=42)
    train_hf, val_hf = split["train"], split["test"]

    train_dataset = NYUDepthDataset(train_hf, K=args.K, train=True)
    val_dataset = NYUDepthDataset(val_hf, K=args.K, train=False)

    # Bin edges/centers for eval
    bin_edges = torch.linspace(0.5, 10.0, args.K + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                            shuffle=True, seed=args.seed, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                          shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=local_batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=local_batch_size,
        sampler=val_sampler, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=True,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)

    # LR schedule
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs_hard_cap * steps_per_epoch
    global_step = 0

    # W&B
    wandb_run = None
    use_wandb = args.wandb and not args.no_wandb and is_main_process()
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project="cont-maxrl-depth",
            config=vars(args),
            tags=["zx7k"],
            name=args.wandb_run_name,
        )

    # Output dir
    run_dir = Path(args.output_dir)
    ckpt_dir = run_dir / "checkpoints"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping state
    best_mae = float("inf")
    best_epoch = 0
    all_epoch_metrics = []
    start_time = time.time()

    for epoch in range(1, args.epochs_hard_cap + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = []
        epoch_grad_norms = []
        epoch_rl_stats = {}

        for step_idx, batch in enumerate(train_loader):
            if args.max_steps is not None and step_idx >= args.max_steps:
                break

            # Move to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # Update LR
            global_step += 1
            lr = get_poly_lr(global_step, total_steps, args.lr, args.lr_power)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Training step
            if args.method in SUPERVISED_METHODS:
                result = train_supervised_step(model, batch, optimizer, None, args.method, args.K)
            else:
                result = train_rl_step(model, batch, optimizer, None, args.method, args.G, args.K)

            epoch_losses.append(result["loss"])
            epoch_grad_norms.append(result["grad_norm"])

            # Collect RL stats
            if args.method in RL_METHODS:
                for stat_key in ["reward_mean", "reward_std", "advantage_abs_mean",
                                 "exact_match_rate"]:
                    if stat_key in result:
                        epoch_rl_stats.setdefault(stat_key, []).append(result[stat_key])

            # Per-step W&B logging
            if wandb_run and global_step % 10 == 0:
                import wandb
                log_dict = {"train/loss": result["loss"], "train/grad_norm": result["grad_norm"],
                            "train/lr": lr, "train/step": global_step}
                if args.method in RL_METHODS:
                    log_dict.update({
                        "train_rl/reward_mean": result.get("reward_mean", 0),
                        "train_rl/exact_match_rate": result.get("exact_match_rate", 0),
                        "train_rl/advantage_abs_mean": result.get("advantage_abs_mean", 0),
                    })
                wandb.log(log_dict, step=global_step)

            # Print memory on first step
            if args.print_memory and step_idx == 0 and epoch == 1:
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated(device) / 1e9
                    print(f"[Rank {rank}] Peak GPU memory after step 1: {peak:.2f} GB")

        # Eval
        if use_ddp:
            accum = evaluate_accumulators(model, val_loader, args.K, args.method,
                                          device, bin_centers, bin_edges)
            for key in accum:
                dist.all_reduce(accum[key], op=dist.ReduceOp.SUM)
            val_metrics = accumulators_to_metrics(accum, args.method)
        else:
            val_metrics = evaluate_model(model, val_loader, args.K, args.method,
                                         device, bin_centers, bin_edges)

        val_metrics["epoch"] = epoch
        val_metrics["train/loss_mean"] = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_metrics["train/grad_norm_mean"] = sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1)

        # RL epoch stats
        if args.method in RL_METHODS:
            for stat_key, vals in epoch_rl_stats.items():
                val_metrics[f"train_rl/{stat_key}"] = sum(vals) / max(len(vals), 1)

        if is_main_process():
            all_epoch_metrics.append(val_metrics)

            # Track best
            mae = val_metrics.get("val/mae_meters", float("inf"))
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                # Save best checkpoint
                raw_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch, "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args), "metrics": val_metrics,
                }, ckpt_dir / "best.pt")

            # Gradient analysis checkpoints
            if epoch in GRADIENT_CHECKPOINT_EPOCHS:
                raw_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch, "model_state_dict": raw_model.state_dict(),
                    "args": vars(args), "metrics": val_metrics,
                }, ckpt_dir / f"epoch_{epoch}.pt")

            # Print
            elapsed = time.time() - start_time
            mae_str = f"MAE={mae:.4f}m"
            loss_str = f"loss={val_metrics['train/loss_mean']:.4f}"
            print(f"Epoch {epoch:3d}/{args.epochs_hard_cap} | {loss_str} | {mae_str} | "
                  f"best={best_mae:.4f}m@{best_epoch} | {elapsed:.0f}s")

            # W&B epoch log
            if wandb_run:
                import wandb
                wandb.log(val_metrics, step=global_step)

        # Early stopping decision (all ranks participate)
        should_stop = False
        if is_main_process():
            if epoch >= args.epochs_min_before_es and epoch - best_epoch >= args.early_stop_patience:
                should_stop = True
            if epoch >= args.epochs and epoch - best_epoch >= args.early_stop_patience:
                should_stop = True

        if use_ddp:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            if is_main_process():
                print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

        # Also stop if max_steps was set (for memory_check)
        if args.max_steps is not None:
            break

    # Final save
    if is_main_process():
        raw_model = model.module if hasattr(model, "module") else model
        torch.save({
            "epoch": epoch, "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }, ckpt_dir / "final.pt")

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_epoch_metrics, f, indent=2)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.0f}s. Best MAE: {best_mae:.4f}m at epoch {best_epoch}")
        print(f"Results: {run_dir}")

        if wandb_run:
            import wandb
            wandb.run.summary.update({
                "final/best_mae_meters": best_mae,
                "final/best_epoch": best_epoch,
                "final/total_epochs": epoch,
                "final/wall_clock_seconds": total_time,
            })
            wandb.finish()

    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
