"""SFT training for Qwen2 ~3.94M on maze navigation.

Trains from scratch (or from checkpoint) with PATH_START masking:
only action tokens after PATH_START are supervised.

Matches the MaxRL paper (Tajwar et al., arXiv 2602.02710):
  - Custom PyTorch training loop with gradient accumulation
  - Cosine LR schedule, AdamW(betas=(0.9, 0.95))
  - Generative evaluation with Pass@k metrics
  - 1500 steps, batch_size=32, lr=5e-4

Usage:
    PYTHONPATH=. python -m experiment6_maze.src.sft \
        --model_path checkpoints/maze17x17_qwen2_3m_init \
        --train_data data/maze_17x17/sft_train.parquet \
        --val_data data/maze_17x17/sft_val.parquet \
        --output_dir checkpoints/maze17x17_qwen2_3m_sft
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MazeSFTDataset(Dataset):
    """Load parquet with 'sequence' column, tokenize, mask labels before PATH_START."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[dict] = []

        path_start_id = tokenizer.encode("PATH_START", add_special_tokens=False)
        if len(path_start_id) != 1:
            raise ValueError(f"PATH_START should be 1 token, got {path_start_id}")
        self.path_start_id = path_start_id[0]

        logger.info(f"Loading {data_path}...")
        t0 = time.time()
        df = pd.read_parquet(data_path)
        for _, row in df.iterrows():
            ex = self._process(row["sequence"], row.get("prompt", None))
            if ex is not None:
                self.examples.append(ex)
        logger.info(f"Loaded {len(self.examples)}/{len(df)} examples in {time.time() - t0:.1f}s")

    def _process(self, sequence: str, prompt: Optional[str] = None) -> Optional[dict]:
        input_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        try:
            ps_idx = input_ids.index(self.path_start_id)
        except ValueError:
            return None

        # Mask everything up to and including PATH_START
        labels = [-100] * (ps_idx + 1) + input_ids[ps_idx + 1 :]

        if prompt is None:
            prompt = sequence.split("PATH_START")[0].strip() + " PATH_START"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt": prompt,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"],
            "labels": ex["labels"],
            "attention_mask": torch.ones_like(ex["input_ids"]),
        }

    def get_prompt(self, idx) -> str:
        return self.examples[idx]["prompt"]


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    bs = len(batch)
    input_ids = torch.full((bs, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"]
        attention_mask[i, :n] = item["attention_mask"]
        labels[i, :n] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Generative evaluator
# ---------------------------------------------------------------------------


class MazeEvaluator:
    """Evaluate generated paths by simulating them on the maze grid."""

    ACTIONS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

    def parse_and_validate(self, sequence: str) -> tuple[bool, str]:
        """Parse grid + actions from a generated sequence, simulate, check success."""
        try:
            gs = sequence.find("GRID_START")
            ge = sequence.find("GRID_END")
            if gs == -1 or ge == -1:
                return False, "invalid_grid"

            grid_content = sequence[gs + len("GRID_START") : ge].strip()
            lines = [l.strip() for l in grid_content.split("NEWLINE") if l.strip()]
            grid, start, goal = [], None, None
            for r, line in enumerate(lines):
                row = []
                for c, tok in enumerate(line.split()):
                    if tok == "WALL":
                        row.append(1)
                    elif tok == "START":
                        row.append(0)
                        start = (r, c)
                    elif tok == "GOAL":
                        row.append(0)
                        goal = (r, c)
                    else:
                        row.append(0)
                if row:
                    grid.append(row)

            if not grid or start is None or goal is None:
                return False, "invalid_grid"
            size = len(grid)

            # Parse actions after PATH_START
            ps = sequence.find("PATH_START")
            if ps == -1:
                return False, "no_actions"
            actions_str = sequence[ps + len("PATH_START") :].strip().split()
            actions = []
            for t in actions_str:
                if t in self.ACTIONS:
                    actions.append(t)
                elif t == "DONE":
                    break

            if not actions:
                return False, "no_actions"

            # Simulate
            pos = start
            for a in actions:
                dr, dc = self.ACTIONS[a]
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr][nc] == 0:
                    pos = (nr, nc)
                if pos == goal:
                    return True, "success"

            return False, "not_at_goal"
        except Exception:
            return False, "error"

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """Unbiased Pass@k estimator."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MazeSFTTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model from {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Datasets
        self.train_ds = MazeSFTDataset(args.train_data, self.tokenizer, args.max_length)
        self.val_ds = MazeSFTDataset(args.val_data, self.tokenizer, args.max_length)

        pad_id = self.tokenizer.pad_token_id
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id),
            num_workers=4,
            pin_memory=True,
        )

        # Optimizer + scheduler
        self.optimizer = AdamW(
            self.model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
        )
        steps_per_epoch = len(self.train_loader)
        if args.max_steps > 0:
            total_steps = args.max_steps
        else:
            total_steps = steps_per_epoch * args.num_epochs
        self.total_steps = total_steps
        warmup_steps = int(total_steps * args.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.evaluator = MazeEvaluator()
        self.global_step = 0
        os.makedirs(args.output_dir, exist_ok=True)

        # wandb
        self.use_wandb = args.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=args.project_name, name=args.experiment_name, config=vars(args))

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {num_params:,} params")
        logger.info(f"Train: {len(self.train_ds)} samples, Val: {len(self.val_ds)} samples")
        logger.info(f"Steps/epoch: {steps_per_epoch}, Total: {self.total_steps}, Warmup: {warmup_steps}")

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        micro_bs = self.args.micro_batch_size
        bs = input_ids.size(0)
        accum_steps = max(1, bs // micro_bs)

        self.optimizer.zero_grad()
        total_loss = 0.0

        for i in range(0, bs, micro_bs):
            j = min(i + micro_bs, bs)
            outputs = self.model(
                input_ids=input_ids[i:j],
                attention_mask=attention_mask[i:j],
                labels=labels[i:j],
            )
            loss = outputs.loss / accum_steps
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return total_loss

    @torch.no_grad()
    def validate_loss(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.val_loader:
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            total += outputs.loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def generative_eval(
        self, num_prompts: int = 100, n_samples: int = 8, temperature: float = 1.0
    ) -> dict[str, float]:
        """Sample n_samples completions per prompt, compute Pass@k."""
        self.model.eval()
        indices = np.random.choice(len(self.val_ds), min(num_prompts, len(self.val_ds)), replace=False)

        done_id = self.tokenizer.encode("DONE", add_special_tokens=False)
        eos_ids = [self.tokenizer.eos_token_id] + (done_id if done_id else [])

        success_counts = []
        for idx in tqdm(indices, desc="GenEval"):
            prompt = self.val_ds.get_prompt(idx)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            try:
                out = self.model.generate(
                    input_ids,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=n_samples,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
                count = 0
                for i in range(n_samples):
                    text = self.tokenizer.decode(out[i], skip_special_tokens=False)
                    ok, _ = self.evaluator.parse_and_validate(text)
                    if ok:
                        count += 1
                success_counts.append(count)
            except Exception as e:
                logger.debug(f"Generation error: {e}")
                success_counts.append(0)

        metrics = {}
        for k in [1, 2, 4, 8]:
            if k <= n_samples:
                vals = [MazeEvaluator.pass_at_k(n_samples, c, k) for c in success_counts]
                metrics[f"eval/pass@{k}"] = float(np.mean(vals))
        metrics["eval/avg_success"] = sum(success_counts) / max(len(success_counts) * n_samples, 1)
        return metrics

    def save_checkpoint(self, step: int):
        path = os.path.join(self.args.output_dir, f"ckpt-{step}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved checkpoint to {path}")

    def train(self):
        logger.info("Starting training...")
        done = False
        epoch = 0
        while not done:
            epoch += 1
            epoch_loss, nbatch = 0.0, 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                self.global_step += 1
                loss = self.train_step(batch)
                epoch_loss += loss
                nbatch += 1
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    step=f"{self.global_step}/{self.total_steps}",
                )

                if self.use_wandb:
                    wandb.log(
                        {"train/loss": loss, "train/lr": self.scheduler.get_last_lr()[0]},
                        step=self.global_step,
                    )

                if self.global_step % self.args.eval_steps == 0:
                    val_loss = self.validate_loss()
                    log_msg = f"Step {self.global_step} - Val Loss: {val_loss:.4f}"
                    metrics = {"eval/val_loss": val_loss}

                    if self.args.use_generative_eval:
                        gen_metrics = self.generative_eval(
                            num_prompts=self.args.eval_samples,
                            n_samples=self.args.n_samples_per_prompt,
                            temperature=self.args.eval_temperature,
                        )
                        metrics.update(gen_metrics)
                        log_msg += f" Pass@1={gen_metrics.get('eval/pass@1', 0):.3f}"

                    logger.info(log_msg)
                    if self.use_wandb:
                        wandb.log(metrics, step=self.global_step)

                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint(self.global_step)

                if self.global_step >= self.total_steps:
                    done = True
                    break

            logger.info(f"Epoch {epoch} avg loss: {epoch_loss / max(nbatch, 1):.4f}")

        # Final save
        self.save_checkpoint(self.global_step)
        if self.use_wandb:
            wandb.finish()
        logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Maze SFT Trainer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to init model + tokenizer")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Training
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 = use num_epochs)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    # Eval
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--use_generative_eval", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--n_samples_per_prompt", type=int, default=8)
    parser.add_argument("--eval_temperature", type=float, default=1.0)

    # Logging
    parser.add_argument("--project_name", type=str, default="maze-sft")
    parser.add_argument("--experiment_name", type=str, default="qwen2_3m_sft")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()
    trainer = MazeSFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
