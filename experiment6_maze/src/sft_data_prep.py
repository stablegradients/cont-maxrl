"""Generate SFT training data (JSONL) for maze navigation.

Each record contains:
  - prompt: system prompt + user maze description
  - completion: optimal actions separated by spaces, ending with DONE

SFT seed ranges (100000+) must NOT overlap with RL seed ranges (0-49999, 60000-60511).

Usage:
    PYTHONPATH=. python -m experiment6_maze.src.sft_data_prep \
        --size 17 --n_train 100000 --n_val 512 \
        --train_seed_start 100000 --val_seed_start 200000 \
        --output_dir data/maze_17x17
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiment6_maze.src.maze_env import MAZE_SYSTEM_PROMPT, MazeGenerator


def build_sft_record(seed: int, size: int = 17) -> dict:
    """Build a single SFT training record for a maze with the given seed."""
    gen = MazeGenerator(size=size, seed=seed)
    gen.generate()

    optimal_actions = gen.optimal_path_actions()
    completion = " ".join(optimal_actions) + " DONE"

    return {
        "prompt": f"{MAZE_SYSTEM_PROMPT}\n\n{gen.to_prompt_string()}",
        "completion": completion,
    }


def generate_sft_dataset(
    n: int,
    seed_start: int,
    size: int = 17,
) -> list[dict]:
    """Generate a list of n SFT training records."""
    records = []
    for i in range(n):
        seed = seed_start + i
        records.append(build_sft_record(seed, size=size))
    return records


def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records to a JSONL file."""
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT JSONL data for maze training")
    parser.add_argument("--size", type=int, default=17, help="Maze grid size (must be odd, >= 5)")
    parser.add_argument("--n_train", type=int, default=100000, help="Number of training mazes")
    parser.add_argument("--n_val", type=int, default=512, help="Number of validation mazes")
    parser.add_argument("--train_seed_start", type=int, default=100000, help="First seed for training set")
    parser.add_argument("--val_seed_start", type=int, default=200000, help="First seed for validation set")
    parser.add_argument("--output_dir", type=str, default="data/maze_17x17", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Train set
    print(f"Generating {args.n_train} SFT training mazes (seeds {args.train_seed_start}..{args.train_seed_start + args.n_train - 1})...")
    train_records = generate_sft_dataset(
        n=args.n_train,
        seed_start=args.train_seed_start,
        size=args.size,
    )
    train_path = Path(args.output_dir) / "sft_train.jsonl"
    write_jsonl(train_records, train_path)
    print(f"  Written {len(train_records)} records to {train_path}")

    # Val set
    print(f"Generating {args.n_val} SFT validation mazes (seeds {args.val_seed_start}..{args.val_seed_start + args.n_val - 1})...")
    val_records = generate_sft_dataset(
        n=args.n_val,
        seed_start=args.val_seed_start,
        size=args.size,
    )
    val_path = Path(args.output_dir) / "sft_val.jsonl"
    write_jsonl(val_records, val_path)
    print(f"  Written {len(val_records)} records to {val_path}")


if __name__ == "__main__":
    main()
