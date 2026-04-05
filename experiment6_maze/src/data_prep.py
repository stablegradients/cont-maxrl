"""Generate RL training data (parquet) for maze navigation.

Each record contains:
  - data_source: "maze_{size}x{size}"
  - prompt: JSON-encoded chat messages (system + user)
  - reward_model: JSON-encoded dict with ground_truth text sequence
  - extra_info: JSON-encoded dict with l_max, optimal_len, maze_seed

Usage:
    PYTHONPATH=. python -m experiment6_maze.src.data_prep \
        --size 17 --n_train 50000 --n_val 512 \
        --train_seed_start 0 --val_seed_start 60000 \
        --l_max 64 --output_dir data/maze_17x17
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from experiment6_maze.src.maze_env import MAZE_SYSTEM_PROMPT, MazeGenerator


def build_record(seed: int, size: int = 17, l_max: int = 64, algorithm: str = "prim") -> dict:
    """Build a single RL training record for a maze with the given seed."""
    gen = MazeGenerator(size=size, seed=seed, algorithm=algorithm)
    gen.generate()

    optimal_actions = gen.optimal_path_actions()
    optimal_len = len(optimal_actions)

    return {
        "data_source": f"maze_{size}x{size}",
        "prompt": json.dumps([
            {"role": "system", "content": MAZE_SYSTEM_PROMPT},
            {"role": "user", "content": gen.to_prompt_string()},
        ]),
        "reward_model": json.dumps({"ground_truth": gen.to_text_sequence()}),
        "extra_info": json.dumps({
            "l_max": l_max,
            "optimal_len": optimal_len,
            "maze_seed": seed,
        }),
    }


def generate_dataset(
    n: int,
    seed_start: int,
    size: int = 17,
    l_max: int = 64,
    algorithm: str = "prim",
) -> pd.DataFrame:
    """Generate a DataFrame of n RL training records."""
    records = []
    for i in range(n):
        seed = seed_start + i
        records.append(build_record(seed, size=size, l_max=l_max, algorithm=algorithm))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate RL parquet data for maze training")
    parser.add_argument("--size", type=int, default=17, help="Maze grid size (must be odd, >= 5)")
    parser.add_argument("--algorithm", type=str, default="prim", help="Maze generation algorithm")
    parser.add_argument("--n_train", type=int, default=50000, help="Number of training mazes")
    parser.add_argument("--n_val", type=int, default=512, help="Number of validation mazes")
    parser.add_argument("--train_seed_start", type=int, default=0, help="First seed for training set")
    parser.add_argument("--val_seed_start", type=int, default=60000, help="First seed for validation set")
    parser.add_argument("--l_max", type=int, default=64, help="Maximum allowed path length for reward")
    parser.add_argument("--output_dir", type=str, default="data/maze_17x17", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Train set
    print(f"Generating {args.n_train} training mazes (seeds {args.train_seed_start}..{args.train_seed_start + args.n_train - 1})...")
    train_df = generate_dataset(
        n=args.n_train,
        seed_start=args.train_seed_start,
        size=args.size,
        l_max=args.l_max,
        algorithm=args.algorithm,
    )
    train_path = Path(args.output_dir) / "maze_train.parquet"
    train_df.to_parquet(train_path)
    print(f"  Written {len(train_df)} records to {train_path}")

    # Val set
    print(f"Generating {args.n_val} validation mazes (seeds {args.val_seed_start}..{args.val_seed_start + args.n_val - 1})...")
    val_df = generate_dataset(
        n=args.n_val,
        seed_start=args.val_seed_start,
        size=args.size,
        l_max=args.l_max,
        algorithm=args.algorithm,
    )
    val_path = Path(args.output_dir) / "maze_val.parquet"
    val_df.to_parquet(val_path)
    print(f"  Written {len(val_df)} records to {val_path}")


if __name__ == "__main__":
    main()
