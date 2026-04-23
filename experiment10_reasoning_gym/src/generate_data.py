"""Generate train/test parquet files for Reasoning Gym tasks.

Each parquet row has the schema required by verl's RLHFDataset + our
reward_manager.py:

    data_source  = "reasoning_gym_{task}"
    prompt       = [{"role": "user", "content": "<question>\\n\\nPut your answer within <answer> </answer> tags."}]
    reward_model = {"ground_truth": entry['answer']}
    extra_info   = {"task": task, "difficulty": <int|str>, "entry": json.dumps(entry)}

Usage:
    python -m experiment10_reasoning_gym.src.generate_data \\
        --task word_sorting --difficulty 8 \\
        --train_size 20000 --test_size 1000 --seed 42

Output:
    experiment10_reasoning_gym/data/{task}_diff{difficulty}/train.parquet
    experiment10_reasoning_gym/data/{task}_diff{difficulty}/test.parquet
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import reasoning_gym


ANSWER_INSTRUCTION = "\n\nPut your answer within <answer> </answer> tags."

# Optional per-task prompt preamble. For word_sorting, the built-in question
# already describes the task; we just append the answer-tag instruction.
TASK_PREAMBLE: dict[str, str] = {}


def _difficulty_kwargs(task: str, difficulty: int) -> dict[str, Any]:
    """Translate a single difficulty integer into task-specific kwargs.

    - word_sorting:         min_words = max_words = difficulty
    - sentence_reordering:  min_words_in_sentence = max_words_in_sentence = difficulty
    - knights_knaves:       n_people = difficulty
    - game_of_life:         grid_size_x = grid_size_y = difficulty
    """
    if task == "word_sorting":
        return {"min_words": difficulty, "max_words": difficulty}
    if task == "sentence_reordering":
        return {"min_words_in_sentence": difficulty, "max_words_in_sentence": difficulty}
    if task == "knights_knaves":
        return {"n_people": difficulty}
    if task == "game_of_life":
        return {"grid_size_x": difficulty, "grid_size_y": difficulty}
    raise ValueError(f"Unknown task: {task}")


def build_record(task: str, difficulty: int, entry: dict) -> dict:
    """Convert a reasoning_gym entry into a verl-compatible parquet row."""
    preamble = TASK_PREAMBLE.get(task, "")
    question_text = (preamble + entry["question"] if preamble else entry["question"]) + ANSWER_INSTRUCTION

    # JSON-encode the full entry so nested metadata survives parquet roundtrip.
    entry_json = json.dumps(entry, default=str)

    return {
        # diff-aware data_source so verl splits per-(task, difficulty) val metrics
        "data_source": f"reasoning_gym_{task}_diff{difficulty}",
        "prompt": [{"role": "user", "content": question_text}],
        "reward_model": {"ground_truth": str(entry["answer"])},
        "extra_info": {
            "task": task,
            "difficulty": int(difficulty),
            "entry": entry_json,
        },
    }


def generate_dataset(task: str, difficulty: int, size: int, seed: int) -> pd.DataFrame:
    """Generate ``size`` records for a (task, difficulty) using reasoning_gym."""
    kwargs = _difficulty_kwargs(task, difficulty)
    ds = reasoning_gym.create_dataset(task, size=size, seed=seed, **kwargs)
    records = [build_record(task, difficulty, ds[i]) for i in range(size)]
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning_gym parquets")
    parser.add_argument("--task", required=True,
                        choices=["word_sorting", "sentence_reordering",
                                 "knights_knaves", "game_of_life"])
    parser.add_argument("--difficulty", type=int, required=True,
                        help="Task-specific difficulty int (see _difficulty_kwargs)")
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed. Train uses this; test uses seed + 81 (non-overlapping).")
    parser.add_argument("--output_root", type=str,
                        default="experiment10_reasoning_gym/data")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / f"{args.task}_diff{args.difficulty}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_seed = args.seed
    test_seed = args.seed + 81  # disjoint RNG stream
    print(f"[{args.task} diff={args.difficulty}] train_size={args.train_size} test_size={args.test_size}")
    print(f"  train seed={train_seed}, test seed={test_seed}")

    print("Generating train set...")
    train_df = generate_dataset(args.task, args.difficulty,
                                size=args.train_size, seed=train_seed)
    train_path = out_dir / "train.parquet"
    train_df.to_parquet(train_path)
    print(f"  Wrote {len(train_df)} rows to {train_path}")

    print("Generating test set...")
    test_df = generate_dataset(args.task, args.difficulty,
                               size=args.test_size, seed=test_seed)
    test_path = out_dir / "test.parquet"
    test_df.to_parquet(test_path)
    print(f"  Wrote {len(test_df)} rows to {test_path}")


if __name__ == "__main__":
    main()
