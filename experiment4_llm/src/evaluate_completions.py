"""
Evaluate pre-generated completions (from a JSON file) without re-running generation.

Usage:
    python experiment4_llm/src/evaluate_completions.py \
        --completions_file results/.../completions.json \
        --data_file data/apps_test.jsonl \
        --output_file results/.../eval_results.json
"""
from __future__ import annotations

import argparse
import json
import os

from experiment4_llm.src.evaluate import evaluate_completions, load_problems


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pre-generated completions"
    )
    parser.add_argument("--completions_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--timeout_per_test", type=int, default=10)
    parser.add_argument(
        "--k_values", type=int, nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100],
    )
    args = parser.parse_args()

    # Load
    print(f"Loading problems from {args.data_file}")
    problems = load_problems(args.data_file)
    print(f"  {len(problems)} problems")

    print(f"Loading completions from {args.completions_file}")
    with open(args.completions_file) as f:
        completions = json.load(f)
    print(f"  {len(completions)} problem sets, {len(completions[0])} completions each")

    assert len(completions) == len(problems), (
        f"Mismatch: {len(completions)} completion sets vs {len(problems)} problems"
    )

    # Evaluate
    print("Evaluating...")
    results = evaluate_completions(
        completions=completions,
        problems=problems,
        k_values=args.k_values,
        timeout_per_test=args.timeout_per_test,
    )

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")

    # Print summary
    print("\n=== pass@k Results ===")
    for k, v in results["pass_at_k"].items():
        if v is not None:
            print(f"  pass@{k}: {v:.4f} ({v*100:.1f}%)")


if __name__ == "__main__":
    main()
