"""
Evaluation: generate completions and compute pass@k.

N=200 completions per problem at temperature 0.8.
Binary pass/fail (all tests pass or not).
Unbiased pass@k for k in {1, 2, 5, 10, 20, 50, 100}.

Usage:
    python experiment4_llm/src/evaluate.py \
        --checkpoint_dir results/7b_apps_maxrl_unnorm_g16_s42/checkpoint-300 \
        --data_file data/apps_test.jsonl \
        --output_dir results/7b_apps_maxrl_unnorm_g16_s42/eval_step300 \
        --n_completions 200 \
        --temperature 0.8 \
        --max_tokens 2048
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.

    Args:
        n: total completions generated
        c: number that pass ALL test cases
        k: number of attempts allowed

    Returns:
        Estimated probability of at least one passing solution in k attempts.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_problems(data_file: str) -> list[dict]:
    """Load problems from parquet or jsonl file (auto-detected by extension)."""
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_file)
        return df.to_dict(orient="records")

    # JSONL fallback
    problems = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def generate_completions(
    model_path: str,
    problems: list[dict],
    n_completions: int = 200,
    temperature: float = 0.8,
    max_tokens: int = 2048,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
) -> list[list[str]]:
    """
    Generate N completions per problem using vLLM.

    Args:
        model_path: path to model checkpoint or HF model ID
        problems: list of problem dicts with "prompt" field
        n_completions: completions per problem
        temperature: sampling temperature
        max_tokens: max tokens per completion
        gpu_memory_utilization: vLLM GPU memory fraction
        tensor_parallel_size: number of GPUs for tensor parallelism

    Returns:
        list of lists: completions[i] = [str, ...] for problem i
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        n=n_completions,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # Format prompts using chat template
    formatted_prompts = []
    for problem in problems:
        prompt_messages = problem["prompt"]
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(text)

    # Generate
    outputs = llm.generate(formatted_prompts, sampling_params)

    all_completions = []
    for output in outputs:
        completions = [o.text for o in output.outputs]
        all_completions.append(completions)

    return all_completions


def evaluate_completions(
    completions: list[list[str]],
    problems: list[dict],
    k_values: list[int] | None = None,
    timeout_per_test: int = 10,
    n_workers: int = 1,
) -> dict:
    """
    Evaluate completions and compute pass@k.

    Args:
        completions: completions[i] = list of code strings for problem i
        problems: list of problem dicts with ground_truth
        k_values: k values for pass@k
        timeout_per_test: timeout per test case
        n_workers: number of parallel workers for evaluation

    Returns:
        dict with:
            - pass_at_k: {k: macro_average_pass_at_k}
            - per_problem: [{problem_idx, n, c, pass_at_k: {k: val}}]
            - summary: {total_problems, mean_reward, ...}
    """
    from experiment4_llm.src.reward_function import compute_score_binary

    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50, 100]

    per_problem_results = []

    for i, (problem, problem_completions) in enumerate(zip(problems, completions)):
        ground_truth = problem["reward_model"]["ground_truth"]
        n = len(problem_completions)

        # Count passing completions (binary: all tests pass)
        c = 0
        for code in problem_completions:
            score = compute_score_binary(
                code, ground_truth, timeout_per_test=timeout_per_test
            )
            if score == 1.0:
                c += 1

        # Compute pass@k for each k
        problem_pass_at_k = {}
        for k in k_values:
            if k <= n:
                problem_pass_at_k[k] = pass_at_k(n, c, k)
            else:
                problem_pass_at_k[k] = None  # k > n, can't compute

        per_problem_results.append({
            "problem_idx": i,
            "n": n,
            "c": c,
            "pass_at_k": problem_pass_at_k,
        })

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(problems)} problems")

    # Macro-average pass@k
    avg_pass_at_k = {}
    for k in k_values:
        values = [
            r["pass_at_k"][k]
            for r in per_problem_results
            if r["pass_at_k"][k] is not None
        ]
        if values:
            avg_pass_at_k[k] = sum(values) / len(values)
        else:
            avg_pass_at_k[k] = None

    # Summary statistics
    total_passing = sum(r["c"] for r in per_problem_results)
    total_completions = sum(r["n"] for r in per_problem_results)

    return {
        "pass_at_k": avg_pass_at_k,
        "per_problem": per_problem_results,
        "summary": {
            "total_problems": len(problems),
            "total_completions": total_completions,
            "total_passing": total_passing,
            "mean_pass_rate": total_passing / max(total_completions, 1),
            "k_values": k_values,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with pass@k")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_completions", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--timeout_per_test", type=int, default=10)
    parser.add_argument(
        "--k_values", type=int, nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load problems
    print(f"Loading problems from {args.data_file}")
    problems = load_problems(args.data_file)
    print(f"  {len(problems)} problems loaded")

    # Generate completions
    print(f"Generating {args.n_completions} completions per problem...")
    completions = generate_completions(
        model_path=args.checkpoint_dir,
        problems=problems,
        n_completions=args.n_completions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Save completions
    completions_path = Path(args.output_dir) / "completions.json"
    with open(completions_path, "w") as f:
        json.dump(completions, f)
    print(f"  Completions saved to {completions_path}")

    # Evaluate
    print("Evaluating completions...")
    results = evaluate_completions(
        completions=completions,
        problems=problems,
        k_values=args.k_values,
        timeout_per_test=args.timeout_per_test,
    )

    # Save results
    results_path = Path(args.output_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Print summary
    print("\n=== pass@k Results ===")
    for k, v in results["pass_at_k"].items():
        if v is not None:
            print(f"  pass@{k}: {v:.4f} ({v*100:.1f}%)")
    print(f"\n  Total problems: {results['summary']['total_problems']}")
    print(f"  Mean pass rate: {results['summary']['mean_pass_rate']:.4f}")


if __name__ == "__main__":
    main()
