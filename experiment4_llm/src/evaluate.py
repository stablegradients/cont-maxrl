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
    top_p: float = 0.95,
    top_k: int = -1,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    enable_thinking: bool = False,
    max_num_seqs: int = 256,
    max_model_len: int | None = None,
    system_prompt: str | None = None,
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
    llm_kwargs = dict(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        dtype="bfloat16",
        enforce_eager=True,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=n_completions,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )

    # Format prompts using chat template
    chat_kwargs = {}
    if enable_thinking:
        chat_kwargs["enable_thinking"] = True
    formatted_prompts = []
    for problem in problems:
        prompt_messages = list(problem["prompt"])
        if system_prompt is not None:
            # Override system message
            prompt_messages = [
                {"role": "system", "content": system_prompt}
                if msg.get("role") == "system" else msg
                for msg in prompt_messages
            ]
        text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            **chat_kwargs,
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
    timeout_per_test: int = 6,
    n_workers: int = 128,
) -> dict:
    """
    Evaluate completions and compute pass@k.

    Uses compute_scores_batch for efficiency: syntax pre-check, code dedup,
    batched subprocess execution (one process per unique completion instead of
    one per test case), and early-stop on consecutive failures.

    Args:
        completions: completions[i] = list of code strings for problem i
        problems: list of problem dicts with ground_truth
        k_values: k values for pass@k
        timeout_per_test: timeout per test case (seconds)
        n_workers: number of parallel threads for evaluation

    Returns:
        dict with:
            - pass_at_k: {k: macro_average_pass_at_k}
            - per_problem: [{problem_idx, n, c, pass_at_k: {k: val}}]
            - summary: {total_problems, mean_reward, ...}
    """
    from experiment4_llm.src.reward_function import compute_scores_batch

    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50, 100]

    # Build flat list of reward inputs for batch scoring
    reward_inputs = []
    problem_indices = []
    for i, (problem, problem_completions) in enumerate(zip(problems, completions)):
        ground_truth = problem["reward_model"]["ground_truth"]
        for code in problem_completions:
            reward_inputs.append({
                "solution_str": code,
                "ground_truth": ground_truth,
            })
            problem_indices.append(i)

    print(f"  Evaluating {len(reward_inputs)} completions with {n_workers} workers (batched)...")

    # Batch scoring: syntax filter, dedup, one subprocess per unique completion
    scores = compute_scores_batch(
        reward_inputs,
        timeout_per_test=timeout_per_test,
        num_workers=n_workers,
    )

    # Aggregate scores per problem
    from collections import defaultdict
    problem_pass_counts: dict[int, int] = defaultdict(int)
    problem_total_counts: dict[int, int] = defaultdict(int)
    problem_scores: dict[int, list[float]] = defaultdict(list)
    for prob_idx, score_dict in zip(problem_indices, scores):
        problem_total_counts[prob_idx] += 1
        problem_scores[prob_idx].append(score_dict["score"])
        if score_dict["acc"] == 1.0:
            problem_pass_counts[prob_idx] += 1

    # Unique solution rate and output length per problem
    from experiment4_llm.src.reward_function import extract_code
    problem_unique_rates: dict[int, float] = {}
    problem_mean_lengths: dict[int, float] = {}
    for i, problem_completions in enumerate(completions):
        codes = [extract_code(c) for c in problem_completions]
        unique_codes = set(codes)
        problem_unique_rates[i] = len(unique_codes) / max(len(codes), 1)
        problem_mean_lengths[i] = sum(len(c) for c in problem_completions) / max(len(problem_completions), 1)

    per_problem_results = []
    for i in range(len(problems)):
        n = problem_total_counts[i]
        c = problem_pass_counts[i]

        problem_pass_at_k = {}
        for k in k_values:
            if k <= n:
                problem_pass_at_k[k] = pass_at_k(n, c, k)
            else:
                problem_pass_at_k[k] = None

        per_problem_results.append({
            "problem_idx": i,
            "n": n,
            "c": c,
            "pass_at_k": problem_pass_at_k,
            "mean_score": sum(problem_scores[i]) / max(len(problem_scores[i]), 1),
            "unique_solution_rate": problem_unique_rates[i],
            "mean_output_length": problem_mean_lengths[i],
        })

    print(f"  Evaluated {len(problems)} problems")

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
    all_scores = [s for sl in problem_scores.values() for s in sl]

    return {
        "pass_at_k": avg_pass_at_k,
        "per_problem": per_problem_results,
        "summary": {
            "total_problems": len(problems),
            "total_completions": total_completions,
            "total_passing": total_passing,
            "mean_pass_rate": total_passing / max(total_completions, 1),
            "mean_score": sum(all_scores) / max(len(all_scores), 1),
            "mean_unique_solution_rate": sum(problem_unique_rates.values()) / max(len(problem_unique_rates), 1),
            "mean_output_length": sum(problem_mean_lengths.values()) / max(len(problem_mean_lengths), 1),
            "k_values": k_values,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with pass@k")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_completions", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Pass enable_thinking=True to chat template (Qwen3 reasoning mode)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_num_seqs", type=int, default=256,
                        help="Max concurrent sequences in vLLM scheduler (lower = fewer KV cache preemptions)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Override model's default max context length (lower = more concurrent sequences)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override the system prompt baked into the dataset")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--timeout_per_test", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=128,
                        help="Number of parallel threads for reward evaluation")
    parser.add_argument(
        "--k_values", type=int, nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load problems
    print(f"Loading problems from {args.data_file}")
    problems = load_problems(args.data_file)
    print(f"  {len(problems)} problems loaded")

    # Generate completions (skip if already saved from a previous run)
    completions_path = Path(args.output_dir) / "completions.json"
    if completions_path.exists():
        print(f"Loading cached completions from {completions_path}")
        with open(completions_path) as f:
            completions = json.load(f)
        print(f"  {len(completions)} problems × {len(completions[0])} completions loaded")
    else:
        print(f"Generating {args.n_completions} completions per problem...")
        completions = generate_completions(
            model_path=args.checkpoint_dir,
            problems=problems,
            n_completions=args.n_completions,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_thinking=args.enable_thinking,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            system_prompt=args.system_prompt,
        )
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
        n_workers=args.n_workers,
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
    s = results["summary"]
    print(f"\n  Total problems:           {s['total_problems']}")
    print(f"  Mean pass rate:           {s['mean_pass_rate']:.4f}")
    print(f"  Mean score (frac tests):  {s['mean_score']:.4f}")
    print(f"  Mean unique solution rate: {s['mean_unique_solution_rate']:.4f}")
    print(f"  Mean output length (chars): {s['mean_output_length']:.0f}")


if __name__ == "__main__":
    main()
