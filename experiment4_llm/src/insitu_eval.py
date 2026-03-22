"""
In-situ LiveCodeBench evaluation during training.

Generates N completions per problem using the training rollout engine,
scores them, computes pass@k, and saves metrics_epochN.json at configured
epoch boundaries.

This runs inline during training — the trainer pauses while eval runs,
then resumes. The existing vLLM rollout engine is reused (no extra GPU needed).
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from experiment4_llm.src.evaluate import pass_at_k
from experiment4_llm.src.reward_function import compute_scores_batch, extract_code


def score_and_save(
    completions: list[list[str]],
    ground_truths: list[str],
    epoch: int,
    global_step: int,
    output_dir: str,
    n_target: int = 200,
    k_values: list[int] | None = None,
    timeout_per_test: int = 6,
    n_workers: int = 128,
) -> dict:
    """Score completions, compute pass@k, and save metrics_epochN.json.

    Args:
        completions: completions[i] = list of decoded strings for problem i
        ground_truths: ground_truths[i] = JSON string with test cases for problem i
        epoch: current epoch number
        global_step: current global training step
        output_dir: directory to save metrics_epochN.json
        n_target: target number of completions per problem (truncate to this)
        k_values: k values for pass@k computation
        timeout_per_test: timeout per test case (seconds)
        n_workers: number of parallel threads for scoring

    Returns:
        dict with pass_at_k, per_problem, summary
    """
    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50, 100, 200]

    n_problems = len(completions)
    print(f"  In-situ eval: scoring {n_problems} problems, {n_target} completions each...")

    # Build flat list of reward inputs for batch scoring
    reward_inputs = []
    problem_indices = []
    for i in range(n_problems):
        gt = ground_truths[i]
        for code in completions[i][:n_target]:
            reward_inputs.append({
                "solution_str": code,
                "ground_truth": gt,
            })
            problem_indices.append(i)

    # Batch scoring: syntax filter, dedup, one subprocess per unique completion
    scores = compute_scores_batch(
        reward_inputs,
        timeout_per_test=timeout_per_test,
        num_workers=n_workers,
    )

    # Aggregate per problem
    problem_pass_counts: dict[int, int] = defaultdict(int)
    problem_total_counts: dict[int, int] = defaultdict(int)
    problem_scores_list: dict[int, list[float]] = defaultdict(list)
    for prob_idx, score_dict in zip(problem_indices, scores):
        problem_total_counts[prob_idx] += 1
        problem_scores_list[prob_idx].append(score_dict["score"])
        if score_dict["acc"] == 1.0:
            problem_pass_counts[prob_idx] += 1

    # Unique solution rate and output length per problem
    problem_unique_rates: dict[int, float] = {}
    problem_mean_lengths: dict[int, float] = {}
    for i in range(n_problems):
        codes = [extract_code(c) for c in completions[i][:n_target]]
        unique_codes = set(codes)
        n_actual = max(len(codes), 1)
        problem_unique_rates[i] = len(unique_codes) / n_actual
        problem_mean_lengths[i] = sum(len(c) for c in completions[i][:n_target]) / n_actual

    per_problem_results = []
    for i in range(n_problems):
        n = problem_total_counts[i]
        c = problem_pass_counts[i]
        problem_pass_at_k = {}
        for k in k_values:
            if k <= n:
                problem_pass_at_k[str(k)] = pass_at_k(n, c, k)
            else:
                problem_pass_at_k[str(k)] = None
        per_problem_results.append({
            "problem_idx": i,
            "n": n,
            "c": c,
            "pass_at_k": problem_pass_at_k,
            "mean_score": sum(problem_scores_list[i]) / max(len(problem_scores_list[i]), 1),
            "unique_solution_rate": problem_unique_rates[i],
            "mean_output_length": problem_mean_lengths[i],
        })

    # Macro-average pass@k
    avg_pass_at_k = {}
    for k in k_values:
        values = [
            r["pass_at_k"][str(k)]
            for r in per_problem_results
            if r["pass_at_k"][str(k)] is not None
        ]
        avg_pass_at_k[str(k)] = sum(values) / len(values) if values else None

    # Summary
    total_passing = sum(r["c"] for r in per_problem_results)
    total_completions = sum(r["n"] for r in per_problem_results)
    all_scores = [s for sl in problem_scores_list.values() for s in sl]

    # Number of problems with at least one passing solution
    n_problems_solved = sum(1 for r in per_problem_results if r["c"] > 0)

    # Global unique code rate: unique extracted codes / total completions
    all_codes = []
    for i in range(n_problems):
        all_codes.extend(extract_code(c) for c in completions[i][:n_target])
    unique_code_rate_global = len(set(all_codes)) / max(len(all_codes), 1)

    metrics = {
        "epoch": epoch,
        "global_step": global_step,
        "pass_at_k": avg_pass_at_k,
        "per_problem": per_problem_results,
        "summary": {
            "total_problems": n_problems,
            "total_completions": total_completions,
            "total_passing": total_passing,
            "n_problems_solved": n_problems_solved,
            "mean_pass_rate": total_passing / max(total_completions, 1),
            "mean_score": sum(all_scores) / max(len(all_scores), 1),
            "mean_unique_solution_rate": sum(problem_unique_rates.values()) / max(len(problem_unique_rates), 1),
            "mean_output_length": sum(problem_mean_lengths.values()) / max(len(problem_mean_lengths), 1),
            "unique_code_rate_global": unique_code_rate_global,
        },
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_step{global_step}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  In-situ eval metrics saved to {path}")

    # Print summary
    print(f"  pass@1={avg_pass_at_k.get('1', 'N/A'):.4f}  "
          f"pass@10={avg_pass_at_k.get('10', 'N/A'):.4f}  "
          f"pass@100={avg_pass_at_k.get('100', 'N/A'):.4f}  "
          f"pass@200={avg_pass_at_k.get('200', 'N/A'):.4f}")

    return metrics
