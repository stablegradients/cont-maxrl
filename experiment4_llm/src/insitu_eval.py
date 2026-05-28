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

# Route the scorer based on PIE_MODE. PIE submissions are C++ programs graded
# by perf instruction counts; experiment4_llm's scorer assumes Python and
# silently fails (every candidate scores 0). The reeval script handles this
# via a sys.modules shim, but during training that hook isn't installed —
# so the import has to be conditional here.
if os.environ.get("PIE_MODE") == "1":
    from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch, extract_code
else:
    from experiment4_llm.src.reward_function import compute_scores_batch, extract_code


def _best_at_k_unbiased(scores: list[float], k: int) -> float:
    """Unbiased estimator of E[max(K random samples from n scores)]. See
    experiment11_code_optim/eval/post_hoc_eval.py::best_at_k for full math.
    For binary {0,1} scores this equals pass@k."""
    n = len(scores)
    if n == 0:
        return 0.0
    if k >= n:
        return max(scores)
    s = sorted(scores, reverse=True)
    denom = math.comb(n, k)
    total = 0.0
    for i, x in enumerate(s, start=1):
        rem = n - i
        if rem < k - 1:
            break
        total += x * math.comb(rem, k - 1)
    return total / denom


def _env_k_values(default: list[int]) -> list[int]:
    v = os.environ.get("INSITU_EVAL_K_VALUES")
    if not v:
        return default
    try:
        out = json.loads(v)
        assert isinstance(out, list) and all(isinstance(x, int) for x in out)
        return out
    except (json.JSONDecodeError, AssertionError):
        print(f"[insitu_eval] WARN: INSITU_EVAL_K_VALUES={v!r} invalid; using {default}")
        return default


def score_and_save(
    completions: list[list[str]],
    ground_truths: list[str],
    epoch: int,
    global_step: int,
    output_dir: str,
    n_target: int = 200,
    k_values: list[int] | None = None,
    # Default 6s is the legacy DeepCoder value (Python programs run well
    # under that). PIE at -O0 needs 30s — set via PIE_PER_TEST_TIMEOUT=30
    # in the SLURM scripts so the env override propagates without changing
    # the DeepCoder default. If we don't read this env var here, every
    # insitu eval at -O0 systematically deflates pass@K relative to training
    # (training-side timeout_per_test=30 vs eval-side=6).
    timeout_per_test: int = int(os.environ.get("PIE_PER_TEST_TIMEOUT", "6")),
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
        k_values = _env_k_values(default=[1, 2, 5, 10, 20, 50, 100, 200])
    compute_best_at_k = bool(int(os.environ.get("INSITU_EVAL_BEST_AT_K", "0")))

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

    # Batch scoring: syntax filter, dedup, one subprocess per unique completion.
    # global_timeout=0 → None inside compute_scores_batch — disables the
    # phase-3 wall-clock cap. Insitu eval grades 195 * 256 ≈ 50k rollouts at
    # -O0, which takes ~45 min wall time and would silently zero out under
    # the default 300s cap. The PIE_INSITU_EVAL_GLOBAL_TIMEOUT env var lets
    # SLURM scripts override (e.g., set to 3600 if you want a safety cap).
    insitu_global_timeout_s = int(os.environ.get("PIE_INSITU_EVAL_GLOBAL_TIMEOUT", "0"))
    scores = compute_scores_batch(
        reward_inputs,
        timeout_per_test=timeout_per_test,
        num_workers=n_workers,
        global_timeout=insitu_global_timeout_s,
    )

    # Aggregate per problem
    problem_pass_counts: dict[int, int] = defaultdict(int)
    problem_total_counts: dict[int, int] = defaultdict(int)
    problem_scores_list: dict[int, list[float]] = defaultdict(list)
    # PIE: track per-problem speedup ratios when the reward fn provides them.
    # speedup is src_instr/cand_instr on gate-pass, 0 on gate-fail (see
    # experiment11_code_optim/reward/pie_reward_fn.py). Used for
    # best_speedup_at_k and frac_problems_{with_speedup,mode2} downstream.
    problem_speedups_list: dict[int, list[float]] = defaultdict(list)
    have_speedup = False
    for prob_idx, score_dict in zip(problem_indices, scores):
        problem_total_counts[prob_idx] += 1
        problem_scores_list[prob_idx].append(score_dict["score"])
        if score_dict["acc"] == 1.0:
            problem_pass_counts[prob_idx] += 1
        if "speedup" in score_dict:
            problem_speedups_list[prob_idx].append(float(score_dict["speedup"]))
            have_speedup = True

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
        problem_best_at_k = {}
        for k in k_values:
            if k <= n:
                problem_pass_at_k[str(k)] = pass_at_k(n, c, k)
                if compute_best_at_k:
                    problem_best_at_k[str(k)] = _best_at_k_unbiased(problem_scores_list[i], k)
            else:
                problem_pass_at_k[str(k)] = None
                if compute_best_at_k:
                    problem_best_at_k[str(k)] = None
        entry = {
            "problem_idx": i,
            "n": n,
            "c": c,
            "pass_at_k": problem_pass_at_k,
            "mean_score": sum(problem_scores_list[i]) / max(len(problem_scores_list[i]), 1),
            "unique_solution_rate": problem_unique_rates[i],
            "mean_output_length": problem_mean_lengths[i],
        }
        if compute_best_at_k:
            entry["best_at_k"] = problem_best_at_k
        per_problem_results.append(entry)

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

    # Macro-average best@k (only when enabled via INSITU_EVAL_BEST_AT_K=1)
    avg_best_at_k = None
    if compute_best_at_k:
        avg_best_at_k = {}
        for k in k_values:
            values = [
                r["best_at_k"][str(k)]
                for r in per_problem_results
                if r["best_at_k"][str(k)] is not None
            ]
            avg_best_at_k[str(k)] = sum(values) / len(values) if values else None

    # PIE: best_speedup@K and per-problem extras when score_dicts carry speedup.
    # best_speedup@K uses the same unbiased E[max(K samples)] estimator as
    # best_at_K, but applied to src/cand ratios instead of rewards. For gate-fail
    # rollouts speedup is 0 (so they pull the estimator down correctly when K is
    # large enough to include them).
    avg_best_speedup_at_k: dict[str, float] | None = None
    frac_problems_with_speedup: float | None = None
    frac_problems_mode2: float | None = None
    mean_speedup_given_pass: float | None = None
    if have_speedup:
        avg_best_speedup_at_k = {}
        for k in k_values:
            per_problem_vals: list[float] = []
            for i in range(n_problems):
                speedups_i = problem_speedups_list[i]
                if not speedups_i:
                    continue
                if k <= len(speedups_i):
                    per_problem_vals.append(_best_at_k_unbiased(speedups_i, k))
            avg_best_speedup_at_k[str(k)] = (
                sum(per_problem_vals) / len(per_problem_vals) if per_problem_vals else None
            )

        # A problem is "with speedup" iff at least one of its rollouts has
        # reward > 0 (= speedup > 1). "Mode-2" iff at least one rollout has
        # reward >= 0.3 (= speedup >= 1.23×, the empirical bimodality cutoff
        # from docs/DATA_ANALYSIS_2026_05_22_23.md).
        n_with_speedup = 0
        n_mode2 = 0
        for i in range(n_problems):
            scores_i = problem_scores_list[i]
            if any(s > 0 for s in scores_i):
                n_with_speedup += 1
            if any(s >= 0.3 for s in scores_i):
                n_mode2 += 1
        frac_problems_with_speedup = n_with_speedup / max(n_problems, 1)
        frac_problems_mode2 = n_mode2 / max(n_problems, 1)

        # mean_speedup over gate-passing rollouts (where speedup > 0).
        # Includes gate-pass-no-speedup rollouts (speedup ~ 1.0) so this is the
        # "average X×" the paper reports, not biased away from 1.
        all_pass_speedups = [s for sl in problem_speedups_list.values() for s in sl if s > 0]
        mean_speedup_given_pass = (
            sum(all_pass_speedups) / len(all_pass_speedups) if all_pass_speedups else None
        )

    metrics = {
        "epoch": epoch,
        "global_step": global_step,
        "pass_at_k": avg_pass_at_k,
        "best_at_k": avg_best_at_k,  # None when INSITU_EVAL_BEST_AT_K is unset
        "best_speedup_at_k": avg_best_speedup_at_k,  # None when no speedup field
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
            # PIE extras (None when reward fn doesn't expose speedup)
            "frac_problems_with_speedup": frac_problems_with_speedup,
            "frac_problems_mode2": frac_problems_mode2,
            "mean_speedup_given_pass": mean_speedup_given_pass,
        },
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_step{global_step}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  In-situ eval metrics saved to {path}")

    # Print summary — iterate over whatever K values were actually requested
    # so we don't blow up when (e.g.) K=[1,4,16,64,256] doesn't include 10/100/200.
    summary_parts = []
    for k in k_values:
        v = avg_pass_at_k.get(str(k))
        if v is None:
            continue
        summary_parts.append(f"pass@{k}={v:.4f}")
    print("  " + "  ".join(summary_parts))

    return metrics
