"""In-situ maze evaluation during training."""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

from experiment6_maze.src.maze_env import (
    MazeEnv, parse_actions, check_success_with_steps, continuous_reward,
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def score_and_save(
    completions: list[list[str]],
    ground_truths: list[str],
    extra_infos: list[dict],
    epoch: int,
    global_step: int,
    output_dir: str,
    n_target: int = 64,
    k_values: list[int] | None = None,
) -> dict:
    if k_values is None:
        k_values = [1, 4, 16]

    n_problems = len(completions)
    print(f"  Maze eval: scoring {n_problems} problems, up to {n_target} completions each...")

    per_problem = []
    all_cont_rewards = []
    all_steps_success = []
    all_efficiency = []
    all_format_errors = 0
    all_total = 0

    for i in range(n_problems):
        gt = ground_truths[i]
        ei = extra_infos[i]
        l_max = ei.get("l_max", 64)
        optimal_len = ei.get("optimal_len", 1)

        n_success = 0
        n_total = 0
        n_format_error = 0
        problem_cont_rewards = []
        problem_steps = []

        for sol in completions[i][:n_target]:
            n_total += 1
            all_total += 1

            actions = parse_actions(sol)
            if actions is None:
                n_format_error += 1
                all_format_errors += 1
                problem_cont_rewards.append(0.0)
                continue

            env = MazeEnv.from_sequence(gt)
            if env is None:
                problem_cont_rewards.append(0.0)
                continue

            r_cont = continuous_reward(actions, env, l_max)
            problem_cont_rewards.append(r_cont)
            all_cont_rewards.append(r_cont)

            env2 = MazeEnv.from_sequence(gt)
            success, steps = check_success_with_steps(env2, actions)
            if success:
                n_success += 1
                problem_steps.append(steps)
                all_steps_success.append(steps)
                if optimal_len > 0:
                    all_efficiency.append(optimal_len / steps)

        problem_pass_at_k = {}
        for k in k_values:
            if k <= n_total:
                problem_pass_at_k[str(k)] = pass_at_k(n_total, n_success, k)

        per_problem.append({
            "problem_idx": i,
            "n": n_total,
            "c": n_success,
            "pass_at_k": problem_pass_at_k,
            "mean_cont_reward": sum(problem_cont_rewards) / max(len(problem_cont_rewards), 1),
            "avg_steps_success": sum(problem_steps) / max(len(problem_steps), 1) if problem_steps else -1,
            "format_error_rate": n_format_error / max(n_total, 1),
        })

    total_success = sum(r["c"] for r in per_problem)
    total_completions = sum(r["n"] for r in per_problem)

    avg_pass_at_k = {}
    for k in k_values:
        vals = [r["pass_at_k"].get(str(k)) for r in per_problem if str(k) in r["pass_at_k"]]
        avg_pass_at_k[str(k)] = sum(vals) / len(vals) if vals else None

    metrics = {
        "epoch": epoch,
        "global_step": global_step,
        "pass_at_k": avg_pass_at_k,
        "summary": {
            "success_rate": total_success / max(total_completions, 1),
            "avg_steps_success": sum(all_steps_success) / max(len(all_steps_success), 1) if all_steps_success else -1,
            "efficiency_ratio": sum(all_efficiency) / max(len(all_efficiency), 1) if all_efficiency else 0.0,
            "continuous_reward_mean": sum(all_cont_rewards) / max(len(all_cont_rewards), 1) if all_cont_rewards else 0.0,
            "format_error_rate": all_format_errors / max(all_total, 1),
            "total_problems": n_problems,
            "total_completions": total_completions,
            "total_success": total_success,
        },
        "per_problem": per_problem,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_step{global_step}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Maze eval saved to {path}")

    p1 = avg_pass_at_k.get("1", "N/A")
    sr = metrics["summary"]["success_rate"]
    avg_s = metrics["summary"]["avg_steps_success"]
    eff = metrics["summary"]["efficiency_ratio"]
    p1_str = f"{p1:.4f}" if isinstance(p1, float) else str(p1)
    print(f"  pass@1={p1_str}  success_rate={sr:.4f}  avg_steps={avg_s:.1f}  efficiency={eff:.4f}")

    return metrics
