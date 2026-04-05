"""verl-compatible reward functions for maze navigation."""
from __future__ import annotations

from experiment6_maze.src.maze_env import (
    MazeEnv, continuous_reward, binary_reward, parse_actions,
    check_success_with_steps,
)


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None):
    """Continuous maze reward."""
    l_max = (extra_info or {}).get("l_max", 64)
    actions = parse_actions(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    if actions is None or env is None:
        return {"score": 0.0, "acc": 0.0, "success": 0.0, "steps": -1, "format_ok": 0.0}
    r_cont = continuous_reward(actions, env, l_max)
    env = MazeEnv.from_sequence(ground_truth)  # re-parse (state mutated)
    success, steps = check_success_with_steps(env, actions)
    return {
        "score": float(r_cont),
        "acc": 1.0 if success else 0.0,
        "success": 1.0 if success else 0.0,
        "steps": steps if success else -1,
        "format_ok": 1.0,
    }


def compute_score_binary(solution_str, ground_truth, data_source=None, extra_info=None):
    """Binary maze reward."""
    actions = parse_actions(solution_str)
    env = MazeEnv.from_sequence(ground_truth)
    if actions is None or env is None:
        return {"score": 0.0, "acc": 0.0, "success": 0.0, "steps": -1, "format_ok": 0.0}
    r_bin = binary_reward(actions, env)
    env = MazeEnv.from_sequence(ground_truth)
    success, steps = check_success_with_steps(env, actions)
    return {
        "score": float(r_bin),
        "acc": 1.0 if success else 0.0,
        "success": 1.0 if success else 0.0,
        "steps": steps if success else -1,
        "format_ok": 1.0,
    }
