"""Tests for verl-compatible maze reward functions."""
from __future__ import annotations

import pytest

from experiment6_maze.src.maze_env import MazeGenerator
from experiment6_maze.src.reward_function import compute_score, compute_score_binary

REQUIRED_KEYS = {"score", "acc", "success", "steps", "format_ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ground_truth(size: int, seed: int = 0) -> tuple[str, list[str], int]:
    """Return (ground_truth_text, optimal_actions, optimal_steps)."""
    gen = MazeGenerator(size=size, seed=seed)
    gen.generate()
    actions = gen.optimal_path_actions()
    gt = gen.to_text_sequence()
    return gt, actions, len(actions)


def _solution_str(actions: list[str]) -> str:
    return " ".join(actions) + " DONE"


# ---------------------------------------------------------------------------
# compute_score (continuous)
# ---------------------------------------------------------------------------


class TestComputeScore:
    def test_returns_dict_with_required_keys(self):
        gt, actions, n_steps = _make_ground_truth(size=17, seed=42)
        result = compute_score(_solution_str(actions), gt)
        assert isinstance(result, dict)
        assert set(result.keys()) == REQUIRED_KEYS
        assert result["score"] > 0.0
        assert result["success"] == 1.0
        assert result["acc"] == 1.0
        assert result["steps"] == n_steps
        assert result["format_ok"] == 1.0

    def test_empty_solution(self):
        gt, _, _ = _make_ground_truth(size=17, seed=42)
        result = compute_score("", gt)
        assert result["score"] == 0.0

    def test_no_done(self):
        gt, actions, _ = _make_ground_truth(size=17, seed=42)
        # Valid actions but missing DONE token
        result = compute_score(" ".join(actions), gt)
        assert result["score"] == 0.0

    def test_invalid_action(self):
        gt, _, _ = _make_ground_truth(size=17, seed=42)
        result = compute_score("JUMP DONE", gt)
        assert result["score"] == 0.0

    def test_l_max_propagates(self):
        gt, actions, n_steps = _make_ground_truth(size=17, seed=42)
        sol = _solution_str(actions)
        r_32 = compute_score(sol, gt, extra_info={"l_max": 32})
        r_64 = compute_score(sol, gt, extra_info={"l_max": 64})
        r_128 = compute_score(sol, gt, extra_info={"l_max": 128})
        # For a successful path: score = 1 - steps/l_max
        # Larger l_max => higher score (less penalty per step)
        # Only compare if the path length fits within l_max
        # n_steps for size=17 is typically > 32, so r_32 may be 0
        # Pick l_max values that are all >= n_steps
        if n_steps <= 32:
            assert r_32["score"] < r_64["score"] < r_128["score"]
        else:
            # r_32 is 0 (path too long), r_64 and r_128 should differ
            assert r_32["score"] <= r_64["score"] <= r_128["score"]
            # At least the larger two should differ
            assert r_64["score"] < r_128["score"] or r_32["score"] < r_64["score"]

    def test_default_l_max(self):
        """None extra_info defaults to l_max=64."""
        gt, actions, _ = _make_ground_truth(size=17, seed=42)
        sol = _solution_str(actions)
        r_none = compute_score(sol, gt, extra_info=None)
        r_64 = compute_score(sol, gt, extra_info={"l_max": 64})
        assert r_none["score"] == r_64["score"]


# ---------------------------------------------------------------------------
# compute_score_binary
# ---------------------------------------------------------------------------


class TestComputeScoreBinary:
    def test_binary_success(self):
        gt, actions, _ = _make_ground_truth(size=17, seed=42)
        result = compute_score_binary(_solution_str(actions), gt)
        assert result["score"] == 1.0
        assert result["success"] == 1.0

    def test_binary_failure(self):
        gt, _, _ = _make_ground_truth(size=17, seed=42)
        # Move that won't reach goal
        result = compute_score_binary("UP DONE", gt)
        assert result["score"] == 0.0
        assert result["success"] == 0.0


# ---------------------------------------------------------------------------
# Cross-check: binary vs continuous
# ---------------------------------------------------------------------------


class TestBinaryMatchesContinuousThreshold:
    def test_binary_matches_continuous_threshold(self):
        """Over 100 random mazes: r_bin==1.0 iff r_cont>0."""
        for seed in range(100):
            gt, actions, n_steps = _make_ground_truth(size=5, seed=seed)

            # Successful path
            sol_ok = _solution_str(actions)
            r_cont = compute_score(sol_ok, gt, extra_info={"l_max": 128})
            r_bin = compute_score_binary(sol_ok, gt)
            assert (r_bin["score"] == 1.0) == (r_cont["score"] > 0), (
                f"seed={seed}: r_bin={r_bin['score']}, r_cont={r_cont['score']}"
            )

            # Failed path (empty actions, just DONE)
            sol_bad = "DONE"
            r_cont_bad = compute_score(sol_bad, gt, extra_info={"l_max": 128})
            r_bin_bad = compute_score_binary(sol_bad, gt)
            assert (r_bin_bad["score"] == 1.0) == (r_cont_bad["score"] > 0), (
                f"seed={seed} (bad): r_bin={r_bin_bad['score']}, r_cont={r_cont_bad['score']}"
            )
