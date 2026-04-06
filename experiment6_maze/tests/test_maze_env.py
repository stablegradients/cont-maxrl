"""Comprehensive tests for maze_env module."""

import numpy as np
import pytest

from experiment6_maze.src.maze_env import (
    ACTIONS,
    ACTION_NAMES,
    MazeEnv,
    MazeGenerator,
    binary_reward,
    check_success_with_steps,
    continuous_reward,
    parse_actions,
)


# =========================================================================
# TestMazeGenerator
# =========================================================================


class TestMazeGenerator:
    """Tests for MazeGenerator."""

    def test_generate_shape(self):
        gen = MazeGenerator(size=17, seed=42)
        grid = gen.generate()
        assert grid.shape == (17, 17)
        assert grid.dtype == np.int8

    def test_start_goal_are_passages(self):
        gen = MazeGenerator(size=17, seed=42)
        grid = gen.generate()
        assert grid[1, 1] == 0, "Start cell should be a passage"
        assert grid[15, 15] == 0, "Goal cell should be a passage"

    def test_border_is_walls(self):
        gen = MazeGenerator(size=17, seed=42)
        grid = gen.generate()
        # Top and bottom rows
        assert np.all(grid[0, :] == 1), "Top border should be all walls"
        assert np.all(grid[-1, :] == 1), "Bottom border should be all walls"
        # Left and right columns
        assert np.all(grid[:, 0] == 1), "Left border should be all walls"
        assert np.all(grid[:, -1] == 1), "Right border should be all walls"

    def test_bfs_solution_exists(self):
        gen = MazeGenerator(size=17, seed=42)
        gen.generate()
        moves = gen.solve_bfs()
        assert len(moves) > 0, "BFS should find a non-empty path"

    def test_1000_seeds_solvable(self):
        """Every maze with seeds 0..999 should be solvable."""
        for seed in range(1000):
            gen = MazeGenerator(size=9, seed=seed)
            gen.generate()
            try:
                moves = gen.solve_bfs()
                assert len(moves) > 0
            except RuntimeError:
                pytest.fail(f"Maze with seed={seed} is not solvable")

    def test_10k_seeds_distinct(self):
        """10,000 different seeds should produce distinct grids."""
        grids = set()
        for seed in range(10_000):
            gen = MazeGenerator(size=17, seed=seed)
            grid = gen.generate()
            grids.add(grid.tobytes())
        # Larger grids have much more combinatorial space
        assert len(grids) > 9_950, (
            f"Expected >9950 distinct grids from 10K seeds, got {len(grids)}"
        )

    def test_deterministic_generation(self):
        """Same seed produces identical grids."""
        g1 = MazeGenerator(size=17, seed=123).generate()
        g2 = MazeGenerator(size=17, seed=123).generate()
        assert np.array_equal(g1, g2)

    def test_optimal_path_actions(self):
        gen = MazeGenerator(size=9, seed=0)
        gen.generate()
        actions = gen.optimal_path_actions()
        assert isinstance(actions, list)
        assert all(a in ACTIONS for a in actions)
        # Verify the actions actually reach the goal
        env = MazeEnv(grid=gen._grid, start=gen.start, goal=gen.goal, size=gen.size)
        assert env.check_success(actions)

    def test_to_text_sequence_roundtrip(self):
        """to_text_sequence -> from_sequence should preserve grid, start, goal, size."""
        gen = MazeGenerator(size=11, seed=77)
        gen.generate()
        text = gen.to_text_sequence()

        env = MazeEnv.from_sequence(text)
        assert np.array_equal(env.grid, gen._grid)
        assert env.start == gen.start
        assert env.goal == gen.goal
        assert env.size == gen.size

    def test_to_prompt_string_fields(self):
        gen = MazeGenerator(size=9, seed=0)
        gen.generate()
        prompt = gen.to_prompt_string()
        # Compact grid format: # . S G, space-separated, newline-delimited
        assert "S" in prompt, "start marker missing"
        assert "G" in prompt, "goal marker missing"
        assert "#" in prompt, "wall marker missing"
        assert "." in prompt, "path marker missing"
        lines = prompt.strip().split("\n")
        assert len(lines) == 9, f"expected 9 rows, got {len(lines)}"
        for line in lines:
            cells = line.split(" ")
            assert len(cells) == 9, f"expected 9 cols, got {len(cells)}"

    def test_size_too_small(self):
        with pytest.raises(ValueError, match="size must be >= 5"):
            MazeGenerator(size=3)

    def test_size_even(self):
        with pytest.raises(ValueError, match="size must be odd"):
            MazeGenerator(size=6)


# =========================================================================
# TestMazeEnv
# =========================================================================


class TestMazeEnv:
    """Tests for MazeEnv."""

    def _make_env(self, size=9, seed=0):
        gen = MazeGenerator(size=size, seed=seed)
        grid = gen.generate()
        return MazeEnv(grid=grid, start=gen.start, goal=gen.goal, size=size), gen

    def test_check_success_with_steps_optimal(self):
        """Optimal path should succeed and steps == path length."""
        env, gen = self._make_env()
        actions = gen.optimal_path_actions()
        success, steps = check_success_with_steps(env, actions)
        assert success
        assert steps == len(actions)

    def test_early_termination(self):
        """Extra actions after reaching the goal should not matter;
        steps_taken should be the index+1 of the goal-reaching action."""
        env, gen = self._make_env()
        actions = gen.optimal_path_actions()
        # Append extra actions after goal is reached
        extended = actions + ["UP", "DOWN", "LEFT", "RIGHT"]
        success, steps = check_success_with_steps(env, extended)
        assert success
        assert steps == len(actions), (
            f"steps_taken should be {len(actions)}, got {steps}"
        )

    def test_wall_collision(self):
        """Moving into a wall should leave position unchanged."""
        env, _ = self._make_env(size=9, seed=0)
        env.reset()
        # Try moving UP from (1,1) — that's into row 0 which is a wall
        pos_before = env.pos
        new_pos, done, success = env.step("UP")
        assert new_pos == pos_before, "Should stay in place when hitting a wall"
        assert not done

    def test_from_sequence_invalid(self):
        """from_sequence should return None on invalid input."""
        assert MazeEnv.from_sequence("GARBAGE INPUT") is None
        assert MazeEnv.from_sequence("GRID_START\n1 0\nGRID_END\n") is None
        assert MazeEnv.from_sequence("") is None

    def test_reset(self):
        env, gen = self._make_env()
        actions = gen.optimal_path_actions()
        env.check_success(actions)
        # After check_success the position may be at goal; reset should go back
        pos = env.reset()
        assert pos == env.start


# =========================================================================
# TestParseActions
# =========================================================================


class TestParseActions:
    """Tests for parse_actions."""

    def test_valid(self):
        result = parse_actions("UP DOWN LEFT RIGHT DONE")
        assert result == ["UP", "DOWN", "LEFT", "RIGHT"]

    def test_empty_before_done(self):
        result = parse_actions("DONE")
        assert result == []

    def test_extra_after_done(self):
        """Only actions before the first DONE are returned."""
        result = parse_actions("UP DONE DOWN")
        assert result == ["UP"]

    def test_no_done(self):
        result = parse_actions("UP DOWN LEFT")
        assert result is None

    def test_invalid_token(self):
        result = parse_actions("UP JUMP DOWN DONE")
        assert result is None

    def test_empty_string(self):
        result = parse_actions("")
        assert result is None

    def test_whitespace_only(self):
        result = parse_actions("   ")
        assert result is None


# =========================================================================
# TestContinuousReward
# =========================================================================


class TestContinuousReward:
    """Tests for continuous_reward."""

    def _make_env(self, size=9, seed=0):
        gen = MazeGenerator(size=size, seed=seed)
        grid = gen.generate()
        return MazeEnv(grid=grid, start=gen.start, goal=gen.goal, size=size), gen

    def test_exact_value(self):
        """Reward for the optimal path with l_max = 2 * len(path)."""
        env, gen = self._make_env()
        actions = gen.optimal_path_actions()
        n = len(actions)
        l_max = 2 * n
        reward = continuous_reward(actions, env, l_max)
        expected = 1.0 - n / l_max  # = 0.5
        assert abs(reward - expected) < 1e-9

    def test_zero_for_failure(self):
        """Wrong actions that don't reach the goal should give 0."""
        env, _ = self._make_env()
        actions = ["UP"] * 5  # unlikely to reach goal
        reward = continuous_reward(actions, env, l_max=100)
        assert reward == 0.0

    def test_zero_exceeding_lmax(self):
        """If len(actions) > l_max, reward is 0."""
        env, gen = self._make_env()
        actions = gen.optimal_path_actions()
        reward = continuous_reward(actions, env, l_max=1)
        assert reward == 0.0

    def test_zero_empty_actions(self):
        env, _ = self._make_env()
        reward = continuous_reward([], env, l_max=100)
        assert reward == 0.0


# =========================================================================
# TestBinaryReward
# =========================================================================


class TestBinaryReward:
    """Tests for binary_reward."""

    def _make_env(self, size=9, seed=0):
        gen = MazeGenerator(size=size, seed=seed)
        grid = gen.generate()
        return MazeEnv(grid=grid, start=gen.start, goal=gen.goal, size=size), gen

    def test_success(self):
        env, gen = self._make_env()
        actions = gen.optimal_path_actions()
        assert binary_reward(actions, env) == 1.0

    def test_failure(self):
        env, _ = self._make_env()
        assert binary_reward(["UP"] * 3, env) == 0.0

    def test_collapses_continuous(self):
        """For 100 random seeds, binary_reward should equal
        1.0 whenever continuous_reward > 0, and 0.0 otherwise."""
        for seed in range(100):
            env, gen = self._make_env(size=9, seed=seed)
            actions = gen.optimal_path_actions()
            l_max = 2 * len(actions)

            cr = continuous_reward(actions, env, l_max)
            br = binary_reward(actions, env)

            if cr > 0:
                assert br == 1.0, f"seed={seed}: continuous>0 but binary!=1"
            # Also test failure case
            env2, _ = self._make_env(size=9, seed=seed)
            br_fail = binary_reward(["UP"] * 3, env2)
            cr_fail = continuous_reward(["UP"] * 3, env2, l_max=100)
            if cr_fail == 0.0:
                assert br_fail == 0.0, f"seed={seed}: continuous==0 but binary!=0"
