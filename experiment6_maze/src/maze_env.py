"""Maze environment: generator, simulator, reward functions."""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ACTIONS: dict[str, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

ACTION_NAMES: dict[tuple[int, int], str] = {v: k for k, v in ACTIONS.items()}

# ---------------------------------------------------------------------------
# System prompt (used when formatting maze prompts for LLMs)
# ---------------------------------------------------------------------------

MAZE_SYSTEM_PROMPT = (
    "You are a maze-solving assistant. "
    "Given a maze grid, start position, and goal position, "
    "output the sequence of actions (UP, DOWN, LEFT, RIGHT) "
    "to navigate from start to goal, ending with DONE. "
    "Walls are marked as 1 and passages as 0. "
    "You cannot move through walls or outside the grid boundary. "
    "Find the shortest path if possible."
)

# ---------------------------------------------------------------------------
# MazeGenerator
# ---------------------------------------------------------------------------


class MazeGenerator:
    """Generate random mazes using Prim's algorithm."""

    def __init__(self, size: int = 17, seed: int = 0, algorithm: str = "prim"):
        if size < 5:
            raise ValueError(f"size must be >= 5, got {size}")
        if size % 2 == 0:
            raise ValueError(f"size must be odd, got {size}")
        self.size = size
        self.seed = seed
        self.algorithm = algorithm
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self._grid: Optional[np.ndarray] = None

    # -- generation ----------------------------------------------------------

    def generate(self) -> np.ndarray:
        """Generate a maze grid via Prim's algorithm.

        Returns an (size, size) int8 ndarray where 1 = wall, 0 = passage.
        """
        rng = np.random.RandomState(self.seed)
        size = self.size
        grid = np.ones((size, size), dtype=np.int8)  # all walls

        # Carve starting cell
        sr, sc = self.start
        grid[sr, sc] = 0

        # Frontier walls: walls that are 2 cells away from a carved passage
        frontier: list[tuple[int, int, int, int]] = []
        # (wall_r, wall_c, passage_r, passage_c) — the wall between current
        # carved cell and the unvisited cell 2 steps away.

        def _add_frontier(r: int, c: int) -> None:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + 2 * dr, c + 2 * dc
                if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr, nc] == 1:
                    frontier.append((r + dr, c + dc, nr, nc))

        _add_frontier(sr, sc)

        while frontier:
            idx = rng.randint(len(frontier))
            wall_r, wall_c, nr, nc = frontier[idx]
            # Swap-remove for O(1)
            frontier[idx] = frontier[-1]
            frontier.pop()

            if grid[nr, nc] == 1:
                grid[wall_r, wall_c] = 0
                grid[nr, nc] = 0
                _add_frontier(nr, nc)

        self._grid = grid
        return grid

    # -- solving -------------------------------------------------------------

    def solve_bfs(self) -> list[tuple[int, int]]:
        """BFS shortest path from start to goal.

        Returns list of (dr, dc) move tuples.
        """
        grid = self._grid
        if grid is None:
            grid = self.generate()

        start, goal = self.start, self.goal
        visited = set()
        visited.add(start)
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque()
        queue.append((start, []))

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.size
                    and 0 <= nc < self.size
                    and grid[nr, nc] == 0
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(dr, dc)]))

        raise RuntimeError("No path found from start to goal")

    def optimal_path_actions(self) -> list[str]:
        """BFS shortest path as list of action name strings."""
        moves = self.solve_bfs()
        return [ACTION_NAMES[m] for m in moves]

    # -- serialization -------------------------------------------------------

    def to_prompt_string(self) -> str:
        """Human-readable prompt representation of the maze."""
        grid = self._grid
        if grid is None:
            grid = self.generate()

        walls: list[tuple[int, int]] = []
        for r in range(self.size):
            for c in range(self.size):
                if grid[r, c] == 1:
                    walls.append((r, c))

        lines = [
            f"WALLS: {walls}",
            f"START: {self.start}",
            f"GOAL: {self.goal}",
            f"GRID_SIZE: {self.size}",
            f"STATE: {self.start}",
            "OUTPUT FORMAT: UP/DOWN/LEFT/RIGHT actions separated by spaces, ending with DONE",
        ]
        return "\n".join(lines)

    def to_text_sequence(self) -> str:
        """Serialize grid + optimal path as a ground-truth text sequence.

        Format:
            GRID_START
            <row0 as space-separated 0/1>
            ...
            GRID_END
            START <r> <c>
            GOAL <r> <c>
            SIZE <n>
            PATH_START
            <ACTION1>
            <ACTION2>
            ...
            PATH_END
        """
        grid = self._grid
        if grid is None:
            grid = self.generate()

        actions = self.optimal_path_actions()

        parts: list[str] = ["GRID_START"]
        for r in range(self.size):
            parts.append(" ".join(str(int(grid[r, c])) for c in range(self.size)))
        parts.append("GRID_END")
        parts.append(f"START {self.start[0]} {self.start[1]}")
        parts.append(f"GOAL {self.goal[0]} {self.goal[1]}")
        parts.append(f"SIZE {self.size}")
        parts.append("PATH_START")
        for a in actions:
            parts.append(a)
        parts.append("PATH_END")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# MazeEnv
# ---------------------------------------------------------------------------


class MazeEnv:
    """Lightweight maze simulator."""

    def __init__(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        size: int,
    ):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = size
        self.pos = start

    def reset(self) -> tuple[int, int]:
        self.pos = self.start
        return self.pos

    def step(self, action_str: str) -> tuple[tuple[int, int], bool, bool]:
        """Execute one action.

        Returns (new_pos, done, success).
        If the action would move into a wall or out of bounds, position is unchanged.
        """
        delta = ACTIONS.get(action_str)
        if delta is None:
            raise ValueError(f"Invalid action: {action_str}")

        dr, dc = delta
        nr, nc = self.pos[0] + dr, self.pos[1] + dc

        # Bounds check and wall check — stay in place if invalid
        if 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] == 0:
            self.pos = (nr, nc)

        done = self.pos == self.goal
        return self.pos, done, done

    def check_success(self, actions: list[str]) -> bool:
        """Run a sequence of actions from start and return whether the goal is reached."""
        self.reset()
        for a in actions:
            _, done, _ = self.step(a)
            if done:
                return True
        return False

    @classmethod
    def from_sequence(cls, text_sequence: str) -> Optional["MazeEnv"]:
        """Parse a ground-truth text sequence back into a MazeEnv.

        Returns None if the text is malformed or missing required fields.
        """
        try:
            lines = text_sequence.strip().split("\n")

            gs_idx = lines.index("GRID_START")
            ge_idx = lines.index("GRID_END")

            grid_lines = lines[gs_idx + 1 : ge_idx]
            grid = np.array(
                [[int(x) for x in row.split()] for row in grid_lines], dtype=np.int8
            )

            start: Optional[tuple[int, int]] = None
            goal: Optional[tuple[int, int]] = None
            size: Optional[int] = None

            for line in lines[ge_idx + 1 :]:
                if line.startswith("START "):
                    parts = line.split()
                    start = (int(parts[1]), int(parts[2]))
                elif line.startswith("GOAL "):
                    parts = line.split()
                    goal = (int(parts[1]), int(parts[2]))
                elif line.startswith("SIZE "):
                    size = int(line.split()[1])

            if start is None or goal is None or size is None:
                return None

            return cls(grid=grid, start=start, goal=goal, size=size)
        except (ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def parse_actions(solution_str: str) -> Optional[list[str]]:
    """Parse an LLM solution string into a list of action names.

    Expected format: "UP DOWN LEFT ... DONE"
    Returns actions *before* the first DONE.
    Returns None if:
      - no DONE token found
      - any token before DONE is not a valid action
      - input is empty
    """
    if not solution_str or not solution_str.strip():
        return None

    tokens = solution_str.strip().split()

    if "DONE" not in tokens:
        return None

    done_idx = tokens.index("DONE")
    action_tokens = tokens[:done_idx]

    for t in action_tokens:
        if t not in ACTIONS:
            return None

    return action_tokens


def check_success_with_steps(
    env: MazeEnv, actions: list[str]
) -> tuple[bool, int]:
    """Run actions and return (success, steps_taken).

    steps_taken is the 1-indexed position of the action that reached the goal.
    If the goal is never reached, steps_taken == len(actions).
    """
    env.reset()
    for i, a in enumerate(actions):
        _, done, _ = env.step(a)
        if done:
            return True, i + 1
    return False, len(actions)


def continuous_reward(actions: list[str], env: MazeEnv, l_max: int) -> float:
    """Continuous reward: 1 - steps/l_max for success, 0 otherwise.

    Returns 0 if:
      - actions is empty
      - len(actions) > l_max
      - the path does not reach the goal
    """
    if len(actions) == 0:
        return 0.0
    if len(actions) > l_max:
        return 0.0

    success, steps_taken = check_success_with_steps(env, actions)
    if not success:
        return 0.0

    return 1.0 - steps_taken / l_max


def binary_reward(actions: list[str], env: MazeEnv) -> float:
    """Binary reward: 1.0 if the path reaches the goal, 0.0 otherwise."""
    success, _ = check_success_with_steps(env, actions)
    return 1.0 if success else 0.0
