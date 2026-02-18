from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_K_VALUES = (1, 2, 5, 10, 20, 50, 100)


@dataclass
class ProblemEvaluation:
    problem_id: str
    binary_passes: np.ndarray  # shape [n_samples], dtype=bool
    continuous_rewards: np.ndarray | None = None


@dataclass
class RunSummary:
    model: str
    method: str
    seed: int
    group_size: int | None
    n_samples: int
    num_problems: int
    expected_reward: float | None
    pass_at_k: dict[str, float]
    best_of_n: float


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator used in code-generation benchmarks."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    c = max(0, min(c, n))
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k_curve(
    binary_passes_matrix: np.ndarray,
    k_values: Iterable[int],
) -> dict[str, float]:
    if binary_passes_matrix.ndim != 2:
        raise ValueError("binary_passes_matrix must have shape [num_problems, n_samples]")

    num_problems, n_samples = binary_passes_matrix.shape
    if num_problems == 0:
        return {str(k): 0.0 for k in k_values}

    counts = binary_passes_matrix.sum(axis=1)
    curve: dict[str, float] = {}
    for k in k_values:
        k_int = int(k)
        if k_int > n_samples:
            continue
        values = [pass_at_k(n_samples, int(c), k_int) for c in counts]
        curve[str(k_int)] = float(np.mean(values))
    return curve


def summarize_problem_rows(
    rows: list[dict[str, Any]],
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> RunSummary:
    if not rows:
        raise ValueError("No rows provided")

    model = str(rows[0].get("model", "unknown"))
    method = str(rows[0].get("method", "unknown"))
    seed = int(rows[0].get("seed", 0))
    group_sizes = {
        int(row["group_size"])
        for row in rows
        if row.get("group_size") is not None
    }
    if len(group_sizes) > 1:
        raise ValueError(
            "Inconsistent group_size values inside one run: "
            f"{sorted(group_sizes)}"
        )
    group_size = next(iter(group_sizes)) if group_sizes else None

    binary_vectors = []
    rewards_means = []

    for row in rows:
        passes = np.asarray(row["binary_passes"], dtype=bool)
        if passes.ndim != 1:
            raise ValueError("binary_passes must be 1D per row")
        binary_vectors.append(passes)

        if "continuous_rewards" in row and row["continuous_rewards"] is not None:
            rewards = np.asarray(row["continuous_rewards"], dtype=float)
            if rewards.shape != passes.shape:
                raise ValueError("continuous_rewards must match binary_passes length")
            rewards_means.append(float(np.mean(rewards)))

    n_samples_set = {vec.shape[0] for vec in binary_vectors}
    if len(n_samples_set) != 1:
        raise ValueError("Inconsistent number of samples across problems in a run")

    n_samples = next(iter(n_samples_set))
    matrix = np.stack(binary_vectors, axis=0)
    curve = compute_pass_at_k_curve(matrix, k_values=k_values)
    counts = matrix.sum(axis=1)
    best_of_n = float(np.mean([pass_at_k(n_samples, int(c), n_samples) for c in counts]))

    expected_reward = float(np.mean(rewards_means)) if rewards_means else None

    return RunSummary(
        model=model,
        method=method,
        seed=seed,
        group_size=group_size,
        n_samples=n_samples,
        num_problems=matrix.shape[0],
        expected_reward=expected_reward,
        pass_at_k=curve,
        best_of_n=best_of_n,
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_runs(rows: list[dict[str, Any]], k_values: tuple[int, ...]) -> list[RunSummary]:
    grouped: dict[tuple[str, str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_size = row.get("group_size")
        group_size_key = int(group_size) if group_size is not None else None
        key = (
            str(row.get("model", "unknown")),
            str(row.get("method", "unknown")),
            int(row.get("seed", 0)),
            group_size_key,
        )
        grouped[key].append(row)

    summaries: list[RunSummary] = []
    for group_rows in grouped.values():
        summaries.append(summarize_problem_rows(group_rows, k_values=k_values))

    summaries.sort(
        key=lambda s: (
            s.model,
            s.group_size if s.group_size is not None else -1,
            s.method,
            s.seed,
        )
    )
    return summaries


def to_serializable(summaries: list[RunSummary]) -> dict[str, Any]:
    rows = []
    for item in summaries:
        rows.append(
            {
                "model": item.model,
                "method": item.method,
                "seed": item.seed,
                "group_size": item.group_size,
                "n_samples": item.n_samples,
                "num_problems": item.num_problems,
                "expected_reward": item.expected_reward,
                "pass_at_k": item.pass_at_k,
                "best_of_n": item.best_of_n,
            }
        )
    return {"runs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pass@k from completion-level records.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help=(
            "JSONL with one row per problem containing: model, method, seed, "
            "binary_passes (list of 0/1), and optional continuous_rewards"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path for aggregated run summaries",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="k values for pass@k computation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    summaries = aggregate_runs(rows, k_values=tuple(args.k_values))

    payload = to_serializable(summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {len(summaries)} run summaries to {args.output_json}")


if __name__ == "__main__":
    main()
