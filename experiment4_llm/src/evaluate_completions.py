from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluate import aggregate_runs, to_serializable
from .reward_function import RewardConfig, compute_reward


@dataclass(frozen=True)
class CompletionEvalConfig:
    timeout_seconds: int = 5
    memory_limit_mb: int = 256
    pass_tolerance: float = 1e-12


def load_jsonl_rows(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _normalize_group_size(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _evaluate_completion_row(
    row: dict[str, Any],
    config: CompletionEvalConfig,
) -> dict[str, Any]:
    if "generated_code" not in row:
        raise ValueError("Each completion row must include generated_code")
    if "test_cases" not in row:
        raise ValueError("Each completion row must include test_cases")

    reward = float(
        compute_reward(
            generated_code=str(row["generated_code"]),
            test_cases=row["test_cases"],
            config=RewardConfig(
                timeout_seconds=config.timeout_seconds,
                memory_limit_mb=config.memory_limit_mb,
            ),
        )
    )

    binary_pass = int(reward >= (1.0 - config.pass_tolerance))
    return {
        "model": str(row.get("model", "unknown")),
        "method": str(row.get("method", "unknown")),
        "seed": int(row.get("seed", 0)),
        "group_size": _normalize_group_size(row.get("group_size")),
        "problem_id": str(row.get("problem_id", "")),
        "binary_pass": binary_pass,
        "continuous_reward": reward,
    }


def evaluate_completion_rows(
    rows: list[dict[str, Any]],
    config: CompletionEvalConfig | None = None,
) -> list[dict[str, Any]]:
    if config is None:
        config = CompletionEvalConfig()

    grouped: dict[tuple[str, str, int, int | None, str], dict[str, Any]] = defaultdict(
        lambda: {
            "model": "unknown",
            "method": "unknown",
            "seed": 0,
            "group_size": None,
            "problem_id": "",
            "binary_passes": [],
            "continuous_rewards": [],
        }
    )

    for row in rows:
        evaluated = _evaluate_completion_row(row, config=config)
        key = (
            evaluated["model"],
            evaluated["method"],
            evaluated["seed"],
            evaluated["group_size"],
            evaluated["problem_id"],
        )
        agg = grouped[key]
        agg["model"] = evaluated["model"]
        agg["method"] = evaluated["method"]
        agg["seed"] = evaluated["seed"]
        agg["group_size"] = evaluated["group_size"]
        agg["problem_id"] = evaluated["problem_id"]
        agg["binary_passes"].append(evaluated["binary_pass"])
        agg["continuous_rewards"].append(evaluated["continuous_reward"])

    out = list(grouped.values())
    out.sort(
        key=lambda row: (
            row["model"],
            row["group_size"] if row["group_size"] is not None else -1,
            row["method"],
            row["seed"],
            row["problem_id"],
        )
    )
    return out


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw completion-level code samples and aggregate them "
            "into problem-level rows compatible with experiment4_llm.src.evaluate"
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    parser.add_argument("--timeout-seconds", type=int, default=5)
    parser.add_argument("--memory-limit-mb", type=int, default=256)
    parser.add_argument("--pass-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = load_jsonl_rows(args.input_jsonl, max_rows=args.max_rows)
    grouped_rows = evaluate_completion_rows(
        raw_rows,
        config=CompletionEvalConfig(
            timeout_seconds=args.timeout_seconds,
            memory_limit_mb=args.memory_limit_mb,
            pass_tolerance=args.pass_tolerance,
        ),
    )
    write_jsonl(grouped_rows, args.output_jsonl)
    print(f"Wrote {len(grouped_rows)} problem-level rows to {args.output_jsonl}")

    if args.summary_json is not None:
        summaries = aggregate_runs(grouped_rows, k_values=tuple(args.k_values))
        payload = to_serializable(summaries)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {len(summaries)} run summaries to {args.summary_json}")


if __name__ == "__main__":
    main()
