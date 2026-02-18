#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiment4_llm.src.advantages import compute_grouped_advantages
from experiment4_llm.src.custom_adv_estimator import compute_token_advantages
from experiment4_llm.src.data_prep import filter_apps_records, write_jsonl
from experiment4_llm.src.evaluate import aggregate_runs, to_serializable
from experiment4_llm.src.reward_function import RewardConfig, compute_reward


def _build_smoke_records() -> list[dict]:
    return [
        {
            "problem_id": "smoke_fn_1",
            "difficulty": "introductory",
            "question": "Implement double(x).",
            "starter_code": "def double(x):\n    pass",
            "input_output": json.dumps(
                {
                    "inputs": [[1], [2], [3], [4], [5]],
                    "outputs": ["2", "4", "6", "8", "10"],
                    "fn_name": "double",
                }
            ),
        },
        {
            "problem_id": "smoke_io_1",
            "difficulty": "interview",
            "question": "Read int n and print n*3.",
            "starter_code": "",
            "input_output": json.dumps(
                {
                    "inputs": ["1\n", "2\n", "3\n", "4\n", "5\n"],
                    "outputs": ["3", "6", "9", "12", "15"],
                }
            ),
        },
    ]


def run_data_prep_smoke(output_dir: Path) -> None:
    records = _build_smoke_records()
    converted = filter_apps_records(
        records,
        difficulties=("introductory", "interview"),
        min_test_cases=5,
    )
    assert len(converted) == 2, f"expected 2 converted records, got {len(converted)}"

    out_path = output_dir / "apps_smoke.jsonl"
    write_jsonl(converted, out_path)
    assert out_path.exists(), "APPS smoke JSONL not written"


def run_reward_smoke() -> None:
    code_fn = """
def double(x):
    if x == 5:
        return 999
    return 2 * x
"""
    tests_fn = {
        "inputs": [[1], [2], [3], [4], [5]],
        "outputs": ["2", "4", "6", "8", "10"],
        "fn_name": "double",
    }
    reward_fn = compute_reward(code_fn, tests_fn, config=RewardConfig(timeout_seconds=2, memory_limit_mb=128))
    assert abs(reward_fn - 0.8) < 1e-8, f"expected 0.8 reward, got {reward_fn}"

    code_io = """
import sys
text = sys.stdin.read().strip()
if text:
    print(int(text) * 3)
"""
    tests_io = {
        "inputs": ["1\n", "2\n", "3\n", "4\n", "5\n"],
        "outputs": ["3", "6", "9", "12", "15"],
    }
    reward_io = compute_reward(code_io, tests_io, config=RewardConfig(timeout_seconds=2, memory_limit_mb=128))
    assert abs(reward_io - 1.0) < 1e-8, f"expected 1.0 reward, got {reward_io}"


def run_advantage_smoke() -> None:
    rewards = torch.tensor(
        [
            [0.1, 0.2, 0.3, 1.0],
            [0.0, 0.4, 0.4, 0.9],
        ],
        dtype=torch.float64,
    )
    for method in ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        adv = compute_grouped_advantages(rewards, method=method)
        assert adv.shape == rewards.shape
        row_sums = adv.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-8)

    token_rewards = torch.zeros((8, 6), dtype=torch.float32)
    eos_mask = torch.zeros((8, 6), dtype=torch.float32)
    eos_mask[:, -1] = 1
    token_rewards[:, -1] = torch.tensor([0.0, 0.2, 0.6, 1.0, 0.1, 0.1, 0.3, 0.9], dtype=torch.float32)
    token_adv = compute_token_advantages(
        token_level_rewards=token_rewards,
        eos_mask=eos_mask,
        method="maxrl_unnorm",
        group_size=4,
    )
    assert token_adv.shape == token_rewards.shape


def run_eval_smoke(output_dir: Path) -> Path:
    rng = np.random.default_rng(42)

    methods = ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]
    rows = []
    group_sizes = [4, 16]
    for group_idx, group_size in enumerate(group_sizes):
        for method_idx, method in enumerate(methods):
            for problem_idx in range(5):
                n = 20
                base = 0.15 + 0.03 * method_idx + 0.02 * group_idx + 0.01 * problem_idx
                probs = np.clip(base + 0.02 * rng.normal(size=n), 0.01, 0.95)
                binary = (rng.random(n) < probs).astype(int).tolist()
                cont = np.clip(probs + 0.05 * rng.normal(size=n), 0.0, 1.0).tolist()
                rows.append(
                    {
                        "model": "qwen_1.5b_smoke",
                        "method": method,
                        "seed": 42,
                        "group_size": group_size,
                        "problem_id": f"p{problem_idx}",
                        "binary_passes": binary,
                        "continuous_rewards": cont,
                    }
                )

    input_jsonl = output_dir / "completions_smoke.jsonl"
    with input_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summaries = aggregate_runs(rows, k_values=(1, 2, 5, 10, 20))
    payload = to_serializable(summaries)
    summary_json = output_dir / "metrics_summary_smoke.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    # Exercise figure script CLI as well.
    fig_script = Path(__file__).resolve().parent / "generate_figures.py"
    subprocess.check_call(
        [
            sys.executable,
            str(fig_script),
            "--metrics-summary",
            str(summary_json),
            "--problem-results-jsonl",
            str(input_jsonl),
            "--output-dir",
            str(output_dir),
        ]
    )

    return summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast smoke checks for Experiment 4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_experiment4/smoke"),
        help="Output directory for smoke artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_data_prep_smoke(args.output_dir)
    run_reward_smoke()
    run_advantage_smoke()
    summary = run_eval_smoke(args.output_dir)

    print("Smoke run completed successfully")
    print(f"Artifacts: {args.output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
