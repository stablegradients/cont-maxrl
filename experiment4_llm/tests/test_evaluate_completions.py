from __future__ import annotations

import unittest

from experiment4_llm.src.evaluate_completions import (
    CompletionEvalConfig,
    evaluate_completion_rows,
)


class EvaluateCompletionsTests(unittest.TestCase):
    def test_evaluate_completion_rows_function_and_stdin(self) -> None:
        rows = [
            {
                "model": "qwen_1.5b",
                "method": "grpo",
                "seed": 42,
                "group_size": 16,
                "problem_id": "p_fn",
                "generated_code": "def add(a, b):\n    return a + b\n",
                "test_cases": {
                    "inputs": [[1, 2], [2, 5], [10, 1]],
                    "outputs": ["3", "7", "11"],
                    "fn_name": "add",
                },
            },
            {
                "model": "qwen_1.5b",
                "method": "grpo",
                "seed": 42,
                "group_size": 16,
                "problem_id": "p_fn",
                "generated_code": "def add(a, b):\n    return a - b\n",
                "test_cases": {
                    "inputs": [[1, 2], [2, 5], [10, 1]],
                    "outputs": ["3", "7", "11"],
                    "fn_name": "add",
                },
            },
            {
                "model": "qwen_1.5b",
                "method": "grpo",
                "seed": 42,
                "group_size": 16,
                "problem_id": "p_io",
                "generated_code": "import sys\nx = int(sys.stdin.read().strip())\nprint(x * 2)\n",
                "test_cases": {
                    "inputs": ["1\n", "2\n", "3\n"],
                    "outputs": ["2", "4", "6"],
                },
            },
        ]

        grouped = evaluate_completion_rows(
            rows,
            config=CompletionEvalConfig(timeout_seconds=2, memory_limit_mb=128),
        )
        self.assertEqual(len(grouped), 2)

        by_problem = {row["problem_id"]: row for row in grouped}
        self.assertEqual(by_problem["p_fn"]["binary_passes"], [1, 0])
        self.assertEqual(len(by_problem["p_fn"]["continuous_rewards"]), 2)
        self.assertEqual(by_problem["p_io"]["binary_passes"], [1])
        self.assertAlmostEqual(by_problem["p_io"]["continuous_rewards"][0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
