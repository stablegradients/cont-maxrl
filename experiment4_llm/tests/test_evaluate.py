from __future__ import annotations

import unittest

import numpy as np

from experiment4_llm.src.evaluate import aggregate_runs, compute_pass_at_k_curve, pass_at_k


class EvaluateTests(unittest.TestCase):
    def test_pass_at_k_edge_cases(self) -> None:
        self.assertEqual(pass_at_k(n=10, c=0, k=1), 0.0)
        self.assertEqual(pass_at_k(n=10, c=10, k=1), 1.0)
        self.assertEqual(pass_at_k(n=10, c=1, k=10), 1.0)

    def test_pass_at_k_curve_macro_average(self) -> None:
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=bool,
        )
        curve = compute_pass_at_k_curve(matrix, k_values=[1, 2, 4])
        self.assertAlmostEqual(curve["1"], 0.125, places=6)
        self.assertAlmostEqual(curve["4"], 0.5, places=6)

    def test_aggregate_runs_keeps_group_size_separate(self) -> None:
        rows = [
            {
                "model": "qwen_1.5b",
                "method": "grpo",
                "seed": 42,
                "group_size": 4,
                "problem_id": "p1",
                "binary_passes": [1, 0, 0, 0],
                "continuous_rewards": [1.0, 0.2, 0.3, 0.1],
            },
            {
                "model": "qwen_1.5b",
                "method": "grpo",
                "seed": 42,
                "group_size": 16,
                "problem_id": "p1",
                "binary_passes": [1, 1, 0, 0],
                "continuous_rewards": [1.0, 1.0, 0.2, 0.1],
            },
        ]
        summaries = aggregate_runs(rows, k_values=(1, 2, 4))
        self.assertEqual(len(summaries), 2)
        self.assertEqual({s.group_size for s in summaries}, {4, 16})
        for summary in summaries:
            self.assertIn("1", summary.pass_at_k)
            self.assertGreaterEqual(summary.best_of_n, 0.0)
            self.assertLessEqual(summary.best_of_n, 1.0)


if __name__ == "__main__":
    unittest.main()
