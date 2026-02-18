from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GenerateFiguresTests(unittest.TestCase):
    def test_generates_fig_b_when_training_metrics_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp4_fig_test_") as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "metrics_summary.json"
            problem_rows_path = root / "problem_rows.jsonl"
            training_path = root / "training_metrics.jsonl"
            output_dir = root / "figs"

            summary_payload = {
                "runs": [
                    {
                        "model": "qwen_test",
                        "method": "grpo",
                        "seed": 42,
                        "group_size": 16,
                        "n_samples": 20,
                        "num_problems": 2,
                        "expected_reward": 0.5,
                        "pass_at_k": {"1": 0.1, "2": 0.2, "5": 0.35, "10": 0.5},
                        "best_of_n": 0.75,
                    }
                ]
            }
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

            problem_rows = [
                {
                    "model": "qwen_test",
                    "method": "grpo",
                    "seed": 42,
                    "group_size": 16,
                    "problem_id": "p0",
                    "binary_passes": [0, 1],
                    "continuous_rewards": [0.0, 1.0],
                }
            ]
            with problem_rows_path.open("w", encoding="utf-8") as handle:
                for row in problem_rows:
                    handle.write(json.dumps(row) + "\n")

            training_rows = [
                {
                    "model": "qwen_test",
                    "method": "grpo",
                    "seed": 42,
                    "group_size": 16,
                    "global_step": 10,
                    "pass_at_k": {"1": 0.05, "10": 0.2},
                },
                {
                    "model": "qwen_test",
                    "method": "grpo",
                    "seed": 42,
                    "group_size": 16,
                    "global_step": 20,
                    "pass_at_k": {"1": 0.08, "10": 0.25},
                },
            ]
            with training_path.open("w", encoding="utf-8") as handle:
                for row in training_rows:
                    handle.write(json.dumps(row) + "\n")

            script_path = (
                Path(__file__).resolve().parent.parent / "scripts" / "generate_figures.py"
            )
            subprocess.check_call(
                [
                    sys.executable,
                    str(script_path),
                    "--metrics-summary",
                    str(summary_path),
                    "--problem-results-jsonl",
                    str(problem_rows_path),
                    "--training-metrics-jsonl",
                    str(training_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertTrue((output_dir / "figA_pass_at_k_qwen_test_g16.png").exists())
            self.assertTrue((output_dir / "figB_training_curves_qwen_test_g16.png").exists())


if __name__ == "__main__":
    unittest.main()
