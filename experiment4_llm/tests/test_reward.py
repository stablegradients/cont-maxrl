from __future__ import annotations

import json
import unittest

from experiment4_llm.src.reward_function import RewardConfig, compute_reward, parse_test_cases


class RewardFunctionTests(unittest.TestCase):
    def test_parse_test_cases_from_json(self) -> None:
        raw = json.dumps(
            {
                "inputs": [[1], [2], [3]],
                "outputs": ["2", "4", "6"],
                "fn_name": "double",
            }
        )
        parsed = parse_test_cases(raw)
        self.assertEqual(parsed.fn_name, "double")
        self.assertEqual(len(parsed.inputs), 3)

    def test_function_mode_continuous_reward(self) -> None:
        code = """
def double(x):
    if x == 3:
        return 0
    return x * 2
"""
        test_cases = {
            "inputs": [[1], [2], [3]],
            "outputs": ["2", "4", "6"],
            "fn_name": "double",
        }
        reward = compute_reward(
            generated_code=code,
            test_cases=test_cases,
            config=RewardConfig(timeout_seconds=2, memory_limit_mb=128),
        )
        self.assertAlmostEqual(reward, 2.0 / 3.0, places=6)

    def test_stdin_mode_reward(self) -> None:
        code = """
import sys

data = sys.stdin.read().strip()
if data:
    x = int(data)
    print(x * 2)
"""
        test_cases = {
            "inputs": ["1\n", "2\n", "3\n"],
            "outputs": ["2", "4", "6"],
        }
        reward = compute_reward(
            generated_code=code,
            test_cases=test_cases,
            config=RewardConfig(timeout_seconds=2, memory_limit_mb=128),
        )
        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_failure_returns_zero_reward(self) -> None:
        code = "def broken(:\n    pass"
        test_cases = {
            "inputs": [[1], [2]],
            "outputs": ["2", "4"],
            "fn_name": "double",
        }
        reward = compute_reward(
            generated_code=code,
            test_cases=test_cases,
            config=RewardConfig(timeout_seconds=1, memory_limit_mb=64),
        )
        self.assertEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
