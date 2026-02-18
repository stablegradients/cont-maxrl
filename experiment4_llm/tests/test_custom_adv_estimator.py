from __future__ import annotations

import unittest

import torch

from experiment4_llm.src.custom_adv_estimator import (
    compute_sequence_advantages,
    compute_token_advantages,
    extract_sequence_rewards,
)


class CustomAdvEstimatorTests(unittest.TestCase):
    def test_extract_sequence_rewards(self) -> None:
        rewards = torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        eos = torch.tensor(
            [
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        seq = extract_sequence_rewards(rewards, eos)
        self.assertTrue(torch.allclose(seq, torch.tensor([0.5, 1.0])))

    def test_compute_sequence_advantages_by_group_size(self) -> None:
        seq_rewards = torch.tensor([0.0, 0.2, 0.9, 1.0, 0.1, 0.1, 0.4, 0.8], dtype=torch.float64)
        seq_adv = compute_sequence_advantages(
            sequence_rewards=seq_rewards,
            method="maxrl_unnorm",
            group_size=4,
        )
        self.assertEqual(seq_adv.shape, seq_rewards.shape)
        self.assertAlmostEqual(float(seq_adv[:4].sum()), 0.0, places=10)
        self.assertAlmostEqual(float(seq_adv[4:].sum()), 0.0, places=10)

    def test_compute_sequence_advantages_by_prompt_indices(self) -> None:
        seq_rewards = torch.tensor([0.0, 0.2, 0.9, 1.0, 0.1, 0.1, 0.4, 0.8], dtype=torch.float64)
        prompt_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
        seq_adv = compute_sequence_advantages(
            sequence_rewards=seq_rewards,
            method="grpo",
            prompt_indices=prompt_indices,
        )
        self.assertEqual(seq_adv.shape, seq_rewards.shape)
        self.assertAlmostEqual(float(seq_adv[:4].mean()), 0.0, places=10)
        self.assertAlmostEqual(float(seq_adv[4:].mean()), 0.0, places=10)

    def test_compute_token_advantages(self) -> None:
        token_rewards = torch.zeros((4, 5), dtype=torch.float32)
        eos = torch.zeros((4, 5), dtype=torch.float32)
        token_mask = torch.ones((4, 5), dtype=torch.float32)
        eos[:, -1] = 1
        token_rewards[:, -1] = torch.tensor([0.0, 0.4, 0.4, 1.0], dtype=torch.float32)

        token_adv = compute_token_advantages(
            token_level_rewards=token_rewards,
            eos_mask=eos,
            method="reinforce_baseline",
            group_size=4,
            token_mask=token_mask,
        )
        self.assertEqual(token_adv.shape, token_rewards.shape)
        self.assertAlmostEqual(float(token_adv[:, -1].mean()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
