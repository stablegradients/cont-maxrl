from __future__ import annotations

import unittest

import torch

from experiment4_llm.src.advantages import (
    compute_advantage_1d,
    compute_advantages_from_group_indices,
    compute_grouped_advantages,
    grpo_advantage,
    maxrl_normalized_advantage,
    maxrl_unnormalized_advantage,
    reinforce_baseline_advantage,
)


class AdvantageTests(unittest.TestCase):
    def test_reinforce_baseline_is_mean_centered(self) -> None:
        rewards = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
        adv = reinforce_baseline_advantage(rewards)
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=12)

    def test_grpo_constant_rewards_returns_zeros(self) -> None:
        rewards = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
        adv = grpo_advantage(rewards)
        self.assertTrue(torch.allclose(adv, torch.zeros_like(rewards)))

    def test_maxrl_unnorm_matches_reference_vector(self) -> None:
        rewards = torch.tensor([0.2, 0.4, 1.0], dtype=torch.float64)
        adv = maxrl_unnormalized_advantage(rewards)
        expected = torch.tensor([-0.2666666667, -0.1666666667, 0.4333333333], dtype=torch.float64)
        self.assertTrue(torch.allclose(adv, expected, atol=1e-8, rtol=0.0))
        self.assertAlmostEqual(float(adv.sum()), 0.0, places=10)

    def test_maxrl_norm_has_unit_std_when_nonconstant(self) -> None:
        rewards = torch.tensor([0.1, 0.2, 0.8, 1.0], dtype=torch.float64)
        adv = maxrl_normalized_advantage(rewards)
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(adv.std(unbiased=False)), 1.0, places=6)

    def test_grouped_advantages_apply_per_prompt(self) -> None:
        rewards = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.6, 0.6, 0.6],
            ],
            dtype=torch.float64,
        )
        adv = compute_grouped_advantages(rewards, method="grpo")
        self.assertAlmostEqual(float(adv[0].mean()), 0.0, places=10)
        self.assertTrue(torch.allclose(adv[1], torch.zeros_like(adv[1])))

    def test_group_index_api_matches_grouped_api(self) -> None:
        rewards = torch.tensor([0.1, 0.9, 0.3, 0.7, 0.4, 0.4], dtype=torch.float64)
        group_indices = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

        grouped = rewards.view(3, 2)
        adv_grouped = compute_grouped_advantages(grouped, method="maxrl_unnorm").reshape(-1)
        adv_indexed = compute_advantages_from_group_indices(
            rewards,
            group_indices,
            method="maxrl_unnorm",
        )

        self.assertTrue(torch.allclose(adv_grouped, adv_indexed, atol=1e-10, rtol=0.0))

    def test_dispatch_methods(self) -> None:
        rewards = torch.tensor([0.0, 0.2, 0.9], dtype=torch.float64)
        methods = ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]
        for method in methods:
            out = compute_advantage_1d(rewards, method=method)
            self.assertEqual(out.shape, rewards.shape)


if __name__ == "__main__":
    unittest.main()
