from __future__ import annotations

import unittest

import torch

import experiment1d_hard_bandit as bandit_impl
from experiment4_llm.src.advantages import (
    grpo_advantage,
    maxrl_normalized_advantage,
    maxrl_unnormalized_advantage,
    reinforce_baseline_advantage,
)


class AdvantageBanditEquivalenceTests(unittest.TestCase):
    def test_equivalence_with_bandit_formulas(self) -> None:
        generator = torch.Generator().manual_seed(1234)

        for group_size in (4, 8, 16, 32):
            for _ in range(10):
                rewards_2d = torch.rand((1, group_size), generator=generator, dtype=torch.float64)
                rewards_1d = rewards_2d.squeeze(0)

                grpo_ref = bandit_impl.advantages_grpo(rewards_2d, eps=1e-8).squeeze(0)
                reinforce_ref = bandit_impl.advantages_reinforce_baseline(rewards_2d).squeeze(0)
                maxrl_norm_ref = bandit_impl.advantages_continuous_maxrl_normalized(
                    rewards_2d, eps=1e-8
                ).squeeze(0)
                maxrl_unnorm_ref = bandit_impl.advantages_continuous_maxrl_unnormalized(
                    rewards_2d, eps=1e-8
                ).squeeze(0)

                self.assertTrue(
                    torch.allclose(grpo_advantage(rewards_1d), grpo_ref, atol=1e-10, rtol=0.0)
                )
                self.assertTrue(
                    torch.allclose(
                        reinforce_baseline_advantage(rewards_1d),
                        reinforce_ref,
                        atol=1e-10,
                        rtol=0.0,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        maxrl_normalized_advantage(rewards_1d),
                        maxrl_norm_ref,
                        atol=1e-10,
                        rtol=0.0,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        maxrl_unnormalized_advantage(rewards_1d),
                        maxrl_unnorm_ref,
                        atol=1e-10,
                        rtol=0.0,
                    )
                )


if __name__ == "__main__":
    unittest.main()
