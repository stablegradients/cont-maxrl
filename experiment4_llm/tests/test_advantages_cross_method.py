"""Tests comparing properties across advantage methods."""
import torch
import pytest


def test_grpo_vs_reinforce_relationship():
    """GRPO = REINFORCE / std. Verify the relationship."""
    from experiment4_llm.src.advantages import reinforce_baseline_advantage

    rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    reinforce_adv = reinforce_baseline_advantage(rewards)
    std = reinforce_adv.std(unbiased=False)

    # GRPO should be reinforce / std
    grpo_expected = reinforce_adv / (std + 1e-8)
    grpo_actual = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    assert torch.allclose(grpo_expected, grpo_actual, atol=1e-5)


def test_maxrl_norm_vs_unnorm_relationship():
    """MaxRL-norm = MaxRL-unnorm / std(MaxRL-unnorm)."""
    from experiment4_llm.src.advantages import (
        maxrl_unnormalized_advantage,
        maxrl_normalized_advantage,
    )

    rewards = torch.rand(16)
    unnorm = maxrl_unnormalized_advantage(rewards)
    norm = maxrl_normalized_advantage(rewards)

    if unnorm.std(unbiased=False) > 1e-6:
        expected_norm = unnorm / (unnorm.std(unbiased=False) + 1e-8)
        assert torch.allclose(norm, expected_norm, atol=1e-4), \
            f"norm: {norm[:4]}, expected: {expected_norm[:4]}"


def test_rloo_vs_reinforce_relationship():
    """RLOO = G^2/(G-1) * REINFORCE_baseline (includes * G for PPO loss scaling)."""
    from experiment4_llm.src.advantages import rloo_advantage, reinforce_baseline_advantage
    rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    G = rewards.numel()
    rloo = rloo_advantage(rewards)
    reinforce = reinforce_baseline_advantage(rewards)
    expected = (G / (G - 1)) * reinforce * G
    assert torch.allclose(rloo, expected, atol=1e-6)


def test_rloo_matches_verl_builtin_times_G():
    """Our rloo_advantage = G * verl's compute_rloo_outcome_advantage.

    The factor of G compensates for PPO's token-mean loss (same as GOS line 73).
    verl's built-in RLOO lacks this correction, so we intentionally override it.
    """
    import numpy as np
    from experiment4_llm.src.advantages import rloo_advantage
    from verl.trainer.ppo.core_algos import compute_rloo_outcome_advantage

    # Simulate a group: 1 prompt, G=8 completions
    G = 8
    seq_len = 20
    rewards_1d = torch.rand(G)

    # Our function (with * G)
    our_adv = rloo_advantage(rewards_1d)

    # verl's function (without * G): needs token_level_rewards [G, seq_len], response_mask, index
    token_rewards = torch.zeros(G, seq_len)
    response_mask = torch.zeros(G, seq_len)
    for i in range(G):
        # Place scalar reward at last response token (position seq_len-1)
        token_rewards[i, seq_len - 1] = rewards_1d[i]
        response_mask[i, :] = 1.0  # all tokens are response tokens
    index = np.array([0] * G)  # all same prompt

    verl_adv, _ = compute_rloo_outcome_advantage(token_rewards, response_mask, index)
    # verl broadcasts to [G, seq_len]; extract scalar per sequence
    verl_scalar = verl_adv[:, seq_len - 1]  # advantage at the reward token

    assert torch.allclose(our_adv, verl_scalar * G, atol=1e-5), \
        f"Mismatch: ours={our_adv[:4]}, G*verl={verl_scalar[:4] * G}"


def test_all_methods_agree_on_ordering():
    """All methods should rank completions in the same order."""
    from experiment4_llm.src.advantages import (
        maxrl_unnormalized_advantage,
        maxrl_normalized_advantage,
        reinforce_baseline_advantage,
        rloo_advantage,
    )

    rewards = torch.rand(16)
    adv_unnorm = maxrl_unnormalized_advantage(rewards)
    adv_norm = maxrl_normalized_advantage(rewards)
    adv_reinforce = reinforce_baseline_advantage(rewards)
    adv_rloo = rloo_advantage(rewards)

    order_unnorm = torch.argsort(adv_unnorm)
    order_norm = torch.argsort(adv_norm)
    order_reinforce = torch.argsort(adv_reinforce)
    order_rloo = torch.argsort(adv_rloo)

    assert torch.equal(order_unnorm, order_norm), "unnorm vs norm ordering differs"
    assert torch.equal(order_unnorm, order_reinforce), \
        "unnorm vs reinforce ordering differs"
    assert torch.equal(order_unnorm, order_rloo), "unnorm vs rloo ordering differs"


def test_maxrl_vs_grpo_top_weight_ratio():
    """
    MaxRL gives the top sample a higher proportion of total positive advantage
    than GRPO does. This is THE key property that drives tail exploration.
    """
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    for _ in range(50):
        rewards = torch.rand(16)
        maxrl = maxrl_unnormalized_advantage(rewards)
        grpo = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        # Fraction of total positive advantage that goes to the top sample
        top_idx = rewards.argmax()
        maxrl_pos = maxrl[maxrl > 0].sum()
        grpo_pos = grpo[grpo > 0].sum()

        if maxrl_pos > 1e-6 and grpo_pos > 1e-6:
            maxrl_top_frac = maxrl[top_idx] / maxrl_pos
            grpo_top_frac = grpo[top_idx] / grpo_pos
            assert maxrl_top_frac >= grpo_top_frac - 0.01, \
                f"MaxRL top frac {maxrl_top_frac:.3f} < GRPO {grpo_top_frac:.3f}"


def test_binary_rewards_maxrl_vs_grpo():
    """With binary rewards (0/1), MaxRL and REINFORCE should agree on sign."""
    from experiment4_llm.src.advantages import (
        maxrl_unnormalized_advantage,
        reinforce_baseline_advantage,
    )

    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    maxrl = maxrl_unnormalized_advantage(rewards)
    reinforce = reinforce_baseline_advantage(rewards)

    # Same sign (positive for 1s, negative for 0s)
    assert (torch.sign(maxrl) == torch.sign(reinforce)).all() or \
        torch.allclose(maxrl, torch.zeros_like(maxrl), atol=1e-6)
