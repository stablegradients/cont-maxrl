"""Tests for the full verl-compatible batch pipeline with B prompts x N rollouts."""
import torch
import numpy as np
import pytest


def _make_batch(B: int, N: int, seq_len: int = 32, reward_fn=None):
    """
    Create a synthetic verl-style batch.

    Args:
        B: number of prompts
        N: number of rollouts per prompt
        seq_len: response length
        reward_fn: optional fn(prompt_idx, rollout_idx) -> reward scalar

    Returns:
        token_level_rewards: [B*N, seq_len]
        response_mask: [B*N, seq_len]
        index: np.ndarray of shape [B*N]
        rewards_2d: [B, N] -- ground truth rewards for verification
    """
    batch = B * N
    token_level_rewards = torch.zeros(batch, seq_len)
    response_mask = torch.zeros(batch, seq_len)

    # Variable-length responses: each sequence has a random response length
    rewards_2d = torch.zeros(B, N)
    index = np.zeros(batch, dtype=np.int64)

    for b in range(B):
        for n in range(N):
            flat_idx = b * N + n
            index[flat_idx] = b

            # Random response length (at least 5 tokens)
            resp_len = max(5, int(torch.randint(5, seq_len, (1,)).item()))
            response_mask[flat_idx, :resp_len] = 1.0

            # Place reward at the last response token (EOS position)
            if reward_fn is not None:
                r = reward_fn(b, n)
            else:
                r = torch.rand(1).item()
            token_level_rewards[flat_idx, resp_len - 1] = r
            rewards_2d[b, n] = r

    return token_level_rewards, response_mask, index, rewards_2d


def _get_custom_fn(method):
    """Get custom estimator function by name."""
    import experiment4_llm.src.verl_register as reg
    return reg.CUSTOM_ESTIMATORS[method]


@pytest.mark.parametrize("method", ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"])
@pytest.mark.parametrize("B,N", [(1, 4), (4, 4), (16, 8), (32, 16), (128, 32)])
def test_batch_output_shape(method, B, N):
    """Output shape must be [B*N, seq_len] for all batch sizes."""
    seq_len = 20
    token_rewards, mask, index, _ = _make_batch(B, N, seq_len)
    fn = _get_custom_fn(method)
    adv, ret = fn(token_rewards, mask, index)

    assert adv.shape == (B * N, seq_len), f"adv shape {adv.shape}"
    assert ret.shape == (B * N, seq_len), f"ret shape {ret.shape}"


@pytest.mark.parametrize("method", ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"])
def test_batch_advantages_respect_mask(method):
    """Advantages must be zero where response_mask is zero."""
    B, N, seq_len = 8, 8, 32
    token_rewards, mask, index, _ = _make_batch(B, N, seq_len)
    fn = _get_custom_fn(method)
    adv, _ = fn(token_rewards, mask, index)

    # Where mask=0, advantages must be 0
    padding_advantages = adv * (1 - mask)
    assert (padding_advantages == 0).all(), \
        f"{method}: advantages leak into padding"


@pytest.mark.parametrize("method", ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"])
def test_batch_per_prompt_centering(method):
    """Within each prompt's N rollouts, advantages should sum to ~0."""
    B, N, seq_len = 16, 16, 20
    token_rewards, mask, index, _ = _make_batch(B, N, seq_len)
    fn = _get_custom_fn(method)
    adv, _ = fn(token_rewards, mask, index)

    # Extract scalar advantage per sequence (all response tokens get the same value)
    for b in range(B):
        group_advs = []
        for n in range(N):
            flat_idx = b * N + n
            # Find first non-zero mask position
            resp_positions = mask[flat_idx].nonzero(as_tuple=True)[0]
            if len(resp_positions) > 0:
                group_advs.append(adv[flat_idx, resp_positions[0]].item())
        if group_advs:
            group_sum = sum(group_advs)
            assert abs(group_sum) < 1e-4, \
                f"{method}: prompt {b} advantages sum to {group_sum}"


@pytest.mark.parametrize("method", ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"])
def test_batch_constant_advantage_per_sequence(method):
    """All response tokens in a sequence should get the same advantage scalar."""
    B, N, seq_len = 4, 8, 32
    token_rewards, mask, index, _ = _make_batch(B, N, seq_len)
    fn = _get_custom_fn(method)
    adv, _ = fn(token_rewards, mask, index)

    for i in range(B * N):
        resp_positions = mask[i].nonzero(as_tuple=True)[0]
        if len(resp_positions) > 1:
            resp_advs = adv[i, resp_positions]
            # All response tokens should have the same advantage value
            assert torch.allclose(
                resp_advs, resp_advs[0:1].expand_as(resp_advs), atol=1e-7
            ), f"{method}: seq {i} has non-constant advantages: {resp_advs[:5]}"


def test_batch_matches_per_group():
    """Batch pipeline must produce same results as calling per-group function directly."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage
    import experiment4_llm.src.verl_register as reg

    B, N, seq_len = 8, 8, 20
    token_rewards, mask, index, rewards_2d = _make_batch(B, N, seq_len)

    # Get batch result
    fn = reg.CUSTOM_ESTIMATORS["maxrl_unnorm"]
    adv_batch, _ = fn(token_rewards, mask, index)

    # Get per-group result
    for b in range(B):
        group_rewards = rewards_2d[b]  # [N]
        expected_adv = maxrl_unnormalized_advantage(group_rewards)  # [N]

        for n in range(N):
            flat_idx = b * N + n
            resp_positions = mask[flat_idx].nonzero(as_tuple=True)[0]
            if len(resp_positions) > 0:
                actual_scalar = adv_batch[flat_idx, resp_positions[0]].item()
                expected_scalar = expected_adv[n].item()
                assert abs(actual_scalar - expected_scalar) < 1e-5, \
                    f"Mismatch at prompt {b}, rollout {n}: " \
                    f"{actual_scalar} vs {expected_scalar}"


def test_batch_non_contiguous_index():
    """index array with non-contiguous grouping (interleaved prompts)."""
    import experiment4_llm.src.verl_register as reg

    # 2 prompts, 3 rollouts each, interleaved: [0, 1, 0, 1, 0, 1]
    seq_len = 10
    token_rewards = torch.zeros(6, seq_len)
    mask = torch.zeros(6, seq_len)
    index = np.array([0, 1, 0, 1, 0, 1])

    rewards_by_prompt = {0: [0.2, 0.5, 0.8], 1: [0.1, 0.3, 0.9]}
    global_idx_by_prompt = {0: [0, 2, 4], 1: [1, 3, 5]}

    for pid in [0, 1]:
        for local_i, global_i in enumerate(global_idx_by_prompt[pid]):
            mask[global_i, :8] = 1.0
            token_rewards[global_i, 7] = rewards_by_prompt[pid][local_i]

    fn = reg.CUSTOM_ESTIMATORS["maxrl_unnorm"]
    adv, _ = fn(token_rewards, mask, index)

    # Verify each prompt's advantages are computed correctly
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage
    for pid in [0, 1]:
        r = torch.tensor(rewards_by_prompt[pid])
        expected = maxrl_unnormalized_advantage(r)
        for local_i, global_i in enumerate(global_idx_by_prompt[pid]):
            actual = adv[global_i, 0].item()  # first response token
            assert abs(actual - expected[local_i].item()) < 1e-5, \
                f"Prompt {pid}, rollout {local_i}: " \
                f"{actual} vs {expected[local_i]}"


def test_batch_all_prompts_independent():
    """Changing one prompt's rewards should not affect another's advantages."""
    import experiment4_llm.src.verl_register as reg

    B, N, seq_len = 4, 4, 10
    token_rewards, mask, index, _ = _make_batch(B, N, seq_len)

    fn = reg.CUSTOM_ESTIMATORS["maxrl_unnorm"]
    adv1, _ = fn(token_rewards.clone(), mask, index)

    # Modify prompt 0's rewards
    for n in range(N):
        flat_idx = n
        resp_positions = mask[flat_idx].nonzero(as_tuple=True)[0]
        if len(resp_positions) > 0:
            token_rewards[flat_idx, resp_positions[-1]] = torch.rand(1).item()

    adv2, _ = fn(token_rewards, mask, index)

    # Prompts 1, 2, 3 should be unchanged
    for b in [1, 2, 3]:
        for n in range(N):
            flat_idx = b * N + n
            assert torch.allclose(adv1[flat_idx], adv2[flat_idx], atol=1e-7), \
                f"Prompt {b} changed when only prompt 0 was modified"


@pytest.mark.parametrize("method", ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"])
def test_batch_all_zeros_rewards(method):
    """All zero rewards -> all zero advantages."""
    B, N, seq_len = 4, 8, 10
    token_rewards = torch.zeros(B * N, seq_len)
    mask = torch.ones(B * N, seq_len)
    index = np.repeat(np.arange(B), N)

    fn = _get_custom_fn(method)
    adv, _ = fn(token_rewards, mask, index)
    assert (adv == 0).all(), f"{method}: non-zero advantages for zero rewards"


@pytest.mark.parametrize("method", ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"])
def test_batch_realistic_scale(method):
    """Realistic batch: B=128, N=16, seq_len=256. Must not crash or NaN."""
    B, N, seq_len = 128, 16, 256
    token_rewards, mask, index, _ = _make_batch(B, N, seq_len)
    fn = _get_custom_fn(method)
    adv, ret = fn(token_rewards, mask, index)

    assert adv.shape == (B * N, seq_len)
    assert not torch.isnan(adv).any(), f"{method}: NaN in advantages"
    assert not torch.isinf(adv).any(), f"{method}: Inf in advantages"
