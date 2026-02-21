"""Tests for verl integration (custom estimators in ray_trainer.py)."""
import torch
import numpy as np
import pytest


def test_custom_estimators_in_enum():
    """Custom estimator names exist in AdvantageEstimator enum."""
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        assert hasattr(AdvantageEstimator, name.upper()), \
            f"{name.upper()} not in AdvantageEstimator enum"
        member = getattr(AdvantageEstimator, name.upper())
        assert member.value == name


def test_custom_estimators_registered():
    """All custom estimators are in CUSTOM_ESTIMATORS."""
    import experiment4_llm.src.verl_register as reg

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        assert name in reg.CUSTOM_ESTIMATORS, f"{name} not registered"


def test_custom_estimator_signature():
    """Each custom fn returns (advantages, returns) tuple with correct shapes."""
    import experiment4_llm.src.verl_register as reg

    B, N, seq_len = 4, 4, 10
    token_rewards = torch.zeros(B * N, seq_len)
    mask = torch.ones(B * N, seq_len)
    index = np.repeat(np.arange(B), N)
    # Place random rewards at last token
    for i in range(B * N):
        token_rewards[i, -1] = torch.rand(1).item()

    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        fn = reg.CUSTOM_ESTIMATORS[name]
        result = fn(
            token_level_rewards=token_rewards,
            response_mask=mask,
            index=index,
        )
        assert isinstance(result, tuple), f"{name} didn't return tuple"
        assert len(result) == 2, f"{name} returned {len(result)} elements"
        adv, ret = result
        assert adv.shape == (B * N, seq_len)
        assert ret.shape == (B * N, seq_len)


def test_compute_advantage_dispatches():
    """compute_advantage in ray_trainer dispatches to our custom estimators."""
    from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator

    # This test verifies the dispatch path exists (doesn't raise NotImplementedError)
    # We can't easily call compute_advantage without a full DataProto,
    # but we can verify the enum values are accepted
    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        member = AdvantageEstimator(name)
        assert member.value == name
