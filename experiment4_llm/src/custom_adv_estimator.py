from __future__ import annotations

from functools import partial
from typing import Callable, Mapping, MutableMapping

import torch

from .advantages import (
    AdvantageMethod,
    broadcast_sequence_advantages_to_tokens,
    compute_advantages_from_group_indices,
    compute_grouped_advantages,
)


def extract_sequence_rewards(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract one scalar sequence reward from token-level rewards.

    Expected shape:
        token_level_rewards: [batch, seq_len]
        eos_mask: [batch, seq_len] with 1 on EOS token(s)
    """
    if token_level_rewards.ndim != 2 or eos_mask.ndim != 2:
        raise ValueError("token_level_rewards and eos_mask must be rank-2 tensors")
    if token_level_rewards.shape != eos_mask.shape:
        raise ValueError("token_level_rewards and eos_mask must have the same shape")
    return (token_level_rewards * eos_mask.to(dtype=token_level_rewards.dtype)).sum(dim=1)


def compute_sequence_advantages(
    sequence_rewards: torch.Tensor,
    method: AdvantageMethod,
    group_size: int | None = None,
    prompt_indices: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute sequence-level advantages grouped by prompt.

    Use either:
    - fixed group_size with contiguous layout [num_prompts * group_size]
    - explicit prompt_indices [batch]
    """
    if sequence_rewards.ndim != 1:
        raise ValueError("sequence_rewards must be 1D")

    if prompt_indices is not None:
        return compute_advantages_from_group_indices(
            rewards=sequence_rewards,
            group_indices=prompt_indices,
            method=method,
            eps=eps,
        )

    if group_size is None:
        raise ValueError("Either group_size or prompt_indices must be provided")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if sequence_rewards.shape[0] % group_size != 0:
        raise ValueError("Batch size must be divisible by group_size for contiguous grouping")

    grouped = sequence_rewards.view(-1, group_size)
    grouped_adv = compute_grouped_advantages(grouped, method=method, eps=eps)
    return grouped_adv.reshape(-1)


def compute_token_advantages(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    method: AdvantageMethod,
    group_size: int | None = None,
    prompt_indices: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute token advantages for GRPO/MaxRL style outcome rewards.

    Steps:
    1) extract sequence rewards from eos token rewards
    2) compute per-sequence advantages grouped by prompt
    3) broadcast sequence advantage to all response tokens
    """
    seq_rewards = extract_sequence_rewards(token_level_rewards, eos_mask)
    seq_advantages = compute_sequence_advantages(
        sequence_rewards=seq_rewards,
        method=method,
        group_size=group_size,
        prompt_indices=prompt_indices,
        eps=eps,
    )

    if token_mask is None:
        token_mask = torch.ones_like(token_level_rewards)

    if token_mask.shape != token_level_rewards.shape:
        raise ValueError("token_mask must match token_level_rewards shape")

    return broadcast_sequence_advantages_to_tokens(seq_advantages, token_mask=token_mask)


def build_advantage_fn(
    method: AdvantageMethod,
    group_size: int | None,
    eps: float = 1e-8,
) -> Callable[..., torch.Tensor]:
    """Return a callable suitable for plugging into trainer code."""

    def _fn(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        prompt_indices: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return compute_token_advantages(
            token_level_rewards=token_level_rewards,
            eos_mask=eos_mask,
            method=method,
            group_size=group_size,
            prompt_indices=prompt_indices,
            token_mask=token_mask,
            eps=eps,
        )

    return _fn


def register_advantage_estimators(
    registry: MutableMapping[str, Callable[..., torch.Tensor]],
    group_size: int | None,
    eps: float = 1e-8,
) -> Mapping[str, Callable[..., torch.Tensor]]:
    """
    Register estimators in a dict-like registry.

    Example:
        registry = {}
        register_advantage_estimators(registry, group_size=16)
        token_adv = registry["maxrl_unnorm"](token_level_rewards, eos_mask)
    """
    methods: list[AdvantageMethod] = [
        "grpo",
        "maxrl_unnorm",
        "maxrl_norm",
        "reinforce_baseline",
    ]

    for method in methods:
        registry[method] = partial(
            compute_token_advantages,
            method=method,
            group_size=group_size,
            eps=eps,
        )
    return registry


def _self_test() -> None:
    batch = 8
    seq_len = 5
    group_size = 4

    token_rewards = torch.zeros(batch, seq_len)
    eos_mask = torch.zeros(batch, seq_len)
    eos_mask[:, -1] = 1
    token_rewards[:, -1] = torch.tensor([0.0, 0.3, 0.7, 1.0, 0.1, 0.2, 0.2, 0.9])
    token_mask = torch.ones_like(token_rewards)

    for method in ("grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"):
        token_adv = compute_token_advantages(
            token_level_rewards=token_rewards,
            eos_mask=eos_mask,
            method=method,
            group_size=group_size,
            token_mask=token_mask,
        )
        assert token_adv.shape == token_rewards.shape

    print("custom_adv_estimator self-test passed")


if __name__ == "__main__":
    _self_test()
