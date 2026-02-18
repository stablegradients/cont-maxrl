"""Core modules for Experiment 4 continuous MaxRL code-generation RL."""

from .advantages import (
    compute_advantage_1d,
    compute_advantages_from_group_indices,
    compute_grouped_advantages,
    grpo_advantage,
    maxrl_normalized_advantage,
    maxrl_unnormalized_advantage,
    reinforce_baseline_advantage,
)

__all__ = [
    "compute_advantage_1d",
    "compute_advantages_from_group_indices",
    "compute_grouped_advantages",
    "grpo_advantage",
    "maxrl_normalized_advantage",
    "maxrl_unnormalized_advantage",
    "reinforce_baseline_advantage",
]
