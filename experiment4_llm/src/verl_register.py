"""
Custom advantage estimators for verl integration.

The AdvantageEstimator enum, __init__ check, and compute_advantage dispatch
are patched directly in vendor/verl/verl/trainer/ppo/ray_trainer.py.

This module provides the CUSTOM_ESTIMATORS dict that ray_trainer.py imports
at dispatch time.

Usage:
    # ray_trainer.py does:
    from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from experiment4_llm.src.advantages import (
    maxrl_unnormalized_advantage as _maxrl_unnorm_1d,
    maxrl_normalized_advantage as _maxrl_norm_1d,
    reinforce_baseline_advantage as _reinforce_baseline_1d,
    rloo_advantage as _rloo_1d,
    binary_maxrl_advantage as _binary_maxrl_1d,
)


# ---------------------------------------------------------------------------
# verl-compatible wrapper: matches the exact signature verl calls
# ---------------------------------------------------------------------------

def _make_verl_advantage_fn(per_group_fn):
    """
    Factory that wraps a per-group advantage function into verl's expected
    signature:
        fn(token_level_rewards, response_mask, index, epsilon)
        -> (advantages, returns)

    BATCH STRUCTURE (B prompts x N rollouts):
    ==========================================
    verl flattens B prompts x N rollouts into a single batch dimension.
    If B=128 prompts and N=16 rollouts/prompt, batch has B*N = 2048 rows.

    - token_level_rewards: [B*N, seq_len] -- reward placed at EOS token position
    - response_mask:       [B*N, seq_len] -- 1 for response tokens, 0 for padding
    - index:               [B*N]          -- prompt group ID per row

    We must:
    1. Extract scalar reward per sequence: sum(token_level_rewards * response_mask, dim=-1)
    2. Group the B*N scalars by prompt ID (using `index`)
    3. Apply per_group_fn to each group of N rewards
    4. Scatter advantages back to [B*N]
    5. Broadcast to [B*N, seq_len] via unsqueeze(-1) * response_mask
    """

    def verl_advantage_fn(
        token_level_rewards: torch.Tensor,      # [B*N, seq_len]
        response_mask: torch.Tensor,             # [B*N, seq_len]
        index: np.ndarray,                       # [B*N] prompt group IDs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            # Step 1: scalar reward per sequence  [B*N]
            scores = (token_level_rewards * response_mask).sum(dim=-1)

            # Step 2: group by prompt index -> {uid: [global_indices]}
            id2indices: dict = defaultdict(list)
            bsz = scores.shape[0]
            for i in range(bsz):
                id2indices[index[i]].append(i)

            # Step 3 + 4: per-group advantage, vectorized scatter-back
            seq_advantages = torch.zeros_like(scores)  # [B*N]
            for uid, global_indices in id2indices.items():
                idx_t = torch.tensor(
                    global_indices, dtype=torch.long, device=scores.device
                )
                group_scores = scores[idx_t]            # [N] gather
                group_adv = per_group_fn(group_scores)   # [N] compute
                seq_advantages[idx_t] = group_adv        # [N] scatter

            # Step 5: broadcast scalar advantage -> [B*N, seq_len]
            advantages = seq_advantages.unsqueeze(-1) * response_mask
            returns = advantages  # outcome reward: returns = advantages

        return advantages, returns

    return verl_advantage_fn


# ---------------------------------------------------------------------------
# Custom estimator dispatch table (imported by ray_trainer.py)
# ---------------------------------------------------------------------------

CUSTOM_ESTIMATORS = {
    "maxrl_unnorm": _make_verl_advantage_fn(_maxrl_unnorm_1d),
    "maxrl_norm": _make_verl_advantage_fn(_maxrl_norm_1d),
    "reinforce_baseline": _make_verl_advantage_fn(_reinforce_baseline_1d),
    "rloo": _make_verl_advantage_fn(_rloo_1d),
    "binary_maxrl": _make_verl_advantage_fn(_binary_maxrl_1d),
}


# ---------------------------------------------------------------------------
# DG wrapper: per-token gates, needs old_log_probs (different signature
# from per-group estimators in CUSTOM_ESTIMATORS).
# ---------------------------------------------------------------------------

def _make_dg_verl_fn(eta: float = 1.0, surprisal_clip: float = 20.0):
    """
    DG has per-token gates, so it doesn't use the _make_verl_advantage_fn
    per-group abstraction. Builds [B*N, T] advantages directly.
    """
    from experiment4_llm.src.advantages import dg_advantage_per_token

    def verl_dg_fn(
        token_level_rewards: torch.Tensor,   # [B*N, T]
        response_mask: torch.Tensor,          # [B*N, T]
        index: np.ndarray,                    # [B*N]
        old_log_probs: torch.Tensor,          # [B*N, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            # Scalar reward per sequence: [B*N]
            scores = (token_level_rewards * response_mask).sum(dim=-1)

            # Output buffer: [B*N, T]
            advantages = torch.zeros_like(token_level_rewards)

            # Group by prompt index
            id2indices: dict = defaultdict(list)
            bsz = scores.shape[0]
            for i in range(bsz):
                id2indices[index[i]].append(i)

            for uid, global_indices in id2indices.items():
                idx_t = torch.tensor(
                    global_indices, dtype=torch.long, device=scores.device
                )
                group_rewards = scores[idx_t]                       # [G]
                group_lp = old_log_probs[idx_t]                     # [G, T]
                group_mask = response_mask[idx_t]                   # [G, T]

                group_adv = dg_advantage_per_token(
                    rewards=group_rewards,
                    old_log_probs=group_lp,
                    response_mask=group_mask,
                    eta=eta,
                    surprisal_clip=surprisal_clip,
                )                                                    # [G, T]
                advantages[idx_t] = group_adv

            returns = advantages

        return advantages, returns

    return verl_dg_fn


# Exposed factory; ray_trainer.py calls this at dispatch with hardcoded η=1.0.
DG_FACTORY = _make_dg_verl_fn
