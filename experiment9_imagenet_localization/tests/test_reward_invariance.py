"""Tests for reward-invariance properties of GOS and Proposition 3.3.

This module verifies:
  1. Gradient-direction invariance of GOS under affine and monotone reward transforms.
  2. Proposition 3.3 (reward-level): gos_advantage applied to 1{IoU > tau} equals
     binary_maxrl_advantage applied to the same binarized vector, for each
     threshold tau in {0.0, 0.25, 0.5, 0.75, 0.95}.

All tests run on CPU and complete well under 5 seconds.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiment9_imagenet_localization.advantages import (
    binary_maxrl_advantage,
    gos_advantage,
    grpo_advantage,
)
from experiment9_imagenet_localization.iou import box_iou_xywh


# ---------------------------------------------------------------------------
# Helper: compute policy-gradient gradient w.r.t. logits
# ---------------------------------------------------------------------------


def _compute_pg_gradient(
    logits: torch.Tensor,
    samples: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute and return the gradient of the PG loss w.r.t. `logits`.

    Args:
        logits: (N, K) parameter tensor with requires_grad=True.
        samples: (N,) int64 action indices.
        advantages: (N,) pre-computed, detached advantage values.

    Returns:
        (N*K,) flattened gradient tensor.
    """
    if logits.grad is not None:
        logits.grad.zero_()

    log_probs = torch.log_softmax(logits, dim=-1)
    log_prob_samples = log_probs.gather(1, samples.unsqueeze(1)).squeeze(1)  # (N,)
    loss = -(advantages.detach() * log_prob_samples).mean()
    loss.backward()
    return logits.grad.clone().flatten()


# ===========================================================================
# Test 1: Affine reward transform preserves gradient direction under GOS
# ===========================================================================


def test_gos_affine_transform_preserves_gradient_direction():
    """GOS gradient direction must be invariant to affine reward transforms.

    Setup: N=64, K=10. Random logits and samples. Continuous rewards in [0, 1].
    Compare the PG gradient direction using raw rewards vs. a * r + b (a=2.5, b=-0.3).

    The GOS invariance claim: scaling and shifting rewards does NOT change the
    advantage rank, so gradient direction (cosine similarity) should be > 0.999.
    A broken implementation that drops the *N scaling would fail here because the
    affine shift changes the relative magnitude of advantages differently across
    samples.
    """
    torch.manual_seed(1)
    N, K = 64, 10
    a, b = 2.5, -0.3

    logits1 = torch.nn.Parameter(torch.randn(N, K))
    logits2 = torch.nn.Parameter(torch.randn(N, K))
    # Use same logits for fair comparison — copy initial values
    with torch.no_grad():
        logits2.copy_(logits1)

    samples = torch.randint(0, K, (N,))
    rewards = torch.rand(N)

    adv1 = gos_advantage(rewards)
    adv2 = gos_advantage(a * rewards + b)

    grad1 = _compute_pg_gradient(logits1, samples, adv1)
    grad2 = _compute_pg_gradient(logits2, samples, adv2)

    cos_sim = F.cosine_similarity(grad1.unsqueeze(0), grad2.unsqueeze(0)).item()
    assert cos_sim > 0.999, (
        f"GOS affine invariance failed: cosine similarity = {cos_sim:.6f}, expected > 0.999. "
        f"The gradient direction changed under reward transform a={a}, b={b}."
    )


# ===========================================================================
# Test 2: GOS is more robust than GRPO under monotone transforms
# ===========================================================================


@pytest.mark.parametrize("seed", [10, 20, 30])
def test_gos_monotone_transform_more_robust_than_grpo(seed):
    """GOS gradient direction must be more robust than GRPO under r -> r^2.

    Compares the cosine similarity between gradients from raw rewards and
    from r^2 (a monotone but nonlinear transform) for both GOS and GRPO.
    GOS is rank-based so a monotone transform preserves the rank ordering,
    meaning its gradient direction should change less than GRPO's z-score
    which is sensitive to the moment structure of the distribution.

    Asserted: cos_gos > cos_grpo (strict, per seed).
    """
    torch.manual_seed(seed)
    N, K = 64, 10

    # Four independent logit tensors (two for GOS, two for GRPO)
    logits_gos_r = torch.nn.Parameter(torch.randn(N, K))
    logits_gos_r2 = torch.nn.Parameter(torch.randn(N, K))
    logits_grpo_r = torch.nn.Parameter(torch.randn(N, K))
    logits_grpo_r2 = torch.nn.Parameter(torch.randn(N, K))

    # Share initial logit values
    with torch.no_grad():
        logits_gos_r2.copy_(logits_gos_r)
        logits_grpo_r.copy_(logits_gos_r)
        logits_grpo_r2.copy_(logits_gos_r)

    samples = torch.randint(0, K, (N,))
    rewards = torch.rand(N)
    rewards_sq = rewards ** 2

    adv_gos_r = gos_advantage(rewards)
    adv_gos_r2 = gos_advantage(rewards_sq)
    adv_grpo_r = grpo_advantage(rewards)
    adv_grpo_r2 = grpo_advantage(rewards_sq)

    grad_gos_r = _compute_pg_gradient(logits_gos_r, samples, adv_gos_r)
    grad_gos_r2 = _compute_pg_gradient(logits_gos_r2, samples, adv_gos_r2)
    grad_grpo_r = _compute_pg_gradient(logits_grpo_r, samples, adv_grpo_r)
    grad_grpo_r2 = _compute_pg_gradient(logits_grpo_r2, samples, adv_grpo_r2)

    cos_gos = F.cosine_similarity(grad_gos_r.unsqueeze(0), grad_gos_r2.unsqueeze(0)).item()
    cos_grpo = F.cosine_similarity(grad_grpo_r.unsqueeze(0), grad_grpo_r2.unsqueeze(0)).item()

    assert cos_gos > cos_grpo, (
        f"seed={seed}: Expected GOS more robust than GRPO under r->r^2, "
        f"but cos_gos={cos_gos:.6f} <= cos_grpo={cos_grpo:.6f}."
    )


# ===========================================================================
# Test 3: Proposition 3.3 — gos_advantage on 1{IoU > tau} == binary_maxrl
# ===========================================================================


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5, 0.75, 0.95], ids=lambda t: f"tau={t}")
def test_gos_on_binarized_iou_matches_binary_maxrl(tau):
    """Proposition 3.3 (reward-level): GOS on 1{IoU > tau} must equal binary_maxrl.

    The equivalence holds because binary rewards (0 or 1 exactly) are a special
    case of GOS with exactly one gap at reward value 1.0. When GOS is applied to
    a binary reward vector, it algebraically reduces to the Tajwar binary_maxrl
    formula. This test verifies this at the reward level using realistic IoU values.

    Setup: N=64 random xywh boxes vs. a fixed GT box. Binarize at threshold tau.
    Both methods must produce bit-for-bit equal advantages (atol=1e-5).

    At tau=0.0 or tau=0.95, the outcomes may be all-1 or all-0 depending on the
    random draw — both methods must return zeros in those degenerate cases.
    """
    torch.manual_seed(42)
    N = 64

    sampled_boxes = torch.rand(N, 4)   # x_c, y_c, w, h in [0, 1]
    gt_box = torch.tensor([0.5, 0.5, 0.3, 0.3])

    rewards = box_iou_xywh(sampled_boxes, gt_box)  # (N,) in [0, 1]

    # Binarize: this is the 1{IoU > tau} reward
    r_binary = (rewards > tau).float()

    # GOS applied to the binarized reward (NOT the continuous IoU)
    adv_gos = gos_advantage(r_binary)

    # binary_maxrl applied to the same binarized reward
    adv_binary = binary_maxrl_advantage(r_binary)

    torch.testing.assert_close(adv_gos, adv_binary, atol=1e-5, rtol=0)


# ===========================================================================
# Test 4: All-1 rewards — both methods return zeros
# ===========================================================================


def test_gos_on_fully_saturated_rewards_matches_binary_trivially():
    """All-success rewards must produce zeros from both GOS and binary_maxrl.

    When all rewards are 1.0, there is no relative signal to learn from.
    Both methods must return all-zero advantages so that the policy gradient
    update is zero — no spurious gradient from a degenerate batch.
    """
    torch.manual_seed(5)
    N = 32
    rewards = torch.ones(N)

    adv_gos = gos_advantage(rewards)
    adv_binary = binary_maxrl_advantage(rewards)

    torch.testing.assert_close(adv_gos, torch.zeros(N), atol=1e-6, rtol=0)
    torch.testing.assert_close(adv_binary, torch.zeros(N), atol=1e-6, rtol=0)


# ===========================================================================
# Test 5: All-0 rewards — both methods return zeros
# ===========================================================================


def test_gos_on_fully_zero_rewards_matches_binary_trivially():
    """All-failure rewards must produce zeros from both GOS and binary_maxrl.

    When all rewards are 0.0, there is no relative signal to learn from.
    Both methods must return all-zero advantages — no spurious gradient.
    This is the dual of the all-success case.
    """
    torch.manual_seed(6)
    N = 32
    rewards = torch.zeros(N)

    adv_gos = gos_advantage(rewards)
    adv_binary = binary_maxrl_advantage(rewards)

    torch.testing.assert_close(adv_gos, torch.zeros(N), atol=1e-6, rtol=0)
    torch.testing.assert_close(adv_binary, torch.zeros(N), atol=1e-6, rtol=0)


# ===========================================================================
# Test 6: tau=0 binarization maps all positive-IoU to success (all zeros)
# ===========================================================================


def test_binarization_at_tau_zero_maps_positive_iou_to_success():
    """Binarizing at tau=0 maps all positive rewards to 1; both methods yield zeros.

    With N=16 rewards drawn uniformly in (0, 1) (none exactly 0 or 1),
    the condition rewards > 0.0 is True for all samples.
    binary_maxrl returns zeros because K=N (all successes, no signal).
    gos_advantage returns zeros because all values are equal (all 1.0).

    A failure would indicate a bug in the degenerate all-equal handling
    of either function.
    """
    torch.manual_seed(7)
    N = 16

    # Draw from (epsilon, 1-epsilon) to ensure no exact 0 or 1
    rewards = torch.rand(N) * 0.98 + 0.01  # in [0.01, 0.99]

    r_binary = (rewards > 0.0).float()  # all 1.0

    adv_gos = gos_advantage(r_binary)
    adv_binary = binary_maxrl_advantage(r_binary)

    torch.testing.assert_close(adv_gos, torch.zeros(N), atol=1e-6, rtol=0)
    torch.testing.assert_close(adv_binary, torch.zeros(N), atol=1e-6, rtol=0)


# ===========================================================================
# Test 7: GOS == binary_maxrl on Bernoulli rewards across N sizes
# ===========================================================================


@pytest.mark.parametrize("N", [4, 16, 64, 256])
def test_gos_binary_equivalence_holds_across_N(N):
    """GOS and binary_maxrl must agree on Bernoulli rewards at every batch size.

    This is a scale sanity check replicating the advantage-level equivalence
    test (Proposition 3.3) over a wider range of N. We verify that the
    GOS-to-binary-maxrl reduction is not accidentally broken at unusual batch
    sizes. Bernoulli(p=0.5) rewards give 0/1 values which are the domain where
    Proposition 3.3 is defined.

    All-same draws (all 0 or all 1) are valid degenerate cases — both functions
    return zeros, which is also consistent with the proposition.
    """
    torch.manual_seed(N + 99)  # unique seed per N
    rewards = torch.bernoulli(torch.ones(N) * 0.5)

    adv_gos = gos_advantage(rewards)
    adv_binary = binary_maxrl_advantage(rewards)

    torch.testing.assert_close(adv_gos, adv_binary, atol=1e-5, rtol=0)
