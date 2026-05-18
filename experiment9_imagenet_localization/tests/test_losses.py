"""Tests for experiment9_imagenet_localization.losses.

Each test has a docstring stating the invariant being checked and why failure matters.
Hand-computed expected values are derived in comments directly above assertions.

All tests run on CPU and complete well under 10 seconds.

Test count: 10 tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiment9_imagenet_localization.losses import (
    HEAD_NAMES,
    factored_sample_log_prob,
    localization_exp_iou_ord_ce_loss,
    localization_mse_loss,
    localization_ordinal_ce_loss,
    localization_plain_ce_loss,
    ordinal_ce_single_head,
)


# ===========================================================================
# exp_iou_ord_ce tests (CPU-safe, small K)
# ===========================================================================

def _random_logits(B, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {h: torch.randn(B, K, generator=g) for h in HEAD_NAMES}


def test_exp_iou_ord_ce_runs_and_is_positive():
    """Basic smoke: shape OK, finite, positive. K=5, B=2, M=3, T=10 (small for CPU)."""
    B, K = 2, 5
    logits = _random_logits(B, K)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.4, 0.4], [0.2, 0.3, 0.1, 0.2], [0, 0, 0, 0]],
        [[0.3, 0.7, 0.5, 0.3], [0, 0, 0, 0],          [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, True, False], [True, False, False]])
    loss = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    assert loss.dim() == 0, "loss must be a scalar"
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    assert loss.item() >= 0.0, f"loss must be nonneg, got {loss}"


def test_exp_iou_ord_ce_gradient_flows_to_all_heads():
    """Gradient must flow back to every one of the 4 head logits.

    If any head gets zero gradient we've broken the factored joint product.
    """
    B, K = 2, 5
    logits = {h: torch.randn(B, K, requires_grad=True) for h in HEAD_NAMES}
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.3, 0.3], [0, 0, 0, 0]],
        [[0.2, 0.3, 0.4, 0.5], [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, False], [True, False]])
    loss = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    loss.backward()
    for h in HEAD_NAMES:
        g = logits[h].grad
        assert g is not None, f"head {h!r}: no gradient"
        assert g.abs().sum().item() > 0, f"head {h!r}: zero gradient"


def test_exp_iou_ord_ce_lower_loss_when_policy_matches_gt():
    """Policy concentrating on the correct bins must yield lower loss than uniform.

    At K=5, if GT is at x_c=0.5 (bin 2), y_c=0.5 (bin 2), w=0.4 (bin 1), h=0.4 (bin 1),
    a policy that puts all mass on bin (2,2,1,1) should get near-zero loss
    vs a uniform policy which gets a much larger loss.
    """
    B, K = 1, 5
    gt_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0, 0, 0, 0]]])
    gt_mask = torch.tensor([[True, False]])

    # Peaked policy on the correct bins
    peaked = {h: torch.full((B, K), -10.0) for h in HEAD_NAMES}
    peaked['x_c'][0, 2] = 10.0
    peaked['y_c'][0, 2] = 10.0
    peaked['w'][0, 1]   = 10.0   # w=0.3 (bin 1 center)
    peaked['h'][0, 1]   = 10.0
    loss_peaked = localization_exp_iou_ord_ce_loss(
        peaked, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )

    # Uniform policy
    uniform = {h: torch.zeros(B, K) for h in HEAD_NAMES}
    loss_uniform = localization_exp_iou_ord_ce_loss(
        uniform, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )

    assert loss_peaked.item() < loss_uniform.item() - 0.1, (
        f"peaked loss {loss_peaked.item():.4f} not meaningfully below "
        f"uniform {loss_uniform.item():.4f} — loss does not reward correct bin mass"
    )


def test_exp_iou_ord_ce_handles_invalid_gt_rows():
    """gt_mask should correctly ignore padded GT rows (all-zeros boxes).

    Loss with a real GT + padding rows should equal loss with just the real GT.
    """
    B, K = 1, 5
    logits = _random_logits(B, K, seed=42)
    gt_boxes_padded = torch.tensor([[[0.4, 0.5, 0.3, 0.3], [0, 0, 0, 0], [0.9, 0.9, 0.1, 0.1]]])
    mask_pad1 = torch.tensor([[True, False, False]])   # only real GT
    mask_pad2 = torch.tensor([[True, False, True]])    # real + garbage at idx 2

    loss_just_real = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes_padded[:, :1], torch.tensor([[True]]), K=K, n_thresholds=10,
    )
    loss_masked = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes_padded, mask_pad1, K=K, n_thresholds=10,
    )
    assert abs(loss_just_real.item() - loss_masked.item()) < 1e-4, (
        "Padded invalid rows are being counted despite gt_mask=False"
    )


def test_exp_iou_ord_ce_clamp_pred_false_matches_baseline():
    """clamp_pred=False must produce bit-identical output to omitting the
    kwarg, so the existing exp_iou_ord_ce baseline is unaffected."""
    B, K = 2, 5
    logits = _random_logits(B, K, seed=7)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.4, 0.4], [0.2, 0.3, 0.1, 0.2], [0, 0, 0, 0]],
        [[0.3, 0.7, 0.5, 0.3], [0, 0, 0, 0],          [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, True, False], [True, False, False]])
    base = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    explicit = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=False,
    )
    assert torch.equal(base, explicit), (
        f"clamp_pred=False must not change behavior; got {base} vs {explicit}"
    )


def test_exp_iou_ord_ce_clamp_pred_true_finite_grad_and_differs():
    """clamp_pred=True must return finite/positive loss with grad to all heads
    and must differ from clamp_pred=False on logits where the predicted box
    can extend outside [0,1]."""
    B, K = 2, 5
    g = torch.Generator().manual_seed(11)
    logits = {h: torch.randn(B, K, generator=g, requires_grad=True)
              for h in HEAD_NAMES}
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.6, 0.6], [0, 0, 0, 0]],
        [[0.2, 0.8, 0.5, 0.3], [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, False], [True, False]])

    loss_unclamped = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=False,
    )
    loss_clamped = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=True,
    )

    assert torch.isfinite(loss_clamped), f"clamped loss not finite: {loss_clamped}"
    assert loss_clamped.item() >= 0.0, f"clamped loss must be nonneg, got {loss_clamped}"
    assert not torch.allclose(loss_unclamped, loss_clamped), (
        f"clamp_pred=True did not change anything: "
        f"unclamped={loss_unclamped.item()} clamped={loss_clamped.item()}"
    )
    loss_clamped.backward()
    for h in HEAD_NAMES:
        grad = logits[h].grad
        assert grad is not None, f"no grad reached head {h!r}"
        assert torch.isfinite(grad).all(), f"non-finite grad on head {h!r}: {grad}"
        assert grad.abs().sum().item() > 0.0, f"zero grad on head {h!r}"


# ===========================================================================
# Factored log-prob tests
# ===========================================================================


def test_factored_sum_shape_is_BN_not_BN4():
    """Output of factored_sample_log_prob must be (B, N), not (B, N, 4).

    The #1 trap: gathering per-head log-probs without summing leaves a (B, N, 4)
    tensor that silently broadcasts-wrong in all downstream policy-gradient losses.
    A (B, N, 4) shape would make the REINFORCE loss sum over the last dim instead
    of the sample dim, producing a completely wrong gradient signal.
    """
    B, K, N = 2, 4, 5
    torch.manual_seed(0)
    log_probs = {h: torch.log_softmax(torch.randn(B, K), dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.randint(0, K, (B, N)) for h in HEAD_NAMES}

    out = factored_sample_log_prob(log_probs, samples)

    assert out.shape == (B, N), (
        f"Expected shape ({B}, {N}), got {out.shape}. "
        "The function must SUM across heads, not stack them."
    )


def test_factored_sum_equals_sum_of_gather():
    """factored_sample_log_prob must equal the hand-computed sum of four gathers.

    This verifies the actual arithmetic: gather each head's log-prob at the
    sampled index and sum. Any indexing error (e.g., wrong dim, wrong gather
    axis) would produce a different numerical value here.
    """
    B, K, N = 3, 6, 4
    torch.manual_seed(1)
    log_probs = {h: torch.log_softmax(torch.randn(B, K), dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.randint(0, K, (B, N)) for h in HEAD_NAMES}

    # Hand-compute: sum of per-head gather results
    expected = sum(log_probs[h].gather(1, samples[h]) for h in HEAD_NAMES)  # (B, N)

    out = factored_sample_log_prob(log_probs, samples)

    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_single_sample_per_image():
    """factored_sample_log_prob with N=1 must return shape (B, 1).

    The single-sample case (N=1) is the common inference path. A shape of (B,)
    instead of (B, 1) would break broadcasting in the policy gradient loss.
    """
    B, K, N = 3, 8, 1
    torch.manual_seed(2)
    log_probs = {h: torch.log_softmax(torch.randn(B, K), dim=-1) for h in HEAD_NAMES}
    samples = {h: torch.randint(0, K, (B, N)) for h in HEAD_NAMES}

    out = factored_sample_log_prob(log_probs, samples)

    assert out.shape == (B, N), (
        f"Expected shape ({B}, {N}), got {out.shape}."
    )


# ===========================================================================
# Supervised loss tests
# ===========================================================================


def test_ordinal_ce_reduces_to_mean_of_4_head_losses():
    """localization_ordinal_ce_loss must equal the mean of 4 per-head ordinal CE losses.

    This checks the aggregation logic: the multi-head function must divide the
    sum of per-head losses by 4.0, not just return the sum. A sum-only return
    would inflate the loss scale by 4× and cause gradient magnitude issues.
    """
    B, K = 5, 8
    torch.manual_seed(3)
    logits = {h: torch.randn(B, K) for h in HEAD_NAMES}
    targets = {h: torch.randint(0, K, (B,)) for h in HEAD_NAMES}

    multi = localization_ordinal_ce_loss(logits, targets, K)
    per_head_mean = sum(
        ordinal_ce_single_head(logits[h], targets[h], K) for h in HEAD_NAMES
    ) / 4.0

    torch.testing.assert_close(multi, per_head_mean, atol=1e-6, rtol=0)


def test_perfectly_peaked_logits_small_loss():
    """Logits [100, 0, 0, 0] with target 0 must give near-zero ordinal CE loss.

    When the model is completely confident and correct, the softmax places
    virtually all mass on bin 0, making W_d ≈ 1 for all d, so -log W_d ≈ 0.
    A non-zero loss here would indicate numerical errors in the log computation.
    """
    B, K = 4, 4
    # logits[0] = 100, rest = 0; target = 0 for every sample in every head
    logits = {h: torch.tensor([[100.0, 0.0, 0.0, 0.0]] * B) for h in HEAD_NAMES}
    targets = {h: torch.zeros(B, dtype=torch.long) for h in HEAD_NAMES}

    loss = localization_ordinal_ce_loss(logits, targets, K)

    assert loss.item() < 1e-6, (
        f"Expected near-zero loss for perfectly peaked logits, got {loss.item():.2e}"
    )


def test_uniform_logits_finite_loss():
    """Uniform logits must produce a finite, positive ordinal CE loss.

    With uniform logits (all zeros), softmax gives equal probability to every
    bin. The loss must be finite (not Inf/NaN) and positive. Inf would indicate
    a -log(0) numerical issue; NaN would indicate divide-by-zero somewhere.
    """
    B, K = 4, 4
    logits = {h: torch.zeros(B, K) for h in HEAD_NAMES}
    targets = {h: torch.ones(B, dtype=torch.long) for h in HEAD_NAMES}  # target = 1

    loss = localization_ordinal_ce_loss(logits, targets, K)

    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


def test_plain_ce_equals_mean_of_f_cross_entropy():
    """localization_plain_ce_loss must equal mean([F.cross_entropy(logits[h], targets[h])]).

    This verifies the aggregation formula: sum divided by 4. If the function
    returns the raw sum, the loss would be 4× too large and destabilise training.
    """
    B, K = 6, 10
    torch.manual_seed(4)
    logits = {h: torch.randn(B, K) for h in HEAD_NAMES}
    targets = {h: torch.randint(0, K, (B,)) for h in HEAD_NAMES}

    multi = localization_plain_ce_loss(logits, targets)
    expected = sum(F.cross_entropy(logits[h], targets[h]) for h in HEAD_NAMES) / 4.0

    torch.testing.assert_close(multi, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ordinal_ce_equals_plain_ce_at_K2(seed):
    """At K=2, ordinal_ce_single_head must equal F.cross_entropy exactly.

    At K=2 there is only one threshold (d=0). W_0(b, y) = probs[b, y].
    So the ordinal CE = mean_b[-log probs[b, y]] = F.cross_entropy.
    The (K-1)=1 normalization factor is a no-op.

    This is the binary-recovery identity — if it fails, the ordinal loss is
    not a proper generalization of standard CE, and the K=2 baseline is invalid.
    """
    torch.manual_seed(seed * 17 + 5)
    B, K = 8, 2
    logits = torch.randn(B, K)
    targets = torch.randint(0, K, (B,))

    ordinal_loss = ordinal_ce_single_head(logits, targets, K)
    plain_loss = F.cross_entropy(logits, targets)

    torch.testing.assert_close(ordinal_loss, plain_loss, atol=1e-5, rtol=0)


# ===========================================================================
# MSE baseline tests
# ===========================================================================


def test_mse_zero_when_pred_matches_target():
    """localization_mse_loss must be ~0 when pred exactly equals gt_primary_box.

    If the predicted box is identical to the GT box, the MSE is defined to be 0.
    Any non-zero value indicates an implementation error (e.g., computing loss
    against the wrong tensor or adding an unintended offset).
    """
    B = 5
    torch.manual_seed(6)
    gt = torch.rand(B, 4)
    pred = gt.clone()

    loss = localization_mse_loss(pred, gt)

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_mse_positive_when_pred_differs():
    """localization_mse_loss must be strictly positive when pred != gt_primary_box.

    MSE is only zero when inputs are identical; any difference must produce a
    strictly positive value. A zero loss on mismatched inputs would silently
    suppress all gradient updates to the regression head.
    """
    B = 5
    torch.manual_seed(7)
    gt = torch.rand(B, 4)
    pred = gt + 0.1  # perturb by a fixed offset

    loss = localization_mse_loss(pred, gt)

    assert loss.item() > 0.0, (
        f"Expected positive MSE for perturbed predictions, got {loss.item()}"
    )
