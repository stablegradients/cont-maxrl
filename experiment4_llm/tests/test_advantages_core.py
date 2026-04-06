"""Core tests for per-group advantage functions (single-prompt, N rollouts)."""
import torch
import pytest


# ── Worked Example ──


def test_maxrl_unnorm_worked_example():
    """Verify against the worked example in spec section 4.3."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.tensor([0.3, 0.9, 0.1, 0.6])
    adv = maxrl_unnormalized_advantage(rewards)

    # From worked example (scaled by G=4 to prevent gradient magnitude decay):
    # Result: [-0.5333, 1.2667, -0.8000, 0.0667]
    expected = torch.tensor([-0.5333, 1.2667, -0.8000, 0.0667])
    assert torch.allclose(adv, expected, atol=1e-3), f"Got {adv}"


def test_maxrl_unnorm_worked_example_exact():
    """Verify exact intermediate values from spec section 4.3."""
    rewards = torch.tensor([0.3, 0.9, 0.1, 0.6])
    sorted_r, sort_idx = torch.sort(rewards)

    # Verify sort
    assert torch.allclose(sorted_r, torch.tensor([0.1, 0.3, 0.6, 0.9]))
    assert sort_idx.tolist() == [2, 0, 3, 1]

    # Verify gaps
    prev = torch.cat([torch.zeros(1), sorted_r[:-1]])
    gaps = sorted_r - prev
    assert torch.allclose(gaps, torch.tensor([0.1, 0.2, 0.3, 0.3]), atol=1e-6)

    # Verify survivors
    survivors = torch.arange(4, 0, -1, dtype=torch.float)
    assert survivors.tolist() == [4.0, 3.0, 2.0, 1.0]

    # Verify weights
    w = torch.cumsum(gaps / survivors, dim=0)
    expected_w = torch.tensor([0.025, 0.0917, 0.2417, 0.5417])
    assert torch.allclose(w, expected_w, atol=1e-3)


# ── Algebraic Properties ──


@pytest.mark.parametrize("G", [2, 4, 8, 16, 32])
def test_maxrl_unnorm_mean_centered(G):
    """Advantages must sum to approximately zero for all group sizes."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    for _ in range(50):
        rewards = torch.rand(G)
        adv = maxrl_unnormalized_advantage(rewards)
        assert abs(adv.sum().item()) < 1e-5, f"G={G}, sum={adv.sum()}"


@pytest.mark.parametrize("G", [2, 4, 8, 16, 32])
def test_maxrl_norm_mean_centered(G):
    """Normalized advantages must also sum to approximately zero."""
    from experiment4_llm.src.advantages import maxrl_normalized_advantage

    for _ in range(50):
        rewards = torch.rand(G)
        adv = maxrl_normalized_advantage(rewards)
        assert abs(adv.sum().item()) < 1e-5, f"G={G}, sum={adv.sum()}"


@pytest.mark.parametrize("G", [2, 4, 8, 16, 32])
def test_reinforce_mean_centered(G):
    """REINFORCE+baseline advantages sum to zero."""
    from experiment4_llm.src.advantages import reinforce_baseline_advantage

    for _ in range(50):
        rewards = torch.rand(G)
        adv = reinforce_baseline_advantage(rewards)
        assert abs(adv.sum().item()) < 1e-5, f"G={G}, sum={adv.sum()}"


@pytest.mark.parametrize("G", [2, 4, 8, 16, 32])
def test_maxrl_unnorm_monotonic(G):
    """Higher reward -> higher advantage (strict monotonicity)."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    for _ in range(50):
        rewards = torch.rand(G)
        adv = maxrl_unnormalized_advantage(rewards)
        sorted_r, sort_idx = torch.sort(rewards)
        sorted_adv = adv[sort_idx]
        diffs = sorted_adv[1:] - sorted_adv[:-1]
        assert (diffs >= -1e-6).all(), f"Non-monotonic: G={G}, diffs={diffs}"


@pytest.mark.parametrize("G", [2, 4, 8, 16, 32])
def test_maxrl_norm_monotonic(G):
    """Normalized variant also monotonic."""
    from experiment4_llm.src.advantages import maxrl_normalized_advantage

    for _ in range(50):
        rewards = torch.rand(G)
        adv = maxrl_normalized_advantage(rewards)
        sorted_r, sort_idx = torch.sort(rewards)
        sorted_adv = adv[sort_idx]
        diffs = sorted_adv[1:] - sorted_adv[:-1]
        assert (diffs >= -1e-6).all(), f"Non-monotonic: G={G}"


def test_maxrl_norm_unit_variance():
    """Normalized advantages should have std ~= 1 (when rewards are not all-equal)."""
    from experiment4_llm.src.advantages import maxrl_normalized_advantage

    for _ in range(100):
        rewards = torch.rand(16)
        adv = maxrl_normalized_advantage(rewards)
        std = adv.std(unbiased=False).item()
        if rewards.std() > 1e-6:  # skip degenerate case
            assert abs(std - 1.0) < 0.01, f"std={std}, expected ~1.0"


# ── Edge Cases ──


def test_rloo_advantage_basic():
    """RLOO advantage = G/(G-1) * (r_i - mean(r)) * G (scaled for PPO token-mean loss)."""
    from experiment4_llm.src.advantages import rloo_advantage
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = rloo_advantage(rewards)
    G = 4
    mean = rewards.mean()  # 2.5
    expected = (G / (G - 1)) * (rewards - mean) * G
    assert torch.allclose(adv, expected, atol=1e-6)


def test_rloo_advantage_mean_zero():
    """RLOO advantages sum to zero across the group."""
    from experiment4_llm.src.advantages import rloo_advantage
    rewards = torch.rand(16)
    adv = rloo_advantage(rewards)
    assert abs(adv.sum().item()) < 1e-4


def test_rloo_advantage_single_sample():
    """With G=1, no leave-one-out is possible; return zeros."""
    from experiment4_llm.src.advantages import rloo_advantage
    rewards = torch.tensor([3.14])
    adv = rloo_advantage(rewards)
    assert torch.allclose(adv, torch.zeros(1))


@pytest.mark.parametrize("fn_name", [
    "maxrl_unnormalized_advantage",
    "maxrl_normalized_advantage",
    "reinforce_baseline_advantage",
])
def test_all_equal_rewards(fn_name):
    """All equal rewards -> all zero advantages."""
    from experiment4_llm.src import advantages
    fn = getattr(advantages, fn_name)

    for val in [0.0, 0.5, 1.0]:
        rewards = torch.full((8,), val)
        adv = fn(rewards)
        assert torch.allclose(adv, torch.zeros(8), atol=1e-7), \
            f"{fn_name}(all {val}) = {adv}"


@pytest.mark.parametrize("fn_name", [
    "maxrl_unnormalized_advantage",
    "maxrl_normalized_advantage",
    "reinforce_baseline_advantage",
])
def test_single_reward(fn_name):
    """G=1 -> advantage = 0."""
    from experiment4_llm.src import advantages
    fn = getattr(advantages, fn_name)

    rewards = torch.tensor([0.7])
    adv = fn(rewards)
    assert adv.shape == (1,)
    assert abs(adv.item()) < 1e-7


def test_two_rewards():
    """G=2: simplest non-trivial case."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.tensor([0.0, 1.0])
    adv = maxrl_unnormalized_advantage(rewards)
    # r_(1)=0.0, r_(2)=1.0
    # gaps=[0.0, 1.0], survivors=[2,1]
    # w = cumsum([0/2, 1/1]) = [0, 1], scaled by G=2: [0, 2], mean=1.0
    # centered = [-1.0, 1.0]
    expected = torch.tensor([-1.0, 1.0])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_identical_rewards_in_group():
    """Some rewards tied: should not produce NaN or errors."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.tensor([0.5, 0.5, 0.5, 0.8, 0.8, 1.0])
    adv = maxrl_unnormalized_advantage(rewards)
    assert not torch.isnan(adv).any(), f"NaN in advantages: {adv}"
    assert abs(adv.sum().item()) < 1e-5


# ── MaxRL amplifies top reward more than GRPO ──


def test_maxrl_amplifies_top_more_than_grpo():
    """Key property: MaxRL gives top reward higher relative advantage than GRPO."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    maxrl_adv = maxrl_unnormalized_advantage(rewards)
    grpo_adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

    # MaxRL should give more weight to the top sample relative to its magnitude
    # Ratio: advantage_top / advantage_second_top should be larger for MaxRL
    maxrl_ratio = maxrl_adv[-1] / maxrl_adv[-2]
    grpo_ratio = grpo_adv[-1] / grpo_adv[-2]
    assert maxrl_ratio > grpo_ratio, \
        f"MaxRL ratio {maxrl_ratio:.3f} should exceed GRPO ratio {grpo_ratio:.3f}"


# ── Dtype and Device ──


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_maxrl_dtypes(dtype):
    """Should work with different float dtypes."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.rand(16, dtype=dtype)
    adv = maxrl_unnormalized_advantage(rewards)
    assert adv.dtype == dtype
    assert abs(adv.sum().item()) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_maxrl_cuda():
    """Should work on GPU."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.rand(16, device="cuda")
    adv = maxrl_unnormalized_advantage(rewards)
    assert adv.device.type == "cuda"
    assert abs(adv.sum().item()) < 1e-4


# ── Stress test ──


def test_maxrl_large_group():
    """G=128: should work and maintain properties."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.rand(128)
    adv = maxrl_unnormalized_advantage(rewards)
    assert adv.shape == (128,)
    assert abs(adv.sum().item()) < 1e-4
    # Monotonicity
    sorted_r, sort_idx = torch.sort(rewards)
    sorted_adv = adv[sort_idx]
    assert (sorted_adv[1:] - sorted_adv[:-1] >= -1e-5).all()


# ── Determinism ──


def test_maxrl_deterministic():
    """Same input -> same output."""
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage

    rewards = torch.tensor([0.1, 0.4, 0.7, 0.2, 0.9])
    adv1 = maxrl_unnormalized_advantage(rewards.clone())
    adv2 = maxrl_unnormalized_advantage(rewards.clone())
    assert torch.equal(adv1, adv2)


def test_dg_advantage_per_token_basic():
    """DG gate: suppresses rare+negative U (blunders), amplifies rare+positive U (breakthroughs)."""
    from experiment4_llm.src.advantages import dg_advantage_per_token

    G, T = 4, 8
    rewards = torch.tensor([1.0, 0.5, 0.0, 0.0])
    # U = rewards - mean = [0.625, 0.125, -0.375, -0.375]
    # Rollout 0: rare (high surprisal) + positive U -> breakthrough, gate open
    # Rollout 2: rare (high surprisal) + negative U -> blunder, gate closed
    # Rollouts 1, 3: common (low surprisal) -> gate ~= 0.5
    old_log_probs = torch.zeros(G, T)
    old_log_probs[0] = -5.0
    old_log_probs[1] = -0.5
    old_log_probs[2] = -5.0
    old_log_probs[3] = -0.5
    mask = torch.ones(G, T)

    adv = dg_advantage_per_token(
        rewards, old_log_probs, mask, eta=1.0, scale_by_G=True
    )
    assert adv.shape == (G, T)

    # Breakthrough: sigmoid(0.625 * 5 / 1) * 0.625 * G = sigmoid(3.125) * 2.5
    # ~= 0.958 * 2.5 ~= 2.395
    assert adv[0, 0].item() > 2.0

    # Blunder: sigmoid(-0.375 * 5) * -0.375 * G = sigmoid(-1.875) * -1.5
    # ~= 0.133 * -1.5 ~= -0.2
    assert abs(adv[2, 0].item()) < 0.5

    # Common + positive U, low surprisal: gate ~ sigmoid(~0) ~ 0.5
    # coef ~= 0.5 * 0.125 * 4 = 0.25
    baseline_mag = abs(rewards[1].item() - rewards.mean().item()) * G  # 0.5
    assert 0.3 * baseline_mag < abs(adv[1, 0].item()) < 0.7 * baseline_mag


def test_dg_advantage_masking():
    """Padded tokens receive zero advantage."""
    from experiment4_llm.src.advantages import dg_advantage_per_token

    G, T = 3, 6
    rewards = torch.tensor([1.0, 0.5, 0.0])
    old_log_probs = torch.full((G, T), -1.0)
    mask = torch.ones(G, T)
    mask[:, 4:] = 0  # last 2 tokens are pads

    adv = dg_advantage_per_token(rewards, old_log_probs, mask, eta=1.0)
    assert (adv[:, 4:] == 0).all(), "padded positions must be zero"
    assert not (adv[:, :4] == 0).all(), "unpadded positions must be nonzero"


def test_dg_advantage_all_equal_rewards():
    """If all rewards are equal, U = 0, coef = 0."""
    from experiment4_llm.src.advantages import dg_advantage_per_token

    G, T = 4, 6
    rewards = torch.ones(G) * 0.5
    old_log_probs = torch.full((G, T), -2.0)
    mask = torch.ones(G, T)

    adv = dg_advantage_per_token(rewards, old_log_probs, mask, eta=1.0)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_dg_advantage_G_one():
    """G=1 edge case: returns zeros (no group baseline possible)."""
    from experiment4_llm.src.advantages import dg_advantage_per_token

    rewards = torch.tensor([0.7])
    old_log_probs = torch.full((1, 5), -1.0)
    mask = torch.ones(1, 5)
    adv = dg_advantage_per_token(rewards, old_log_probs, mask)
    assert adv.shape == (1, 5)
    assert torch.allclose(adv, torch.zeros_like(adv))
