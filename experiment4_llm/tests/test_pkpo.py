"""PKPO tests — verbatim from PKPO_IMPLEMENTATION.md spec."""
import math
import numpy as np
import pytest
import torch
from itertools import combinations, product
from math import comb

from experiment4_llm.src.advantages import (
    pkpo_advantage,
    _pkpo_s_sorted,
    _pkpo_b_sorted,
)


# ───────────────────────── §4.1 reference oracle ─────────────────────────

def ref_m_normed(N, K, i, j):
    if i == j and i >= K - 1:
        return (
            K / (N - K + 1)
            * np.prod(np.arange(i - K + 2, i + 1) / np.arange(N - K + 2, N + 1))
        )
    if j > i and j >= K - 1 and K >= 2:
        return (
            K / (N - K + 1) * (K - 1) / N
            * np.prod(np.arange(j - K + 2, j) / np.arange(N - K + 2, N))
        )
    return 0.0


def ref_m_diagonal(N, K):
    return np.array([ref_m_normed(N, K, i, i) for i in range(N)])


def ref_deltas(N, K):
    # Listing 1: range(N-2), returns length N-2.
    return np.array([
        ref_m_normed(N - 1, K, i, i + 1) - ref_m_normed(N - 1, K, i + 1, i + 1)
        for i in range(N - 2)
    ])


def ref_s(g, K):
    """Eq. (19), the raw s_i. g must already be sorted ascending."""
    N = len(g)
    c = (g * ref_m_diagonal(N, K)).astype(np.float64)
    if N >= 2:
        # _deltas(N+1, K) has length (N+1)-2 = N-1, matching c[:N-1] and g[1:].
        c[: N - 1] = c[: N - 1] + g[1:] * ref_deltas(N + 1, K)
    return np.cumsum(c[::-1])[::-1]


def ref_b(g, K):
    """Baseline b. g sorted ascending."""
    N = len(g)
    if N <= 1 or K < 1:
        return np.zeros(N)
    w = (ref_m_diagonal(N - 1, K) * np.arange(1, N)).astype(np.float64)
    if N >= 3:
        # _deltas(N, K) has length N-2, matching w[1:] and arange(1, N-1).
        w[1:] = w[1:] + ref_deltas(N, K) * np.arange(1, N - 1)
    c1 = np.array([(w * g[1:]).sum()])
    c2 = (g[:-1] - g[1:]) * w
    return np.cumsum(np.concatenate((c1, c2)))


def ref_max_at_k_brute(g, k):
    N = len(g)
    total = 0.0
    count = 0
    for I in combinations(range(N), k):
        total += max(g[i] for i in I)
        count += 1
    return total / count


# ───────────────────────── §4.2 bit-exact match ─────────────────────────

GRID_NK = [(n, k) for n in [2, 3, 4, 5, 8, 12, 16] for k in range(1, n + 1)]


@pytest.mark.parametrize("N,K", GRID_NK)
def test_s_matches_reference(N, K):
    rng = np.random.default_rng(0xC0FFEE + 1000 * N + K)
    for trial in range(20):
        g = np.sort(rng.standard_normal(N).astype(np.float64))
        ref = ref_s(g, K)
        ours = _pkpo_s_sorted(torch.from_numpy(g), K).numpy()
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-12,
            err_msg=f"s mismatch at N={N},K={K},trial={trial}")


@pytest.mark.parametrize("N,K", GRID_NK)
def test_b_matches_reference(N, K):
    if K < 1:
        pytest.skip()
    rng = np.random.default_rng(0xBADBEEF + 1000 * N + K)
    for trial in range(20):
        g = np.sort(rng.standard_normal(N).astype(np.float64))
        ref = ref_b(g, K)
        ours = _pkpo_b_sorted(torch.from_numpy(g), K).numpy()
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-12,
            err_msg=f"b mismatch at N={N},K={K},trial={trial}")


# ───────────────────────── §4.3 unbiasedness ─────────────────────────

def test_rho_is_unbiased_for_max_at_k():
    """
    Paper Theorem 3: rho^(g)(N, c, K) = (sort(g) * m_diagonal(N,K)).sum()
    is an unbiased estimator of max_g@k. Verify per-sample equality of the
    closed-form rho with the brute-force E[max] over uniform random size-k
    subsets — they must agree EXACTLY (this is Theorem 3, not a MC claim).
    """
    rng = np.random.default_rng(42)
    for N in [3, 4, 6, 8]:
        for K in range(1, N + 1):
            for trial in range(30):
                g = np.sort(rng.standard_normal(N).astype(np.float64))
                # Closed-form rho.
                from experiment4_llm.src.advantages import _pkpo_m_diagonal
                m_diag = _pkpo_m_diagonal(
                    N, K, torch.float64, torch.device("cpu")
                ).numpy()
                rho = (g * m_diag).sum()
                brute = ref_max_at_k_brute(g, K)
                assert abs(rho - brute) < 1e-12, \
                    f"N={N},K={K}: rho={rho} != brute_max@k={brute}"


def test_s_equals_subset_sum_form():
    """
    Eq. (27): s_i = (1/C(n,k)) * sum_{|I|=k, i in I} max_{j in I} g_j.
    Brute-force verification of this closed form against the recursion-
    based _pkpo_s_sorted. This is the load-bearing identity for gradient
    unbiasedness — Theorem 4 reduces to the policy-gradient corollary
    once eq. (27) holds.
    """
    rng = np.random.default_rng(43)
    for N in [3, 4, 5, 6, 8]:
        for K in range(1, N + 1):
            for trial in range(10):
                g = np.sort(rng.standard_normal(N).astype(np.float64))
                # Brute-force subset-sum form.
                Cnk = comb(N, K)
                s_brute = np.zeros(N, dtype=np.float64)
                for I in combinations(range(N), K):
                    max_in_I = max(g[i] for i in I)
                    for i in I:
                        s_brute[i] += max_in_I
                s_brute /= Cnk
                s_ours = _pkpo_s_sorted(torch.from_numpy(g), K).numpy()
                np.testing.assert_allclose(
                    s_ours, s_brute, rtol=0, atol=1e-12,
                    err_msg=f"s subset-sum mismatch N={N},K={K},trial={trial}"
                )


def test_sloo_minus_one_equals_marginal_subset_sum():
    """
    Eq. (33): s_i^(loo-1) = (1/C(n,k)) * sum_{|I|=k, i in I} (max_I - max_{I\\i}).
    Brute-force verification. This is what gives the loo_minus_one variant
    its shift-invariance property (paired differences cancel any constant).
    """
    rng = np.random.default_rng(44)
    for N in [3, 4, 5, 6, 8]:
        for K in range(2, N + 1):  # k >= 2 for loo-1 to be defined
            for trial in range(10):
                g = np.sort(rng.standard_normal(N).astype(np.float64))
                Cnk = comb(N, K)
                s_brute = np.zeros(N, dtype=np.float64)
                for I in combinations(range(N), K):
                    gI = [g[i] for i in I]
                    max_I = max(gI)
                    for i in I:
                        rest = [g[j] for j in I if j != i]
                        max_minus = max(rest)
                        s_brute[i] += (max_I - max_minus)
                s_brute /= Cnk
                # Our impl (without ×G scaling): s - b * k/(k-1)/N
                s_raw = _pkpo_s_sorted(torch.from_numpy(g), K).numpy()
                b = _pkpo_b_sorted(torch.from_numpy(g), K - 1).numpy()
                s_ours = s_raw - b * K / (K - 1) / N
                np.testing.assert_allclose(
                    s_ours, s_brute, rtol=0, atol=1e-12,
                    err_msg=f"s_loo-1 subset-sum mismatch N={N},K={K},trial={trial}"
                )


# ───────────────────────── §4.4 algebraic identities ─────────────────────────

def test_k_equals_1_is_mean_centered_reinforce():
    rng = np.random.default_rng(1)
    for _ in range(50):
        G = int(rng.integers(2, 17))
        r = torch.from_numpy(rng.standard_normal(G))
        adv = pkpo_advantage(r, k_opt=1, variant="loo_minus_one")
        expected = (r - r.mean()) * G
        torch.testing.assert_close(adv, expected, atol=1e-10, rtol=0)


def test_constant_rewards_collapse_loo_minus_one_to_zero():
    """For constant rewards, max_I g - max_{I\\i} g = 0 for k>=2, so
    loo_minus_one yields exactly zero per-element. raw and loo do NOT
    yield zero per-element (they yield a non-zero CONSTANT, which has
    zero expected gradient under Corollary 1 but isn't elementwise zero)."""
    G = 16
    r = torch.full((G,), 0.7, dtype=torch.float64)
    for k in [2, 4, 8, 16]:
        adv = pkpo_advantage(r, k_opt=k, variant="loo_minus_one")
        torch.testing.assert_close(adv, torch.zeros_like(adv), atol=1e-10, rtol=0)


def test_constant_rewards_make_loo_advantages_uniform():
    """For constant rewards, raw and loo advantages must be CONSTANT across
    rollouts (not zero, but all equal). This is the property that ensures
    they contribute zero to the expected policy gradient via Corollary 1."""
    G = 16
    r = torch.full((G,), 0.7, dtype=torch.float64)
    for k in [2, 4, 8, 16]:
        for variant in ["raw", "loo"]:
            adv = pkpo_advantage(r, k_opt=k, variant=variant)
            assert torch.allclose(adv, adv[0].expand_as(adv), atol=1e-10), \
                f"variant={variant}, k={k}: not uniform: {adv}"


def test_permutation_equivariance():
    rng = np.random.default_rng(2)
    G = 16
    for k in [2, 4, 8, 16]:
        r = torch.from_numpy(rng.standard_normal(G))
        perm = torch.randperm(G)
        a1 = pkpo_advantage(r, k_opt=k)[perm]
        a2 = pkpo_advantage(r[perm], k_opt=k)
        torch.testing.assert_close(a1, a2, atol=1e-10, rtol=0)


def test_scale_equivariance():
    rng = np.random.default_rng(3)
    G = 12
    r = torch.from_numpy(rng.standard_normal(G))
    for c in [0.5, 2.0, 100.0]:
        for k in [2, 4, 8, 12]:
            a1 = pkpo_advantage(r * c, k_opt=k)
            a2 = pkpo_advantage(r, k_opt=k) * c
            torch.testing.assert_close(a1, a2, atol=1e-9, rtol=1e-9)


def test_loo_minus_one_is_shift_invariant():
    """sloo_minus_one = (1/C(n,k)) sum_I (max_I - max_{I\\i}). Shifting g
    by a constant c shifts both maxes by c, so the difference cancels.
    This is a PROPERTY of the loo_minus_one estimator, not a bug — it's
    why this variant is the paper's recommended default for unknown reward
    scales."""
    rng = np.random.default_rng(4)
    G = 12
    r = torch.from_numpy(rng.standard_normal(G))
    for k in [2, 4, 8]:
        a1 = pkpo_advantage(r, k_opt=k, variant="loo_minus_one")
        a2 = pkpo_advantage(r + 5.0, k_opt=k, variant="loo_minus_one")
        torch.testing.assert_close(a1, a2, atol=1e-9, rtol=1e-9)


def test_raw_is_NOT_shift_invariant():
    """The raw s_i estimator is NOT shift invariant — it depends on the
    absolute reward scale via the diagonal weights m_ii. If this test
    starts passing, an implementation bug has accidentally introduced
    centering somewhere."""
    rng = np.random.default_rng(4)
    G = 12
    r = torch.from_numpy(rng.standard_normal(G))
    for k in [2, 4, 8]:
        a1 = pkpo_advantage(r, k_opt=k, variant="raw")
        a2 = pkpo_advantage(r + 5.0, k_opt=k, variant="raw")
        assert not torch.allclose(a1, a2, atol=1e-6), \
            f"k={k}: raw pkpo became shift-invariant (this is a bug)"


# ───────────────────────── §4.5 degenerate inputs ─────────────────────────

def test_G_equals_1_returns_zero():
    r = torch.tensor([0.7])
    adv = pkpo_advantage(r, k_opt=1)
    assert adv.shape == (1,)
    assert adv.item() == 0.0


def test_invalid_k_raises():
    r = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        pkpo_advantage(r, k_opt=0)
    with pytest.raises(ValueError):
        pkpo_advantage(r, k_opt=4)
    with pytest.raises(ValueError):
        pkpo_advantage(r, k_opt=2, variant="bogus")


# ───────────────────────── §4.6 G scaling ─────────────────────────

def test_G_scaling_matches_gos_convention():
    rng = np.random.default_rng(5)
    G = 16
    r = torch.from_numpy(rng.choice([0.0, 1.0], size=G).astype(np.float64))
    if r.std() < 1e-9:
        r[0] = 1.0 - r[0]
    pkpo_adv = pkpo_advantage(r, k_opt=1)
    reinforce_adv = (r - r.mean()) * G
    torch.testing.assert_close(pkpo_adv, reinforce_adv, atol=1e-10, rtol=0)


# ───────────────────────── §4.7 verl_register dispatch ─────────────────────────

def test_verl_register_dispatch():
    from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
    for key in ["pkpo_k4", "pkpo_k8", "pkpo_k16"]:
        assert key in CUSTOM_ESTIMATORS, f"{key} not registered"

    fn = CUSTOM_ESTIMATORS["pkpo_k8"]
    B, N, T = 3, 16, 8
    rng = np.random.default_rng(7)
    rewards = torch.zeros(B * N, T, dtype=torch.float64)
    mask = torch.ones(B * N, T, dtype=torch.float64)
    scalar_rewards = torch.from_numpy(rng.standard_normal(B * N))
    rewards[:, -1] = scalar_rewards
    index = np.repeat(np.arange(B), N)

    adv, ret = fn(rewards, mask, index)
    assert adv.shape == (B * N, T)
    assert ret.shape == (B * N, T)
    for i in range(B * N):
        row = adv[i]
        nz = row[mask[i].bool()]
        assert torch.allclose(nz, nz[0])

    for b in range(B):
        group = scalar_rewards[b * N : (b + 1) * N]
        expected = pkpo_advantage(group, k_opt=8, variant="loo_minus_one")
        actual = adv[b * N : (b + 1) * N, -1]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


# ───────────────────────── §4.8 variance ordering ─────────────────────────

def test_loo_minus_one_reduces_variance():
    rng = np.random.default_rng(8)
    G, K = 16, 8
    n_trials = 5000
    raw_norms = []
    loo1_norms = []
    for _ in range(n_trials):
        r = torch.from_numpy(rng.standard_normal(G))
        raw_norms.append(pkpo_advantage(r, k_opt=K, variant="raw").norm().item())
        loo1_norms.append(pkpo_advantage(r, k_opt=K, variant="loo_minus_one").norm().item())
    raw_var = float(np.var(raw_norms))
    loo1_var = float(np.var(loo1_norms))
    assert loo1_var < raw_var, \
        f"loo_minus_one variance {loo1_var:.4f} not less than raw {raw_var:.4f}"


# ───────────────────────── §4.9 end-to-end gradient unbiasedness ─────────────

def test_gradient_matches_analytic_max_at_k_under_discrete_policy():
    """The most important test in this file. Constructs a discrete policy
    parameterized by softmax logits over A actions, computes the analytic
    gradient of max_g@k w.r.t. those logits, and checks that the Monte
    Carlo estimator built from PKPO's s_loo_minus_one matches.

    This is the property that justifies using PKPO as a policy gradient
    estimator at all. If this test fails, no other test matters.
    """
    torch.manual_seed(0)
    A = 5
    r = torch.tensor([0.0, 0.2, 0.5, 0.7, 1.0], dtype=torch.float64)
    theta = torch.randn(A, dtype=torch.float64, requires_grad=True)
    N, K = 8, 3

    def max_at_k_analytic(theta, r, K):
        p = torch.softmax(theta, dim=0)
        order = torch.argsort(r)
        r_sorted = r[order]
        p_sorted = p[order]
        cdf = torch.cumsum(p_sorted, dim=0)
        cdf_K = cdf ** K
        pmf_max = torch.cat([cdf_K[:1], cdf_K[1:] - cdf_K[:-1]])
        return (r_sorted * pmf_max).sum()

    val = max_at_k_analytic(theta, r, K)
    val.backward()
    analytic = theta.grad.detach().clone()

    n_trials = 30_000
    mc_grad = torch.zeros(A, dtype=torch.float64)
    with torch.no_grad():
        p = torch.softmax(theta, dim=0)
        for _ in range(n_trials):
            actions = torch.multinomial(p, N, replacement=True)
            rewards_sample = r[actions]
            sorted_r, sort_idx = torch.sort(rewards_sample)
            s_raw = _pkpo_s_sorted(sorted_r, K)
            b = _pkpo_b_sorted(sorted_r, K - 1)
            s = s_raw - b * K / (K - 1) / N
            s_orig = torch.zeros_like(rewards_sample)
            s_orig[sort_idx] = s
            for i in range(N):
                a = actions[i].item()
                grad_log_pi = -p.clone()
                grad_log_pi[a] += 1.0
                mc_grad += s_orig[i].item() * grad_log_pi
    mc_grad /= n_trials

    cos = torch.nn.functional.cosine_similarity(
        analytic.unsqueeze(0), mc_grad.unsqueeze(0)
    ).item()
    max_diff = (analytic - mc_grad).abs().max().item()
    assert cos > 0.999, f"cosine={cos:.6f}; analytic={analytic} mc={mc_grad}"
    assert max_diff < 0.005, f"max abs diff {max_diff:.4e} too large"
