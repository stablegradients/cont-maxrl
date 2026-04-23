"""Focused tests for the fixed K&K reward (spec §3.2).

Covers the anti-hack check, the simplified matching formula, and parsing
robustness via reasoning_gym's ``_normalize_answer``.
"""
from __future__ import annotations

import json

import pytest
import reasoning_gym
from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset

from experiment10_reasoning_gym.src.reward_manager import (
    _knk_fixed_score,
    compute_score_continuous,
    compute_score_binary,
)


def _ei(task, entry):
    return {"task": task, "difficulty": 0, "entry": json.dumps(entry)}


# ---------------------------------------------------------------------------
# Matching formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_people,seed", [(2, 0), (3, 1), (4, 2), (5, 3)])
def test_knk_perfect_oracle_scores_one(n_people, seed):
    ds = reasoning_gym.create_dataset("knights_knaves", size=2, seed=seed, n_people=n_people)
    entry = ds[0]
    assert _knk_fixed_score(entry["answer"], entry) == pytest.approx(1.0)


def test_knk_empty_answer_returns_zero():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    assert _knk_fixed_score("", entry) == 0.0
    assert _knk_fixed_score("   ", entry) == 0.0


def test_knk_non_string_answer_returns_zero():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    assert _knk_fixed_score(None, entry) == 0.0
    assert _knk_fixed_score(12345, entry) == 0.0  # type: ignore[arg-type]


def test_knk_anti_hack_too_many_assignments_returns_zero():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    names = entry["metadata"]["names"]
    terms = entry["metadata"]["knight_knave_terms"]
    knight_term, knave_term = terms["knight"], terms["knave"]
    # Enumerate every (name, knight) AND every (name, knave) — 2N distinct pairs
    hack_parts = []
    for name in names:
        hack_parts.append(f"{name} is a {knight_term}")
        hack_parts.append(f"{name} is a {knave_term}")
    hack = ", ".join(hack_parts)
    # 2N > N, so spec's anti-hack kicks in -> 0
    assert _knk_fixed_score(hack, entry) == 0.0


def test_knk_anti_hack_too_few_assignments_returns_zero():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    oracle = KnightsKnavesDataset._normalize_answer(entry["answer"])
    # Submit only 2 of 3 assignments — fewer than oracle -> 0 (spec's strict check)
    subset_parts = []
    for i, (name, role) in enumerate(list(oracle)[:2]):
        subset_parts.append(f"{name} is a {role}")
    assert _knk_fixed_score(", ".join(subset_parts), entry) == 0.0


def test_knk_partial_credit_is_exact_fraction():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=4)
    entry = ds[0]
    oracle = list(KnightsKnavesDataset._normalize_answer(entry["answer"]))
    assert len(oracle) == 4
    terms = entry["metadata"]["knight_knave_terms"]
    knight_term, knave_term = terms["knight"], terms["knave"]
    # Flip exactly one role, keep N=4 assignments
    mutated = []
    for i, (name, role) in enumerate(oracle):
        if i == 0:
            new_role = knave_term if role == knight_term else knight_term
            mutated.append(f"{name} is a {new_role}")
        else:
            mutated.append(f"{name} is a {role}")
    score = _knk_fixed_score(", ".join(mutated), entry)
    assert score == pytest.approx(3 / 4, abs=1e-6)


def test_knk_ordering_does_not_matter():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    oracle = list(KnightsKnavesDataset._normalize_answer(entry["answer"]))
    # Reverse order of assignments — score should still be 1
    reversed_ans = ", ".join(f"{name} is a {role}" for name, role in reversed(oracle))
    assert _knk_fixed_score(reversed_ans, entry) == pytest.approx(1.0)


def test_knk_punctuation_tolerance():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    # Use the built-in normalizer-friendly phrasing with extra punctuation/spacing
    oracle = list(KnightsKnavesDataset._normalize_answer(entry["answer"]))
    messy = ". ".join(f"{name} is a {role}" for name, role in oracle) + "."
    assert _knk_fixed_score(messy, entry) == pytest.approx(1.0)


def test_knk_binary_equals_perfect_indicator():
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    ans_perfect = f"<answer>{entry['answer']}</answer>"
    b = compute_score_binary(ans_perfect, None, None, _ei("knights_knaves", entry))
    c = compute_score_continuous(ans_perfect, None, None, _ei("knights_knaves", entry))
    assert b["binary"] == 1.0 and c["binary"] == 1.0
    assert b["score"] == 1.0 and c["score"] == 1.0


def test_knk_duplicate_assignments_collapse_via_set():
    """Spec §3.2 uses set semantics: duplicates collapse before comparison."""
    ds = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42, n_people=3)
    entry = ds[0]
    oracle = list(KnightsKnavesDataset._normalize_answer(entry["answer"]))
    # Emit 6 assignments where each appears twice -> collapses to 3 unique tuples
    doubled_parts = []
    for name, role in oracle:
        doubled_parts.append(f"{name} is a {role}")
        doubled_parts.append(f"{name} is a {role}")
    # After normalization, pred set = oracle set (both size N), so score = 1.0.
    # This verifies the use of set semantics rather than multiset / list.
    assert _knk_fixed_score(", ".join(doubled_parts), entry) == pytest.approx(1.0)
