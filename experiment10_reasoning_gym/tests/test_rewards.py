"""Unit tests for the reasoning_gym reward manager (spec §11.2 / §12)."""
from __future__ import annotations

import json

import pytest
import reasoning_gym

from experiment10_reasoning_gym.src.reward_manager import (
    extract_answer,
    compute_score_continuous,
    compute_score_binary,
    _knk_fixed_score,
)


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


def test_extract_answer_present():
    assert extract_answer("junk <answer>foo</answer> trail") == "foo"


def test_extract_answer_absent_returns_full_stripped():
    assert extract_answer("  hello world  ") == "hello world"


def test_extract_answer_multiline():
    s = "pre\n<answer>\nA, B, C\n</answer>\npost"
    assert extract_answer(s) == "A, B, C"


def test_extract_answer_none():
    assert extract_answer(None) == ""


# ---------------------------------------------------------------------------
# helpers for constructing extra_info payloads
# ---------------------------------------------------------------------------


def _ei(task: str, entry: dict, difficulty: int = 0) -> dict:
    return {"task": task, "difficulty": difficulty, "entry": json.dumps(entry)}


# ---------------------------------------------------------------------------
# word_sorting
# ---------------------------------------------------------------------------


def test_word_sorting_perfect():
    ds = reasoning_gym.create_dataset("word_sorting", size=2, seed=42,
                                      min_words=4, max_words=4)
    entry = ds[0]
    ans = f"<answer>{entry['answer']}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("word_sorting", entry, 4))
    assert out["score"] == pytest.approx(1.0)
    assert out["continuous"] == pytest.approx(1.0)
    assert out["binary"] == 1.0
    assert out["format_ok"] == 1.0


def test_word_sorting_partial_gets_partial_continuous():
    ds = reasoning_gym.create_dataset("word_sorting", size=2, seed=42,
                                      min_words=4, max_words=4)
    entry = ds[0]
    words = entry["metadata"]["sorted_words"]
    # First two correct, then junk
    partial = ", ".join(words[:2] + ["ZZZ", "ZZZ"])
    ans = f"<answer>{partial}</answer>"
    out_c = compute_score_continuous(ans, None, None, _ei("word_sorting", entry))
    out_b = compute_score_binary(ans, None, None, _ei("word_sorting", entry))
    assert 0.0 < out_c["continuous"] < 1.0
    assert out_b["score"] == 0.0  # binary: only perfect counts
    assert out_c["binary"] == 0.0


def test_word_sorting_no_answer_tags_still_scorable():
    """Falls back to full response — should not crash and should score low."""
    ds = reasoning_gym.create_dataset("word_sorting", size=2, seed=42,
                                      min_words=4, max_words=4)
    entry = ds[0]
    out = compute_score_continuous("totally unrelated text", None, None,
                                   _ei("word_sorting", entry))
    assert out["continuous"] == 0.0
    assert out["format_ok"] == 0.0


# ---------------------------------------------------------------------------
# sentence_reordering
# ---------------------------------------------------------------------------


def test_sentence_reordering_perfect():
    ds = reasoning_gym.create_dataset("sentence_reordering", size=2, seed=42)
    entry = ds[0]
    ans = f"<answer>{entry['answer']}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("sentence_reordering", entry))
    assert out["continuous"] == pytest.approx(1.0)
    assert out["binary"] == 1.0


def test_sentence_reordering_wrong_length_is_penalised():
    ds = reasoning_gym.create_dataset("sentence_reordering", size=2, seed=42)
    entry = ds[0]
    # Drop a word -> reasoning_gym scores 0.05 for length mismatch
    dropped = " ".join(entry["answer"].split()[:-1])
    ans = f"<answer>{dropped}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("sentence_reordering", entry))
    assert out["continuous"] < 0.5
    assert out["binary"] == 0.0


# ---------------------------------------------------------------------------
# game_of_life
# ---------------------------------------------------------------------------


def test_game_of_life_perfect():
    ds = reasoning_gym.create_dataset("game_of_life", size=2, seed=42,
                                      grid_size_x=5, grid_size_y=5)
    entry = ds[0]
    ans = f"<answer>{entry['answer']}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("game_of_life", entry))
    assert out["continuous"] == pytest.approx(1.0)
    assert out["binary"] == 1.0


def test_game_of_life_invalid_json_is_zero():
    ds = reasoning_gym.create_dataset("game_of_life", size=2, seed=42,
                                      grid_size_x=5, grid_size_y=5)
    entry = ds[0]
    ans = "<answer>not json</answer>"
    out = compute_score_continuous(ans, None, None, _ei("game_of_life", entry))
    assert out["continuous"] == 0.0


# ---------------------------------------------------------------------------
# knights_knaves (with our custom fixed reward)
# ---------------------------------------------------------------------------


def test_knk_perfect():
    ds = reasoning_gym.create_dataset("knights_knaves", size=2, seed=42, n_people=3)
    entry = ds[0]
    ans = f"<answer>{entry['answer']}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("knights_knaves", entry))
    assert out["continuous"] == pytest.approx(1.0)
    assert out["binary"] == 1.0


def test_knk_anti_hack_extra_assignments_score_zero():
    """Spec §3.2: if the model outputs more assignments than characters, reward=0."""
    ds = reasoning_gym.create_dataset("knights_knaves", size=2, seed=42, n_people=3)
    entry = ds[0]
    names = entry["metadata"]["names"]
    roles_set = sorted({r for (_, r) in _parse_knk(entry["answer"])})
    # Enumerate everyone as every role -> 2 * N assignments > N
    hack_parts = []
    for name in names:
        for role in roles_set:
            hack_parts.append(f"{name} is a {role}")
    hack = ", ".join(hack_parts)
    ans = f"<answer>{hack}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("knights_knaves", entry))
    assert out["continuous"] == 0.0, (
        f"Anti-hack failed: got {out['continuous']} for hack={hack!r}, "
        f"oracle={entry['answer']!r}"
    )


def test_knk_partial_credit_is_fraction():
    """One wrong assignment of three should give 2/3 (no 0.3 base)."""
    ds = reasoning_gym.create_dataset("knights_knaves", size=2, seed=42, n_people=3)
    entry = ds[0]
    oracle_assignments = _parse_knk(entry["answer"])
    assert len(oracle_assignments) == 3
    # Flip exactly one role
    mutated = []
    flipped = False
    for name, role in oracle_assignments:
        if not flipped:
            # Swap role for another valid role used in the puzzle
            other = next(r for (_, r) in oracle_assignments if r != role)
            mutated.append(f"{name} is a {other}")
            flipped = True
        else:
            mutated.append(f"{name} is a {role}")
    ans = f"<answer>{', '.join(mutated)}</answer>"
    out = compute_score_continuous(ans, None, None, _ei("knights_knaves", entry))
    # Our simplified formula: matching/total = 2/3 ≈ 0.667
    assert out["continuous"] == pytest.approx(2 / 3, abs=1e-6)
    assert out["binary"] == 0.0


def test_knk_empty_answer_is_zero():
    ds = reasoning_gym.create_dataset("knights_knaves", size=2, seed=42, n_people=3)
    entry = ds[0]
    out = compute_score_continuous("<answer></answer>", None, None,
                                   _ei("knights_knaves", entry))
    assert out["continuous"] == 0.0


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def test_missing_extra_info_raises():
    with pytest.raises(ValueError):
        compute_score_continuous("x", None, None, {})  # missing task/entry
    with pytest.raises(ValueError):
        compute_score_continuous("x", None, None, None)


def test_binary_mode_matches_perfect_indicator():
    ds = reasoning_gym.create_dataset("word_sorting", size=2, seed=42,
                                      min_words=4, max_words=4)
    entry = ds[0]
    words = entry["metadata"]["sorted_words"]
    partial = ", ".join(words[:3] + ["junk"])
    ans = f"<answer>{partial}</answer>"
    cont = compute_score_continuous(ans, None, None, _ei("word_sorting", entry))
    bina = compute_score_binary(ans, None, None, _ei("word_sorting", entry))
    assert cont["continuous"] == bina["continuous"]
    assert cont["binary"] == bina["binary"]
    assert bina["score"] == cont["binary"]  # binary mode exposes indicator as the score


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_knk(s: str):
    from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset
    return KnightsKnavesDataset._normalize_answer(s)
