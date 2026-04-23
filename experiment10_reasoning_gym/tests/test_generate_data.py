"""Tests for generate_data.py: schema, metadata packing, and correctness.

Covers:
- output schema matches what verl's RLHFDataset expects
- prompt ends with the ``<answer>`` instruction tail
- extra_info round-trips through parquet (json encoded entry is recoverable)
- each produced row can be fed into compute_score_continuous and scored
- train/test use disjoint seeds (no prompt overlap)
- difficulty kwargs map to the right reasoning_gym config fields
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import reasoning_gym

from experiment10_reasoning_gym.src.generate_data import (
    ANSWER_INSTRUCTION,
    _difficulty_kwargs,
    build_record,
    generate_dataset,
)
from experiment10_reasoning_gym.src.reward_manager import compute_score_continuous


# ---------------------------------------------------------------------------
# _difficulty_kwargs dispatch
# ---------------------------------------------------------------------------


def test_difficulty_kwargs_word_sorting():
    assert _difficulty_kwargs("word_sorting", 5) == {"min_words": 5, "max_words": 5}


def test_difficulty_kwargs_knights_knaves():
    assert _difficulty_kwargs("knights_knaves", 3) == {"n_people": 3}


def test_difficulty_kwargs_sentence_reordering():
    assert _difficulty_kwargs("sentence_reordering", 6) == {
        "min_words_in_sentence": 6,
        "max_words_in_sentence": 6,
    }


def test_difficulty_kwargs_game_of_life():
    assert _difficulty_kwargs("game_of_life", 7) == {
        "grid_size_x": 7,
        "grid_size_y": 7,
    }


def test_difficulty_kwargs_rejects_unknown_task():
    with pytest.raises(ValueError):
        _difficulty_kwargs("fake_task", 5)


# ---------------------------------------------------------------------------
# build_record schema
# ---------------------------------------------------------------------------


def _entry(task: str, difficulty: int):
    kwargs = _difficulty_kwargs(task, difficulty)
    ds = reasoning_gym.create_dataset(task, size=1, seed=0, **kwargs)
    return ds[0]


def test_build_record_has_required_verl_keys():
    entry = _entry("word_sorting", 4)
    rec = build_record("word_sorting", 4, entry)
    assert set(rec.keys()) == {"data_source", "prompt", "reward_model", "extra_info"}


def test_build_record_data_source_prefixed():
    entry = _entry("word_sorting", 4)
    rec = build_record("word_sorting", 4, entry)
    assert rec["data_source"] == "reasoning_gym_word_sorting"


def test_build_record_prompt_is_user_role_list():
    entry = _entry("word_sorting", 4)
    rec = build_record("word_sorting", 4, entry)
    assert isinstance(rec["prompt"], list)
    assert len(rec["prompt"]) == 1
    assert rec["prompt"][0]["role"] == "user"
    assert isinstance(rec["prompt"][0]["content"], str)
    assert len(rec["prompt"][0]["content"]) > 0


def test_build_record_prompt_appends_answer_instruction():
    entry = _entry("knights_knaves", 3)
    rec = build_record("knights_knaves", 3, entry)
    content = rec["prompt"][0]["content"]
    assert content.endswith(ANSWER_INSTRUCTION)
    assert "<answer>" in content


def test_build_record_reward_model_has_ground_truth_string():
    entry = _entry("knights_knaves", 3)
    rec = build_record("knights_knaves", 3, entry)
    assert "ground_truth" in rec["reward_model"]
    assert isinstance(rec["reward_model"]["ground_truth"], str)
    assert rec["reward_model"]["ground_truth"] == str(entry["answer"])


def test_build_record_extra_info_has_task_and_difficulty_and_entry_json():
    entry = _entry("word_sorting", 4)
    rec = build_record("word_sorting", 4, entry)
    ei = rec["extra_info"]
    assert ei["task"] == "word_sorting"
    assert ei["difficulty"] == 4
    assert isinstance(ei["entry"], str)
    recovered = json.loads(ei["entry"])
    assert recovered["answer"] == entry["answer"]


@pytest.mark.parametrize(
    "task,difficulty",
    [
        ("word_sorting", 4),
        ("word_sorting", 8),
        ("knights_knaves", 3),
        ("sentence_reordering", 5),
        ("game_of_life", 5),
    ],
)
def test_build_record_score_roundtrip(task, difficulty):
    """Feeding the oracle answer (wrapped in <answer>) should yield score=1.0."""
    entry = _entry(task, difficulty)
    rec = build_record(task, difficulty, entry)
    perfect = f"<answer>{entry['answer']}</answer>"
    out = compute_score_continuous(
        perfect,
        rec["reward_model"]["ground_truth"],
        rec["data_source"],
        rec["extra_info"],
    )
    assert out["continuous"] == pytest.approx(1.0), (task, difficulty, out)


# ---------------------------------------------------------------------------
# generate_dataset: size, reproducibility, train/test disjoint
# ---------------------------------------------------------------------------


def test_generate_dataset_size_matches_request():
    df = generate_dataset("word_sorting", 4, size=16, seed=42)
    assert len(df) == 16


def test_generate_dataset_is_reproducible_with_same_seed():
    df1 = generate_dataset("word_sorting", 4, size=8, seed=42)
    df2 = generate_dataset("word_sorting", 4, size=8, seed=42)
    # Compare the underlying oracle answers (stable across seed)
    a1 = [rec["reward_model"]["ground_truth"] for rec in df1.to_dict("records")]
    a2 = [rec["reward_model"]["ground_truth"] for rec in df2.to_dict("records")]
    assert a1 == a2


def test_generate_dataset_different_seeds_produce_different_prompts():
    df1 = generate_dataset("word_sorting", 4, size=8, seed=42)
    df2 = generate_dataset("word_sorting", 4, size=8, seed=42 + 81)
    p1 = [rec["prompt"][0]["content"] for rec in df1.to_dict("records")]
    p2 = [rec["prompt"][0]["content"] for rec in df2.to_dict("records")]
    assert set(p1).isdisjoint(p2), "train/test seed streams overlapped"


# ---------------------------------------------------------------------------
# Parquet roundtrip
# ---------------------------------------------------------------------------


def test_parquet_roundtrip_preserves_schema(tmp_path: Path):
    df = generate_dataset("word_sorting", 6, size=4, seed=42)
    p = tmp_path / "train.parquet"
    df.to_parquet(p)
    df2 = pd.read_parquet(p)
    assert len(df2) == len(df)
    assert set(df2.columns) == {"data_source", "prompt", "reward_model", "extra_info"}
    # Nested JSON entry survives
    entry = json.loads(df2.iloc[0]["extra_info"]["entry"])
    assert "answer" in entry and "metadata" in entry


def test_parquet_roundtrip_keeps_compute_score_working(tmp_path: Path):
    df = generate_dataset("knights_knaves", 3, size=4, seed=42)
    p = tmp_path / "train.parquet"
    df.to_parquet(p)
    df2 = pd.read_parquet(p)
    rec = df2.iloc[0]
    entry = json.loads(rec["extra_info"]["entry"])
    perfect = f"<answer>{entry['answer']}</answer>"
    out = compute_score_continuous(perfect, rec["reward_model"]["ground_truth"],
                                   rec["data_source"], rec["extra_info"])
    assert out["continuous"] == pytest.approx(1.0)
