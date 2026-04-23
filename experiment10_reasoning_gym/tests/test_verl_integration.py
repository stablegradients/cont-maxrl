"""Integration-shaped tests: verify verl can load our parquet files.

These tests avoid spinning up Ray or training — they exercise the
RLHFDataset loader and our reward function against a small sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiment10_reasoning_gym.src.generate_data import generate_dataset
from experiment10_reasoning_gym.src.reward_manager import (
    compute_score_continuous,
    compute_score_binary,
)


def test_rlhf_dataset_loads_our_parquet(tmp_path: Path):
    """Spec §12 Test 3 equivalent: verl's RLHFDataset must load our parquet."""
    pytest.importorskip("transformers")
    df = generate_dataset("word_sorting", 4, size=16, seed=42)
    parquet_path = tmp_path / "train.parquet"
    df.to_parquet(parquet_path)

    # Use the actual training tokenizer so we exercise the chat template
    # apply step in RLHFDataset — this is the realistic path at train time.
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from verl.utils.dataset.rl_dataset import RLHFDataset

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = OmegaConf.create({
        "prompt_key": "prompt",
        "max_prompt_length": 512,
        "truncation": "error",
        "filter_overlong_prompts": True,
        "cache_dir": str(tmp_path / "cache"),
        "return_raw_chat": False,
        "filter_overlong_prompts_workers": 1,
        "shuffle": False,
        "return_full_prompt": False,
        "return_multi_modal_inputs": True,
        "trust_remote_code": False,
        "image_key": "images",
        "video_key": "videos",
        "use_shm": False,
    })
    ds = RLHFDataset(data_files=[str(parquet_path)], tokenizer=tok, config=cfg)
    assert len(ds) <= 16  # filter_overlong may drop some rows
    assert len(ds) > 0
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    # reward_model and extra_info propagate as non-tensor fields
    assert "reward_model" in sample
    assert "extra_info" in sample


def test_score_reads_extra_info_from_dataframe_row():
    df = generate_dataset("sentence_reordering", 5, size=4, seed=42)
    for _, row in df.iterrows():
        entry = json.loads(row["extra_info"]["entry"])
        perfect = f"<answer>{entry['answer']}</answer>"
        out = compute_score_continuous(
            perfect,
            row["reward_model"]["ground_truth"],
            row["data_source"],
            row["extra_info"],
        )
        assert out["score"] == pytest.approx(1.0)
        assert out["binary"] == 1.0


def test_binary_compute_score_matches_indicator_of_perfect():
    df = generate_dataset("word_sorting", 4, size=4, seed=42)
    row = df.iloc[0]
    entry = json.loads(row["extra_info"]["entry"])
    words = entry["metadata"]["sorted_words"]
    # Partially correct answer
    partial = ", ".join(words[:2] + ["ZZZ", "ZZZ"])
    c = compute_score_continuous(f"<answer>{partial}</answer>", None,
                                 row["data_source"], row["extra_info"])
    b = compute_score_binary(f"<answer>{partial}</answer>", None,
                             row["data_source"], row["extra_info"])
    assert 0.0 < c["continuous"] < 1.0
    assert c["binary"] == 0.0 and b["binary"] == 0.0
    assert b["score"] == 0.0  # binary mode returns the indicator
    assert c["score"] == c["continuous"]  # continuous mode returns the raw score
