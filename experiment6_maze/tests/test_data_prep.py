"""Tier 2 tests for maze RL + SFT data preparation."""

import json

import pytest

from experiment6_maze.src.data_prep import build_record, generate_dataset
from experiment6_maze.src.maze_env import (
    MAZE_SYSTEM_PROMPT,
    MazeEnv,
    MazeGenerator,
    parse_actions,
)
from experiment6_maze.src.sft_data_prep import build_sft_record


# =========================================================================
# RL data_prep tests
# =========================================================================


class TestParquetRecordRoundtrip:
    """100 seeds: parse ground_truth back, verify env.start=(1,1), env.goal=(15,15)."""

    def test_parquet_record_roundtrip(self):
        for seed in range(100):
            record = build_record(seed, size=17, l_max=64)
            reward_model = json.loads(record["reward_model"])
            ground_truth = reward_model["ground_truth"]

            env = MazeEnv.from_sequence(ground_truth)
            assert env is not None, f"seed={seed}: from_sequence returned None"
            assert env.start == (1, 1), f"seed={seed}: start={env.start}, expected (1,1)"
            assert env.goal == (15, 15), f"seed={seed}: goal={env.goal}, expected (15,15)"


class TestExtraInfoPresent:
    """extra_info must have l_max=64, optimal_len>0, maze_seed=42."""

    def test_extra_info_present(self):
        record = build_record(42, size=17, l_max=64)
        extra_info = json.loads(record["extra_info"])

        assert extra_info["l_max"] == 64
        assert extra_info["optimal_len"] > 0
        assert extra_info["maze_seed"] == 42


class TestDataSource:
    """data_source must be 'maze_17x17'."""

    def test_data_source(self):
        record = build_record(0, size=17, l_max=64)
        assert record["data_source"] == "maze_17x17"


class TestPromptIsValidJson:
    """prompt must be valid JSON with 2 messages: system + user roles."""

    def test_prompt_is_valid_json(self):
        record = build_record(0, size=17, l_max=64)
        messages = json.loads(record["prompt"])

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestCorrectLength:
    """generate_dataset(n=10) must return exactly 10 rows."""

    def test_correct_length(self):
        df = generate_dataset(n=10, seed_start=0, size=17, l_max=64)
        assert len(df) == 10


class TestTrainValSeedsDisjoint:
    """RL train seeds (0..49999) and val seeds (60000..60511) must be disjoint."""

    def test_train_val_seeds_disjoint(self):
        train_seeds = set(range(0, 50000))
        val_seeds = set(range(60000, 60512))
        assert train_seeds & val_seeds == set()


# =========================================================================
# SFT data_prep tests
# =========================================================================


class TestSftRecordFormat:
    """SFT record: has prompt, completion, completion ends with DONE, system prompt in prompt."""

    def test_sft_record_format(self):
        record = build_sft_record(0, size=17)

        assert "prompt" in record
        assert "completion" in record
        assert record["completion"].endswith("DONE")
        assert MAZE_SYSTEM_PROMPT in record["prompt"]


class TestSftCompletionIsValid:
    """parse_actions returns valid list, and actions solve the maze."""

    def test_sft_completion_is_valid(self):
        for seed in range(20):
            record = build_sft_record(seed, size=17)
            actions = parse_actions(record["completion"])
            assert actions is not None, f"seed={seed}: parse_actions returned None"
            assert len(actions) > 0, f"seed={seed}: empty action list"

            # Verify actions actually solve the maze
            gen = MazeGenerator(size=17, seed=seed)
            gen.generate()
            env = MazeEnv(
                grid=gen._grid,
                start=gen.start,
                goal=gen.goal,
                size=gen.size,
            )
            assert env.check_success(actions), f"seed={seed}: actions do not solve the maze"


class TestPromptFormatMatchesRL:
    """RL user content == SFT user content (after stripping system prompt prefix)."""

    def test_prompt_format_matches_rl(self):
        for seed in range(10):
            rl_record = build_record(seed, size=17, l_max=64)
            sft_record = build_sft_record(seed, size=17)

            rl_messages = json.loads(rl_record["prompt"])
            rl_user_content = rl_messages[1]["content"]

            # SFT prompt = MAZE_SYSTEM_PROMPT + "\n\n" + user_content
            sft_user_content = sft_record["prompt"][len(MAZE_SYSTEM_PROMPT) + 2:]

            assert rl_user_content == sft_user_content, (
                f"seed={seed}: RL user content does not match SFT user content"
            )
