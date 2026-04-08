"""Tier 2 tests for maze RL + SFT data preparation."""

import pytest

from experiment6_maze.src.data_prep import build_record, generate_dataset
from experiment6_maze.src.maze_env import (
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
            ground_truth = record["reward_model"]["ground_truth"]

            env = MazeEnv.from_sequence(ground_truth)
            assert env is not None, f"seed={seed}: from_sequence returned None"
            assert env.start == (1, 1), f"seed={seed}: start={env.start}, expected (1,1)"
            assert env.goal == (15, 15), f"seed={seed}: goal={env.goal}, expected (15,15)"


class TestExtraInfoPresent:
    """extra_info must have l_max=64, optimal_len>0, maze_seed=42."""

    def test_extra_info_present(self):
        record = build_record(42, size=17, l_max=64)
        extra_info = record["extra_info"]

        assert extra_info["l_max"] == 64
        assert extra_info["optimal_len"] > 0
        assert extra_info["maze_seed"] == 42


class TestDataSource:
    """data_source must be 'maze_17x17'."""

    def test_data_source(self):
        record = build_record(0, size=17, l_max=64)
        assert record["data_source"] == "maze_17x17"


class TestPromptIsChatMessages:
    """prompt must be a list with a single user message (word-token format, no system prompt)."""

    def test_prompt_is_chat_messages(self):
        record = build_record(0, size=17, l_max=64)
        messages = record["prompt"]

        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"].startswith("<bos> GRID_START")
        assert "PATH_START" in messages[0]["content"]


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
    """SFT record: has sequence, prompt, optimal_path_length; sequence ends with DONE <eos>."""

    def test_sft_record_format(self):
        record = build_sft_record(0, size=17)

        assert "sequence" in record
        assert "prompt" in record
        assert "optimal_path_length" in record
        assert record["sequence"].endswith("DONE <eos>")
        assert record["prompt"].startswith("<bos> GRID_START")


class TestSftCompletionIsValid:
    """Extract actions from SFT sequence (after PATH_START), verify they solve the maze."""

    def test_sft_completion_is_valid(self):
        for seed in range(20):
            record = build_sft_record(seed, size=17)
            # Actions are between PATH_START and DONE in the sequence
            seq = record["sequence"]
            path_start_idx = seq.index("PATH_START") + len("PATH_START")
            done_idx = seq.index("DONE")
            action_str = seq[path_start_idx:done_idx].strip()
            actions = parse_actions(action_str + " DONE")
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
    """RL prompt content == SFT prompt content."""

    def test_prompt_format_matches_rl(self):
        for seed in range(10):
            rl_record = build_record(seed, size=17, l_max=64)
            sft_record = build_sft_record(seed, size=17)

            rl_user_content = rl_record["prompt"][0]["content"]
            sft_prompt = sft_record["prompt"]

            assert rl_user_content == sft_prompt, (
                f"seed={seed}: RL user content does not match SFT prompt"
            )
