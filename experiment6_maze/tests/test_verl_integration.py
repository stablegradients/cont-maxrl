"""Tier 4: verl integration tests — dispatch, shapes, extra_info plumbing."""
import numpy as np
import torch
import pytest

from experiment6_maze.src.maze_env import MazeGenerator
from experiment6_maze.src.data_prep import build_record
from experiment6_maze.src.reward_function import compute_score, compute_score_binary


class TestVerlDispatch:
    def test_custom_estimator_registered(self):
        from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
        assert "binary_maxrl" in CUSTOM_ESTIMATORS
        assert "maxrl_unnorm" in CUSTOM_ESTIMATORS

    def test_verl_advantage_fn_shapes(self):
        """verl wrapper must produce correct tensor shapes."""
        from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
        B, N, L = 4, 16, 384
        token_level_rewards = torch.zeros(B * N, L)
        response_mask = torch.ones(B * N, L)
        for i in range(B * N):
            token_level_rewards[i, L - 1] = torch.rand(1).item()
        index = np.array([i // N for i in range(B * N)])
        for name in ["maxrl_unnorm", "binary_maxrl"]:
            fn = CUSTOM_ESTIMATORS[name]
            adv, ret = fn(token_level_rewards, response_mask, index)
            assert adv.shape == (B * N, L), f"{name}: adv shape {adv.shape}"
            assert ret.shape == (B * N, L), f"{name}: ret shape {ret.shape}"

    def test_binary_maxrl_dispatch_matches_gos_on_binary(self):
        """binary_maxrl and maxrl_unnorm produce identical advantages on binary rewards."""
        from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
        B, N, L = 2, 8, 64
        token_level_rewards = torch.zeros(B * N, L)
        response_mask = torch.ones(B * N, L)
        torch.manual_seed(42)
        for i in range(B * N):
            token_level_rewards[i, L - 1] = (torch.rand(1) < 0.4).float().item()
        index = np.array([i // N for i in range(B * N)])
        adv_gos, _ = CUSTOM_ESTIMATORS["maxrl_unnorm"](token_level_rewards, response_mask, index)
        adv_bml, _ = CUSTOM_ESTIMATORS["binary_maxrl"](token_level_rewards, response_mask, index)
        assert torch.allclose(adv_gos, adv_bml, atol=1e-4), (
            f"Max diff: {torch.max(torch.abs(adv_gos - adv_bml)).item()}"
        )


class TestRewardIntegration:
    def test_compute_score_with_real_record(self):
        record = build_record(seed=0, size=17, l_max=64)
        gt_str = record["reward_model"]["ground_truth"]
        ei = record["extra_info"]
        gen = MazeGenerator(size=17, seed=0)
        gen.generate()
        optimal = " ".join(gen.optimal_path_actions()) + " DONE"
        result = compute_score(optimal, gt_str, extra_info=ei)
        assert result["score"] > 0
        assert result["success"] == 1.0
        assert result["steps"] == ei["optimal_len"]

    def test_compute_score_binary_with_real_record(self):
        record = build_record(seed=0, size=17, l_max=64)
        gt_str = record["reward_model"]["ground_truth"]
        gen = MazeGenerator(size=17, seed=0)
        gen.generate()
        optimal = " ".join(gen.optimal_path_actions()) + " DONE"
        result = compute_score_binary(optimal, gt_str)
        assert result["score"] == 1.0
        assert result["success"] == 1.0
