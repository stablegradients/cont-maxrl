from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from experiment4_llm.src.verl_main_ppo import (
    _build_verl_overrides_from_experiment_cfg,
    _compute_adv_from_rewards_tensor,
    _normalize_extra_overrides,
    _rewrite_argv_for_config,
    _to_hydra_override,
    load_adv_estimator_from_config,
    parse_config_path,
    rewrite_argv_for_verl,
)


class VerlMainPpoTests(unittest.TestCase):
    def test_parse_config_path(self) -> None:
        p1 = parse_config_path(["--config", "cfg.yaml"])
        p2 = parse_config_path(["--foo", "1", "--config=cfg2.yaml"])
        p3 = parse_config_path(["--foo", "1"])

        self.assertEqual(p1, Path("cfg.yaml"))
        self.assertEqual(p2, Path("cfg2.yaml"))
        self.assertIsNone(p3)

    def test_load_adv_estimator_from_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp4_cfg_") as tmpdir:
            path = Path(tmpdir) / "cfg.yaml"
            path.write_text(
                yaml.safe_dump({"algorithm": {"adv_estimator": "maxrl_unnorm"}}),
                encoding="utf-8",
            )
            method = load_adv_estimator_from_config(path)
            self.assertEqual(method, "maxrl_unnorm")

    def test_load_adv_estimator_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp4_cfg_") as tmpdir:
            path = Path(tmpdir) / "cfg.yaml"
            path.write_text(
                yaml.safe_dump({"algorithm": {"adv_estimator": "does_not_exist"}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _ = load_adv_estimator_from_config(path)

    def test_rewrite_argv_for_verl(self) -> None:
        out = rewrite_argv_for_verl(
            [
                "--config",
                "experiment4_llm/configs/qwen_1.5b_grpo.yaml",
                "--training.total_steps",
                "1",
            ]
        )
        self.assertIn("--config-path", out)
        self.assertIn("--config-name", out)
        self.assertIn("qwen_1.5b_grpo", out)
        self.assertIn("--training.total_steps", out)
        self.assertIn("1", out)

    def test_compute_adv_helper(self) -> None:
        rewards_1d = torch.tensor([0.1, 0.2, 1.0], dtype=torch.float64)
        out_1d = _compute_adv_from_rewards_tensor(rewards_1d, method="maxrl_unnorm")
        self.assertEqual(out_1d.shape, rewards_1d.shape)

        rewards_2d = torch.tensor([[0.1, 0.2, 1.0], [0.3, 0.3, 0.3]], dtype=torch.float64)
        out_2d = _compute_adv_from_rewards_tensor(rewards_2d, method="grpo")
        self.assertEqual(out_2d.shape, rewards_2d.shape)

        self.assertAlmostEqual(float(out_2d[0].mean()), 0.0, places=10)
        self.assertTrue(torch.allclose(out_2d[1], torch.zeros_like(out_2d[1])))

    def test_normalize_extra_overrides(self) -> None:
        out = _normalize_extra_overrides(["--config", "cfg.yaml", "--foo.bar", "1", "baz=2"])
        self.assertEqual(out, ["foo.bar=1", "baz=2"])

    def test_build_verl_overrides_from_experiment_config(self) -> None:
        cfg = {
            "experiment": {"name": "exp4_unit"},
            "model": {"name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "max_prompt_length": 64, "max_completion_length": 32},
            "algorithm": {"adv_estimator": "maxrl_unnorm", "kl_coef": 0.001},
            "rollout": {"group_size": 4, "temperature": 1.0},
            "training": {"global_batch_size_prompts": 8, "mini_batch_size_prompts": 4, "total_steps": 2},
            "data": {"train_jsonl": "train.jsonl", "eval_jsonl": "eval.jsonl"},
            "reward": {"module": "experiment4_llm.src.reward_function", "function": "compute_score"},
            "logging": {"output_dir": "results_experiment4/unit"},
        }
        overrides = _build_verl_overrides_from_experiment_cfg(
            cfg,
            config_path=Path("experiment4_llm/configs/qwen_1.5b_grpo.yaml"),
        )
        joined = "\n".join(overrides)
        self.assertIn("algorithm.adv_estimator", joined)
        self.assertIn("actor_rollout_ref.rollout.name", joined)
        self.assertIn("reward.custom_reward_function.path", joined)
        self.assertIn("data.custom_cls.path", joined)

    def test_rewrite_argv_for_custom_config_uses_ppo_trainer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp4_cfg_") as tmpdir:
            path = Path(tmpdir) / "cfg.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "model": {"name_or_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
                        "algorithm": {"adv_estimator": "grpo"},
                        "training": {"total_steps": 1},
                        "data": {"train_jsonl": "train.jsonl", "eval_jsonl": "eval.jsonl"},
                    }
                ),
                encoding="utf-8",
            )
            out = _rewrite_argv_for_config(["--config", str(path)], config_path=path)
            self.assertIn("--config-name", out)
            idx = out.index("--config-name")
            self.assertEqual(out[idx + 1], "ppo_trainer")

    def test_to_hydra_override_string(self) -> None:
        ov = _to_hydra_override("trainer.project_name", "cont_maxrl")
        self.assertEqual(ov, 'trainer.project_name="cont_maxrl"')


if __name__ == "__main__":
    unittest.main()
