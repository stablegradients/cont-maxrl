"""
PIE training entrypoint (spec v9 §2).

Reuses experiment4_llm.src.train.main unchanged so PIE training shares the
verl + custom-estimator dispatch path with DeepCoder. PIE-specific behavior
comes exclusively from:

  1. The SLURM script's Hydra overrides (data paths, custom_reward_function.path,
     reward_kwargs).
  2. The PIE reward module at experiment11_code_optim/reward/pie_reward_fn.py.
  3. The PIE_MODE sys.modules shim installed below — closes deferred fix F6.

Usage (parallel to experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_*.sh):
    PIE_MODE=1 python experiment11_code_optim/train.py \\
      algorithm.adv_estimator=maxrl_unnorm \\
      data.train_files=... \\
      custom_reward_function.path=... \\
      ...
"""
import os
import sys

# F6 (DEFERRED_FIXES.md): install the PIE_MODE sys.modules shim BEFORE any
# experiment4_llm imports fire. insitu_eval.py hard-codes
#     from experiment4_llm.src.reward_function import compute_scores_batch, extract_code
# so without redirection the in-situ LiveCodeBench-style eval would route C++
# programs through DeepCoder's Python reward function — every rollout would
# silently score 0 with no error. This is the silent-failure trap spec §8
# warns about repeatedly. Install must happen before `from experiment4_llm.src
# .train import main` below so the transitive import chain picks up our shim.
if os.environ.get("PIE_MODE") == "1":
    from experiment11_code_optim.reward import pie_reward_fn
    sys.modules["experiment4_llm.src.reward_function"] = pie_reward_fn

# §8.3 actor carve-out: persist every saved checkpoint as a hard-linked
# snapshot under PIE_ACTOR_CARVEOUT_DIR so the §8.5 post-hoc best@K eval
# has the full trajectory even with verl's max_actor_ckpt_to_keep=1.
# Hard-links are zero-cost and survive verl's rolling delete. Hook is a
# no-op if PIE_ACTOR_CARVEOUT_DIR is unset.
from experiment11_code_optim.guards.actor_carveout import install_actor_carveout_hook  # noqa: E402
install_actor_carveout_hook()

# PIE-specific per-step metrics (gate_pass_rate, mean_speedup, max_speedup,
# fraction_dead_groups, fraction_mode2, speedup_rate). Monkey-patches
# verl.trainer.ppo.metric_utils.compute_data_metrics — must be installed
# before `from experiment4_llm.src.train import main` so the patched fn is
# already the bound symbol when verl resolves it. Hook is a no-op outside
# PIE_MODE=1.
if os.environ.get("PIE_MODE") == "1":
    from experiment11_code_optim.guards.pie_metrics import install_pie_metrics_hook  # noqa: E402
    install_pie_metrics_hook()

from experiment4_llm.src.train import main  # noqa: E402 — must follow shim install

if __name__ == "__main__":
    main()
