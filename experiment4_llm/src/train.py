"""
Entrypoint for training. Calls verl's run_ppo via Hydra.

Custom advantage estimators (maxrl_unnorm, maxrl_norm, reinforce_baseline)
are registered directly in vendor/verl/verl/trainer/ppo/ray_trainer.py,
which imports experiment4_llm.src.verl_register.CUSTOM_ESTIMATORS at dispatch time.

Usage:
    python experiment4_llm/src/train.py algorithm.adv_estimator=maxrl_unnorm ...
"""
import os

import hydra

from verl.trainer.main_ppo import run_ppo

# Point Hydra at verl's config directory using absolute path
_VERL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "vendor", "verl", "verl", "trainer", "config",
)


@hydra.main(
    config_path=os.path.realpath(_VERL_CONFIG_DIR),
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
