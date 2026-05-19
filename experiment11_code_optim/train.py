"""
PIE training entrypoint (spec v9 §2).

Reuses experiment4_llm.src.train.main unchanged so PIE training shares the
verl + custom-estimator dispatch path with DeepCoder. PIE-specific behavior
comes exclusively from the SLURM script's Hydra overrides (data paths,
custom_reward_function.path, reward_kwargs, etc.) and the PIE reward module
at experiment11_code_optim/reward/pie_reward_fn.py (built at §14 step 7).

Usage (parallel to experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_*.sh):
    python experiment11_code_optim/train.py \\
      algorithm.adv_estimator=maxrl_unnorm \\
      data.train_files=... \\
      custom_reward_function.path=... \\
      ...
"""
from experiment4_llm.src.train import main

if __name__ == "__main__":
    main()
