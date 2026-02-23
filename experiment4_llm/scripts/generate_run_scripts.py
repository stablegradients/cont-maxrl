"""
Generate all training and evaluation launch scripts.

Produces standalone bash scripts that also work as SLURM sbatch scripts.
One script per (model x dataset x method x G x seed), plus matching eval scripts
for every benchmark.

All config keys are verified against vendor/verl ppo_trainer.yaml.
Environment setup matches the verified dry_run.sh pipeline.

Usage:
    python experiment4_llm/scripts/generate_run_scripts.py
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

REPO_ROOT_PLACEHOLDER = "/zfsauton/scratch/shrinivr/cont-maxrl"
PYTHON_PATH = "/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python"
CUDA_HOME = "/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl"

# wandb config
WANDB_ENTITY = "stablegradients"
WANDB_PROJECT = "cont-maxrl-code"

# ── Training datasets (have train splits) ──

TRAIN_DATASETS = {
    "apps": {
        "train_file": "apps_train.parquet",
        "val_file": "apps_test.parquet",
    },
    "taco": {
        "train_file": "taco_train.parquet",
        "val_file": "taco_val.parquet",
    },
    "code_contests": {
        "train_file": "code_contests_train.parquet",
        "val_file": "code_contests_test.parquet",
    },
}

# ── Eval benchmarks (all test sets) ──

EVAL_BENCHMARKS = {
    "apps": "apps_test.parquet",
    "code_contests": "code_contests_test.parquet",
    "livecodebench": "livecodebench_test.parquet",
}

# ── Experimental matrix ──

MODELS = {
    "1.5b": {
        "hf_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "tensor_parallel_size": 1,
        "fsdp_size": -1,  # full sharding — NO_SHARD OOMs on 24GB GPUs (optimizer states alone need 24GB)
        "ppo_micro_batch_size_per_gpu": 4,
        "ref_log_prob_micro_batch_size_per_gpu": 4,
        "rollout_log_prob_micro_batch_size_per_gpu": 4,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 1024,
        "gpu_memory_utilization": "0.4",
        "seeds": [42],
        "methods": ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"],
        "group_sizes": [4, 16, 32, 64],
        "slurm_time": "47:00:00",
        "slurm_mem": "256G",
        "slurm_gpus": 4,
        "slurm_cpus": 48,
        "slurm_partition": "general",
    },
    "7b": {
        "hf_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "tensor_parallel_size": 1,
        "fsdp_size": -1,  # full sharding across all GPUs
        "ppo_micro_batch_size_per_gpu": 2,
        "ref_log_prob_micro_batch_size_per_gpu": 1,
        "rollout_log_prob_micro_batch_size_per_gpu": 1,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 512,
        "gpu_memory_utilization": "0.4",
        "seeds": [42],
        "methods": ["grpo", "maxrl_unnorm", "maxrl_norm", "reinforce_baseline"],
        "group_sizes": [4, 16, 32, 64],
        "slurm_time": "47:00:00",
        "slurm_mem": "256G",
        "slurm_gpus": 4,
        "slurm_cpus": 32,
        "slurm_partition": "general",
    },
    "32b": {
        "hf_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "tensor_parallel_size": 4,
        "fsdp_size": -1,  # full sharding across all GPUs
        "ppo_micro_batch_size_per_gpu": 1,
        "ref_log_prob_micro_batch_size_per_gpu": 1,
        "rollout_log_prob_micro_batch_size_per_gpu": 1,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 256,
        "gpu_memory_utilization": "0.4",
        "seeds": [42],
        "methods": ["grpo", "maxrl_unnorm", "maxrl_norm"],
        "group_sizes": [16, 32, 64],
        "slurm_time": "47:00:00",
        "slurm_mem": "256G",
        "slurm_gpus": 8,
        "slurm_cpus": 32,
        "slurm_partition": "general",
    },
}

TOTAL_STEPS = 150
TOTAL_ROLLOUTS_PER_STEP = 4096
DEFAULT_BATCH_SIZE = 128
MINI_BATCH = 32
PPO_EPOCHS = 1
LR = "1e-6"
KL_COEF = "0.001"
CLIP_RATIO = "0.2"
CLIP_RATIO_HIGH = "0.28"
TRAIN_TEMP = "1.0"
EVAL_TEMP = "0.8"
MAX_PROMPT_LEN = 1024
MAX_RESPONSE_LEN = 1024
N_EVAL_COMPLETIONS = 200
K_VALUES = "1 2 5 10 20 50 100"
SAVE_FREQ = 25
RAY_NUM_CPUS = 48


def _batch_size_for_g(G: int) -> int:
    """Compute batch size. For G>=16, cap rollouts at 2048 to reduce cost."""
    if G >= 16:
        return 2048 // G  # 128 for G=16, 64 for G=32, 32 for G=64
    return TOTAL_ROLLOUTS_PER_STEP // G  # 1024 for G=4


def generate_train_script(
    model_key: str,
    model_cfg: dict,
    dataset: str,
    dataset_cfg: dict,
    method: str,
    G: int,
    seed: int,
) -> str:
    """Generate a standalone training script (works as both bash and sbatch)."""
    hf_id = model_cfg["hf_id"]
    batch_size = _batch_size_for_g(G)
    exp_name = f"{model_key}_{dataset}_{method}_g{G}_s{seed}"
    wandb_group = f"{model_key}_{dataset}_{method}_g{G}"

    # SLURM header
    slurm_header = f"""#SBATCH --job-name={exp_name}
#SBATCH --partition={model_cfg['slurm_partition']}
#SBATCH --gres=gpu:{model_cfg['slurm_gpus']}
#SBATCH --time={model_cfg['slurm_time']}
#SBATCH --mem={model_cfg['slurm_mem']}
#SBATCH --cpus-per-task={model_cfg['slurm_cpus']}
#SBATCH --nodes=1
#SBATCH --exclude=gpu20,gpu21,gpu22,gpu23
#SBATCH --output={REPO_ROOT_PLACEHOLDER}/results/{exp_name}/slurm_%j.out
#SBATCH --error={REPO_ROOT_PLACEHOLDER}/results/{exp_name}/slurm_%j.err"""

    # KL loss applied uniformly to all methods for fair comparison
    use_kl_loss = "True"

    script = f"""#!/usr/bin/env bash
{slurm_header}
set -euo pipefail

# ============================================================
# Train: {hf_id.split("/")[-1]} | {dataset.upper()} | {method} | G={G} | seed={seed}
# ============================================================

REPO_ROOT="{REPO_ROOT_PLACEHOLDER}"
PYTHON="{PYTHON_PATH}"
export PYTHONNOUSERSITE=1

OUTPUT_DIR="${{REPO_ROOT}}/results/{exp_name}"
mkdir -p "${{OUTPUT_DIR}}"

# Conda CUDA (not system CUDA)
export CUDA_HOME="{CUDA_HOME}"
export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{CUDA_HOME}}/lib:${{LD_LIBRARY_PATH:-}}"

# Caches
export HF_HOME="/zfsauton/scratch/shrinivr/.cache/huggingface"
export TORCH_HOME="${{REPO_ROOT}}/.cache/torch"
export VLLM_CACHE_ROOT="${{REPO_ROOT}}/.cache/vllm"
# verl requires vLLM v0 engine (v1 doesn't implement free_cache_engine)
export VLLM_USE_V1=0
# Ray and tmp on local disk (avoids NFS socket/cleanup issues)
export RAY_TMPDIR="/tmp/ray_${{USER}}"
export TMPDIR="/tmp/${{USER}}_verl_tmp"
mkdir -p "${{TORCH_HOME}}" "${{VLLM_CACHE_ROOT}}" "${{RAY_TMPDIR}}" "${{TMPDIR}}"

# Project must be importable (for custom advantage estimators)
export PYTHONPATH="${{REPO_ROOT}}:${{PYTHONPATH:-}}"

# wandb
export WANDB_ENTITY="{WANDB_ENTITY}"
export WANDB_RUN_GROUP="{wandb_group}"

# Use SLURM-assigned GPU count, fall back to nvidia-smi for local runs
N_GPUS=${{SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo {model_cfg['slurm_gpus']})}}
echo "=== Train: {exp_name} ==="
echo "GPUs: ${{N_GPUS}}, Model: {hf_id}"
echo "Method: {method}, G={G}, Seed={seed}"
echo "Output: ${{OUTPUT_DIR}}"

$PYTHON "${{REPO_ROOT}}/experiment4_llm/src/train.py" \\
  algorithm.adv_estimator={method} \\
  algorithm.use_kl_in_reward=False \\
  algorithm.kl_ctrl.type=fixed \\
  algorithm.kl_ctrl.kl_coef={KL_COEF} \\
  data.train_files="${{REPO_ROOT}}/data/{dataset_cfg['train_file']}" \\
  data.val_files="${{REPO_ROOT}}/data/{dataset_cfg['val_file']}" \\
  data.train_batch_size={batch_size} \\
  data.max_prompt_length={MAX_PROMPT_LEN} \\
  data.max_response_length={MAX_RESPONSE_LEN} \\
  data.shuffle=True \\
  data.filter_overlong_prompts=True \\
  data.reward_fn_key=data_source \\
  actor_rollout_ref.model.path={hf_id} \\
  actor_rollout_ref.model.enable_gradient_checkpointing=True \\
  actor_rollout_ref.model.trust_remote_code=True \\
  actor_rollout_ref.actor.strategy=fsdp \\
  actor_rollout_ref.actor.fsdp_config.fsdp_size={model_cfg['fsdp_size']} \\
  actor_rollout_ref.actor.ppo_mini_batch_size={MINI_BATCH} \\
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={model_cfg['ppo_micro_batch_size_per_gpu']} \\
  actor_rollout_ref.actor.ppo_epochs={PPO_EPOCHS} \\
  actor_rollout_ref.actor.use_kl_loss={use_kl_loss} \\
  actor_rollout_ref.actor.kl_loss_coef={KL_COEF} \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.actor.clip_ratio={CLIP_RATIO} \\
  actor_rollout_ref.actor.clip_ratio_high={CLIP_RATIO_HIGH} \\
  actor_rollout_ref.actor.use_torch_compile=False \\
  actor_rollout_ref.actor.optim.lr={LR} \\
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={model_cfg['ref_log_prob_micro_batch_size_per_gpu']} \\
  actor_rollout_ref.ref.use_torch_compile=False \\
  actor_rollout_ref.rollout.name=vllm \\
  actor_rollout_ref.rollout.n={G} \\
  actor_rollout_ref.rollout.temperature={TRAIN_TEMP} \\
  actor_rollout_ref.rollout.top_p=1.0 \\
  actor_rollout_ref.rollout.gpu_memory_utilization={model_cfg['gpu_memory_utilization']} \\
  actor_rollout_ref.rollout.dtype=bfloat16 \\
  actor_rollout_ref.rollout.tensor_model_parallel_size={model_cfg['tensor_parallel_size']} \\
  actor_rollout_ref.rollout.max_num_batched_tokens={model_cfg['max_num_batched_tokens']} \\
  actor_rollout_ref.rollout.max_num_seqs={model_cfg['max_num_seqs']} \\
  actor_rollout_ref.rollout.enforce_eager=True \\
  actor_rollout_ref.rollout.free_cache_engine=True \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={model_cfg['rollout_log_prob_micro_batch_size_per_gpu']} \\
  actor_rollout_ref.rollout.enable_chunked_prefill=False \\
  actor_rollout_ref.rollout.val_kwargs.n={G} \\
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \\
  actor_rollout_ref.rollout.val_kwargs.temperature={TRAIN_TEMP} \\
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \\
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \\
  reward_model.enable=False \\
  custom_reward_function.path="${{REPO_ROOT}}/experiment4_llm/src/reward_function.py" \\
  custom_reward_function.name=compute_score \\
  trainer.total_training_steps={TOTAL_STEPS} \\
  trainer.total_epochs=30 \\
  trainer.project_name={WANDB_PROJECT} \\
  trainer.experiment_name={exp_name} \\
  "trainer.logger=['console','wandb']" \\
  trainer.default_local_dir="${{OUTPUT_DIR}}" \\
  trainer.n_gpus_per_node=${{N_GPUS}} \\
  trainer.nnodes=1 \\
  trainer.save_freq={SAVE_FREQ} \\
  trainer.test_freq=20 \\
  trainer.val_before_train=False \\
  ray_init.num_cpus={RAY_NUM_CPUS}

echo ""
echo "=== Training complete: {exp_name} ==="
"""
    return script


def generate_eval_script(
    model_key: str,
    model_cfg: dict,
    train_dataset: str,
    method: str,
    G: int,
    seed: int,
    eval_benchmark: str,
    eval_file: str,
) -> str:
    """Generate a standalone evaluation script (works as both bash and sbatch).

    Evaluates a model trained on `train_dataset` against `eval_benchmark`.
    """
    exp_name = f"{model_key}_{train_dataset}_{method}_g{G}_s{seed}"
    eval_name = f"{exp_name}_on_{eval_benchmark}"

    slurm_header = f"""#SBATCH --job-name=eval_{eval_name}
#SBATCH --partition={model_cfg['slurm_partition']}
#SBATCH --gres=gpu:{model_cfg['slurm_gpus']}
#SBATCH --time=47:00:00
#SBATCH --mem={model_cfg['slurm_mem']}
#SBATCH --cpus-per-task={model_cfg['slurm_cpus']}
#SBATCH --nodes=1
#SBATCH --exclude=gpu20,gpu21,gpu22,gpu23
#SBATCH --output={REPO_ROOT_PLACEHOLDER}/results/{exp_name}/eval_{eval_benchmark}/slurm_%j.out
#SBATCH --error={REPO_ROOT_PLACEHOLDER}/results/{exp_name}/eval_{eval_benchmark}/slurm_%j.err"""

    script = f"""#!/usr/bin/env bash
{slurm_header}
set -euo pipefail

# ============================================================
# Eval: {model_key} trained on {train_dataset.upper()} | eval on {eval_benchmark.upper()}
# {method} | G={G} | seed={seed}
# ============================================================

REPO_ROOT="{REPO_ROOT_PLACEHOLDER}"
PYTHON="{PYTHON_PATH}"
export PYTHONNOUSERSITE=1

# Conda CUDA (not system CUDA)
export CUDA_HOME="{CUDA_HOME}"
export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{CUDA_HOME}}/lib:${{LD_LIBRARY_PATH:-}}"

# Caches
export HF_HOME="/zfsauton/scratch/shrinivr/.cache/huggingface"
export TORCH_HOME="${{REPO_ROOT}}/.cache/torch"
export TMPDIR="/tmp/${{USER}}_verl_tmp"
mkdir -p "${{TMPDIR}}"

# Project must be importable
export PYTHONPATH="${{REPO_ROOT}}:${{PYTHONPATH:-}}"

CHECKPOINT_DIR="${{REPO_ROOT}}/results/{exp_name}"
OUTPUT_DIR="${{CHECKPOINT_DIR}}/eval_{eval_benchmark}"
mkdir -p "${{OUTPUT_DIR}}"

N_GPUS=${{SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo {model_cfg['slurm_gpus']})}}
echo "=== Eval: {eval_name} ==="
echo "GPUs: ${{N_GPUS}}"
echo "Checkpoint: ${{CHECKPOINT_DIR}}"
echo "Benchmark: {eval_benchmark} ({eval_file})"

$PYTHON "${{REPO_ROOT}}/experiment4_llm/src/evaluate.py" \\
  --checkpoint_dir "${{CHECKPOINT_DIR}}" \\
  --data_file "${{REPO_ROOT}}/data/{eval_file}" \\
  --output_dir "${{OUTPUT_DIR}}" \\
  --n_completions {N_EVAL_COMPLETIONS} \\
  --temperature {EVAL_TEMP} \\
  --max_tokens {MAX_RESPONSE_LEN} \\
  --tensor_parallel_size ${{N_GPUS}} \\
  --k_values {K_VALUES}

echo ""
echo "=== Evaluation complete: {eval_name} ==="
"""
    return script


def main():
    script_dir = Path(__file__).parent / "runs"
    script_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    eval_count = 0

    for model_key, model_cfg in MODELS.items():
        for dataset, dataset_cfg in TRAIN_DATASETS.items():
            for method in model_cfg["methods"]:
                for G in model_cfg["group_sizes"]:
                    for seed in model_cfg["seeds"]:
                        # Training script
                        train_name = f"train_{model_key}_{dataset}_{method}_g{G}_s{seed}.sh"
                        train_path = script_dir / train_name
                        train_content = generate_train_script(
                            model_key, model_cfg, dataset, dataset_cfg, method, G, seed
                        )
                        with open(train_path, "w") as f:
                            f.write(train_content)
                        os.chmod(train_path, os.stat(train_path).st_mode | stat.S_IEXEC)
                        train_count += 1

                        # Cross-benchmark eval scripts
                        for eval_bench, eval_file in EVAL_BENCHMARKS.items():
                            eval_name = f"eval_{model_key}_{dataset}_{method}_g{G}_s{seed}_on_{eval_bench}.sh"
                            eval_path = script_dir / eval_name
                            eval_content = generate_eval_script(
                                model_key, model_cfg, dataset, method, G, seed,
                                eval_bench, eval_file,
                            )
                            with open(eval_path, "w") as f:
                                f.write(eval_content)
                            os.chmod(eval_path, os.stat(eval_path).st_mode | stat.S_IEXEC)
                            eval_count += 1

    print(f"Generated {train_count} training scripts")
    print(f"Generated {eval_count} eval scripts")
    print(f"Total: {train_count + eval_count} scripts in {script_dir}")

    # Print summary table
    print("\n=== Experiment Matrix ===")
    print(f"{'Model':<6} {'Datasets':<30} {'Methods':<45} {'G (B)':<20} {'Seeds':<12}")
    print("-" * 120)
    for model_key, model_cfg in MODELS.items():
        datasets = ", ".join(TRAIN_DATASETS.keys())
        methods = ", ".join(model_cfg["methods"])
        g_sizes = ", ".join(
            f"{g}(B={_batch_size_for_g(g)})" for g in model_cfg["group_sizes"]
        )
        seeds = ", ".join(str(s) for s in model_cfg["seeds"])
        n_train = (
            len(TRAIN_DATASETS)
            * len(model_cfg["methods"])
            * len(model_cfg["group_sizes"])
            * len(model_cfg["seeds"])
        )
        n_eval = n_train * len(EVAL_BENCHMARKS)
        print(f"{model_key:<6} {datasets:<30} {methods:<45} {g_sizes:<15} {seeds:<12}")
        print(f"       {n_train} train + {n_eval} eval scripts")

    # Print eval benchmarks
    print(f"\nEval benchmarks: {', '.join(EVAL_BENCHMARKS.keys())}")
    print("Each trained model is evaluated on ALL benchmarks (cross-dataset generalization)")

    # Print SLURM submission examples
    print("\n=== SLURM Submission ===")
    print("# Submit a single training job:")
    for model_key in MODELS:
        name = f"train_{model_key}_apps_maxrl_unnorm_g16_s42.sh"
        print(f"  sbatch {script_dir / name}")

    print("\n# Submit all training jobs for a model+dataset:")
    print(f"  for f in {script_dir}/train_1.5b_apps_*.sh; do sbatch \"$f\"; done")

    print("\n# Submit all eval jobs for a trained model:")
    print(f"  for f in {script_dir}/eval_1.5b_apps_maxrl_unnorm_g16_s42_on_*.sh; do sbatch \"$f\"; done")

    print("\n# Run locally (no SLURM, SBATCH lines are ignored):")
    name = f"train_1.5b_apps_maxrl_unnorm_g4_s42.sh"
    print(f"  bash {script_dir / name}")


if __name__ == "__main__":
    main()
