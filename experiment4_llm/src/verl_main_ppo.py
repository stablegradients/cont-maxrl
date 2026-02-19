from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .advantages import (
    AdvantageMethod,
    broadcast_sequence_advantages_to_tokens,
    compute_advantage_1d,
    compute_advantages_from_group_indices,
    compute_grouped_advantages,
)


SUPPORTED_ADV_ESTIMATORS: set[str] = {
    "grpo",
    "maxrl_unnorm",
    "maxrl_norm",
    "reinforce_baseline",
}


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-like YAML at {path}, got {type(data)!r}")
    return data


def _is_experiment4_config(cfg: dict[str, Any]) -> bool:
    # Our experiment config has this shape; native verl Hydra configs do not.
    return all(key in cfg for key in ("model", "algorithm", "training", "data"))


def parse_config_path(argv: list[str]) -> Path | None:
    for idx, arg in enumerate(argv):
        if arg == "--config" and idx + 1 < len(argv):
            return Path(argv[idx + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return None


def _strip_config_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        cleaned.append(arg)
    return cleaned


def _normalize_extra_overrides(argv: list[str]) -> list[str]:
    """
    Normalize CLI overrides into Hydra syntax.

    Examples:
    - --trainer.total_training_steps 1 -> trainer.total_training_steps=1
    - actor_rollout_ref.rollout.n=4    -> unchanged
    """
    args = _strip_config_args(argv)
    normalized: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg.startswith("--"):
            key = arg[2:]
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                normalized.append(f"{key}={args[idx + 1]}")
                idx += 2
            else:
                normalized.append(key)
                idx += 1
            continue
        normalized.append(arg)
        idx += 1
    return normalized


def _parse_legacy_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    # JSON literals for list/dict/string.
    if raw and raw[0] in "[{\"":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _apply_legacy_override(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur: dict[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _split_legacy_and_hydra_overrides(
    overrides: list[str],
) -> tuple[dict[str, Any], list[str]]:
    legacy_roots = {
        "experiment",
        "model",
        "algorithm",
        "rollout",
        "training",
        "data",
        "reward",
        "evaluation",
        "logging",
    }
    legacy: dict[str, Any] = {}
    passthrough: list[str] = []

    for item in overrides:
        if "=" not in item:
            passthrough.append(item)
            continue
        key, raw_value = item.split("=", 1)
        root = key.split(".", 1)[0]
        if root in legacy_roots:
            legacy[key] = _parse_legacy_override_value(raw_value)
        else:
            passthrough.append(item)
    return legacy, passthrough


def _to_hydra_override(key: str, value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, str):
        rhs = json.dumps(value)
    elif isinstance(value, (bool, int, float, list, dict)):
        rhs = json.dumps(value)
    elif value is None:
        rhs = "null"
    else:
        rhs = json.dumps(str(value))
    return f"{key}={rhs}"


def _coerce_kl_loss_type(raw: Any) -> str:
    value = str(raw or "kl").strip().lower()
    aliases = {
        "k1": "kl",
        "k2": "mse",
        "k3": "low_var_kl",
    }
    return aliases.get(value, value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return default


def _resolve_reward_module_path(module_spec: str, config_path: Path) -> str:
    if module_spec.startswith("pkg://"):
        return module_spec

    if module_spec.endswith(".py") or "/" in module_spec or module_spec.startswith("."):
        abs_path = (config_path.parent / module_spec).resolve()
        return str(abs_path)

    # Assume python import path and preserve package loading semantics.
    importlib.import_module(module_spec)
    return f"pkg://{module_spec}"


def _count_visible_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [x for x in visible.split(",") if x.strip()]
        return max(len(entries), 1)
    if torch.cuda.is_available():
        return max(torch.cuda.device_count(), 1)
    return 1


def _build_verl_overrides_from_experiment_cfg(cfg: dict[str, Any], config_path: Path) -> list[str]:
    model = dict(cfg.get("model", {}))
    algorithm = dict(cfg.get("algorithm", {}))
    rollout = dict(cfg.get("rollout", {}))
    training = dict(cfg.get("training", {}))
    data = dict(cfg.get("data", {}))
    reward = dict(cfg.get("reward", {}))
    evaluation = dict(cfg.get("evaluation", {}))
    logging_cfg = dict(cfg.get("logging", {}))
    experiment = dict(cfg.get("experiment", {}))

    model_path = model.get("name_or_path", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
    max_prompt_length = int(model.get("max_prompt_length", 2048))
    max_response_length = int(model.get("max_completion_length", 2048))

    group_size = int(rollout.get("group_size", 16))
    global_batch = int(training.get("global_batch_size_prompts", 128))
    mini_batch = int(training.get("mini_batch_size_prompts", 32))
    warmup_steps = int(training.get("warmup_steps", 50))
    total_steps = int(training.get("total_steps", 200))
    seed = int(training.get("seed", 42))
    learning_rate = _as_float(training.get("learning_rate", 1e-6), 1e-6)
    weight_decay = _as_float(training.get("weight_decay", 0.01), 0.01)
    kl_coef = _as_float(algorithm.get("kl_coef", 0.001), 0.001)
    rollout_temperature = _as_float(rollout.get("temperature", 1.0), 1.0)
    rollout_top_p = _as_float(rollout.get("top_p", 1.0), 1.0)
    eval_temperature = _as_float(evaluation.get("temperature", 0.8), 0.8)

    reward_module = str(reward.get("module", "experiment4_llm.src.reward_function"))
    reward_fn_name = str(reward.get("function", "compute_score"))
    # Our legacy configs used "compute_reward"; verl reward manager expects compute_score signature.
    if reward_fn_name == "compute_reward":
        reward_fn_name = "compute_score"
    reward_module_path = _resolve_reward_module_path(reward_module, config_path=config_path)

    logger_backends: list[str] = ["console"]
    n_gpus = _count_visible_gpus()
    output_dir = str(logging_cfg.get("output_dir", "results_experiment4/verl_runs"))

    max_model_len = max_prompt_length + max_response_length
    kl_loss_type = _coerce_kl_loss_type(algorithm.get("kl_loss_type", "k1"))

    overrides: list[str] = [
        _to_hydra_override("algorithm.adv_estimator", algorithm.get("adv_estimator", "grpo")),
        _to_hydra_override("algorithm.use_kl_in_reward", False),
        _to_hydra_override("algorithm.norm_adv_by_std_in_grpo", True),
        _to_hydra_override("data.train_files", data.get("train_jsonl", "data/apps_train_intro_interview.jsonl")),
        _to_hydra_override("data.val_files", data.get("eval_jsonl", "data/apps_test_intro_interview.jsonl")),
        _to_hydra_override("data.prompt_key", "prompt"),
        _to_hydra_override("data.return_raw_chat", True),
        _to_hydra_override("data.train_batch_size", global_batch),
        _to_hydra_override("data.val_batch_size", max(1, min(global_batch, 64))),
        _to_hydra_override("data.max_prompt_length", max_prompt_length),
        _to_hydra_override("data.max_response_length", max_response_length),
        _to_hydra_override("data.filter_overlong_prompts", True),
        _to_hydra_override("data.truncation", "error"),
        _to_hydra_override("data.shuffle", True),
        _to_hydra_override("data.seed", seed),
        _to_hydra_override("data.dataloader_num_workers", 0),
        _to_hydra_override("data.custom_cls.path", "pkg://experiment4_llm.src.verl_dataset"),
        _to_hydra_override("data.custom_cls.name", "ContMaxRLDataset"),
        _to_hydra_override("actor_rollout_ref.model.path", model_path),
        _to_hydra_override("actor_rollout_ref.model.use_remove_padding", False),
        _to_hydra_override("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        _to_hydra_override("+actor_rollout_ref.model.override_config._attn_implementation", "sdpa"),
        _to_hydra_override("+actor_rollout_ref.model.override_config.attn_implementation", "sdpa"),
        _to_hydra_override("actor_rollout_ref.rollout.name", "vllm"),
        _to_hydra_override("actor_rollout_ref.rollout.mode", "async"),
        _to_hydra_override("actor_rollout_ref.rollout.temperature", rollout_temperature),
        _to_hydra_override("actor_rollout_ref.rollout.top_p", rollout_top_p),
        _to_hydra_override("actor_rollout_ref.rollout.top_k", rollout.get("top_k", -1)),
        _to_hydra_override("actor_rollout_ref.rollout.n", group_size),
        _to_hydra_override("actor_rollout_ref.rollout.prompt_length", max_prompt_length),
        _to_hydra_override("actor_rollout_ref.rollout.response_length", max_response_length),
        _to_hydra_override("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        _to_hydra_override("actor_rollout_ref.rollout.data_parallel_size", 1),
        _to_hydra_override("actor_rollout_ref.rollout.gpu_memory_utilization", 0.4),
        _to_hydra_override("actor_rollout_ref.rollout.max_model_len", max_model_len),
        _to_hydra_override("actor_rollout_ref.rollout.max_num_batched_tokens", max_model_len * max(1, group_size)),
        _to_hydra_override("actor_rollout_ref.rollout.max_num_seqs", max(64, global_batch * group_size)),
        _to_hydra_override("actor_rollout_ref.rollout.enable_chunked_prefill", False),
        _to_hydra_override("actor_rollout_ref.rollout.load_format", "safetensors"),
        _to_hydra_override("actor_rollout_ref.rollout.layered_summon", True),
        _to_hydra_override("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", max(2, mini_batch // 8)),
        _to_hydra_override("actor_rollout_ref.rollout.val_kwargs.temperature", eval_temperature),
        _to_hydra_override("actor_rollout_ref.rollout.val_kwargs.top_p", 1.0),
        _to_hydra_override("actor_rollout_ref.rollout.val_kwargs.top_k", -1),
        _to_hydra_override("actor_rollout_ref.rollout.val_kwargs.n", 1),
        _to_hydra_override("actor_rollout_ref.rollout.val_kwargs.do_sample", False),
        _to_hydra_override("actor_rollout_ref.actor.use_kl_loss", True),
        _to_hydra_override("actor_rollout_ref.actor.kl_loss_coef", kl_coef),
        _to_hydra_override("actor_rollout_ref.actor.kl_loss_type", kl_loss_type),
        _to_hydra_override("actor_rollout_ref.actor.clip_ratio_low", algorithm.get("clip_ratio_low", 0.2)),
        _to_hydra_override("actor_rollout_ref.actor.clip_ratio_high", algorithm.get("clip_ratio_high", 0.28)),
        _to_hydra_override("actor_rollout_ref.actor.ppo_epochs", algorithm.get("ppo_epochs", 1)),
        _to_hydra_override("actor_rollout_ref.actor.loss_agg_mode", algorithm.get("loss_aggregation", "token-mean")),
        _to_hydra_override("actor_rollout_ref.actor.grad_clip", training.get("gradient_clip_norm", 1.0)),
        _to_hydra_override("actor_rollout_ref.actor.data_loader_seed", seed),
        _to_hydra_override("actor_rollout_ref.actor.ppo_mini_batch_size", mini_batch),
        _to_hydra_override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", max(2, mini_batch // 8)),
        _to_hydra_override("actor_rollout_ref.actor.optim.lr", learning_rate),
        _to_hydra_override("actor_rollout_ref.actor.optim.weight_decay", weight_decay),
        _to_hydra_override("actor_rollout_ref.actor.optim.lr_warmup_steps", warmup_steps),
        _to_hydra_override("actor_rollout_ref.actor.optim.total_training_steps", total_steps),
        _to_hydra_override("actor_rollout_ref.actor.fsdp_config.fsdp_size", -1),
        _to_hydra_override("actor_rollout_ref.ref.log_prob_micro_batch_size", max(2, mini_batch // 8)),
        _to_hydra_override("reward.num_workers", 8),
        _to_hydra_override("reward.custom_reward_function.path", reward_module_path),
        _to_hydra_override("reward.custom_reward_function.name", reward_fn_name),
        _to_hydra_override(
            "+reward.custom_reward_function.reward_kwargs.timeout_seconds",
            reward.get("timeout_seconds", 5),
        ),
        _to_hydra_override(
            "+reward.custom_reward_function.reward_kwargs.memory_limit_mb",
            reward.get("memory_limit_mb", 256),
        ),
        _to_hydra_override("trainer.project_name", "cont_maxrl"),
        _to_hydra_override("trainer.experiment_name", experiment.get("name", "experiment4_llm")),
        _to_hydra_override("trainer.logger", logger_backends),
        _to_hydra_override("trainer.default_local_dir", output_dir),
        _to_hydra_override("trainer.val_before_train", False),
        _to_hydra_override("trainer.total_training_steps", total_steps),
        _to_hydra_override("trainer.total_epochs", 1),
        _to_hydra_override("trainer.save_freq", -1),
        _to_hydra_override("trainer.test_freq", -1),
        _to_hydra_override("trainer.nnodes", 1),
        _to_hydra_override("trainer.n_gpus_per_node", n_gpus),
        _to_hydra_override("+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1", "1"),
    ]
    return overrides


def rewrite_argv_for_verl(argv: list[str]) -> list[str]:
    """
    Convert ``--config <path>`` into Hydra-style ``--config-path/--config-name``.
    """
    # Backward-compatible helper retained for tests and native verl config mode.
    return rewrite_argv_for_native_verl(argv)


def rewrite_argv_for_native_verl(argv: list[str]) -> list[str]:
    config_path = parse_config_path(argv)
    if config_path is None:
        raise ValueError("Missing --config <path> in argv")
    abs_cfg = config_path.resolve()
    cleaned = _strip_config_args(argv)
    return [
        "--config-path",
        str(abs_cfg.parent),
        "--config-name",
        abs_cfg.stem,
        *cleaned,
    ]


def load_adv_estimator_from_config(config_path: Path) -> str:
    cfg = _load_yaml_dict(config_path)
    algorithm = cfg.get("algorithm", {})
    method = str(algorithm.get("adv_estimator", "grpo"))
    if method not in SUPPORTED_ADV_ESTIMATORS:
        allowed = ", ".join(sorted(SUPPORTED_ADV_ESTIMATORS))
        raise ValueError(
            f"Unsupported adv_estimator={method!r} in {config_path}. Allowed: {allowed}"
        )
    return method


def _compute_adv_from_rewards_tensor(
    rewards: torch.Tensor,
    method: AdvantageMethod,
    eps: float = 1e-8,
) -> torch.Tensor:
    if rewards.ndim == 1:
        return compute_advantage_1d(rewards, method=method, eps=eps)
    if rewards.ndim == 2:
        return compute_grouped_advantages(rewards, method=method, eps=eps)
    raise ValueError(f"Unsupported rewards ndim={rewards.ndim}; expected 1D or 2D.")


def _canonicalize_group_indices(index: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if index is None:
        return torch.arange(batch_size, dtype=torch.long, device=device)

    if isinstance(index, torch.Tensor):
        flat = index.reshape(-1)
        if flat.numel() != batch_size:
            raise ValueError(f"index length mismatch: got {flat.numel()}, expected {batch_size}")
        if flat.dtype in (torch.int32, torch.int64):
            return flat.to(device=device, dtype=torch.long)
        if torch.is_floating_point(flat):
            return flat.to(device=device, dtype=torch.long)
        index_list = [x.item() for x in flat]
    else:
        arr = np.asarray(index).reshape(-1)
        if arr.size != batch_size:
            raise ValueError(f"index length mismatch: got {arr.size}, expected {batch_size}")
        index_list = arr.tolist()

    remap: dict[str, int] = {}
    mapped: list[int] = []
    for raw in index_list:
        key = str(raw)
        if key not in remap:
            remap[key] = len(remap)
        mapped.append(remap[key])

    return torch.tensor(mapped, dtype=torch.long, device=device)


def _make_verl_outcome_advantage_fn(method: AdvantageMethod):
    def _fn(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: Any = None,
        epsilon: float = 1e-6,
        config: Any = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del config  # unused, kept for compatibility
        if token_level_rewards.ndim != 2 or response_mask.ndim != 2:
            raise ValueError(
                "Expected token_level_rewards and response_mask to be rank-2 "
                f"(got {token_level_rewards.shape}, {response_mask.shape})"
            )
        if token_level_rewards.shape != response_mask.shape:
            raise ValueError(
                "token_level_rewards and response_mask must have identical shapes "
                f"(got {token_level_rewards.shape} vs {response_mask.shape})"
            )

        mask = response_mask.to(dtype=token_level_rewards.dtype)
        seq_rewards = (token_level_rewards * mask).sum(dim=-1)
        group_idx = _canonicalize_group_indices(
            index=index,
            batch_size=seq_rewards.shape[0],
            device=seq_rewards.device,
        )

        seq_adv = compute_advantages_from_group_indices(
            rewards=seq_rewards,
            group_indices=group_idx,
            method=method,
            eps=float(epsilon),
        )
        token_adv = broadcast_sequence_advantages_to_tokens(
            seq_adv,
            token_mask=response_mask,
        )
        return token_adv, token_adv

    _fn.__name__ = f"{method}_outcome_advantage"
    return _fn


def register_custom_estimators_in_verl() -> bool:
    """
    Register custom advantage estimators into verl's advantage registry.

    The registered functions follow verl's expected signature:
      fn(token_level_rewards, response_mask, index, ...) -> (advantages, returns)
    """
    core_algos = importlib.import_module("verl.trainer.ppo.core_algos")
    registry = getattr(core_algos, "ADV_ESTIMATOR_REGISTRY", None)
    if not isinstance(registry, dict):
        warnings.warn(
            "Could not find ADV_ESTIMATOR_REGISTRY in verl.trainer.ppo.core_algos. "
            "Custom advantage estimators were not registered.",
            stacklevel=2,
        )
        return False

    registry["maxrl_unnorm"] = _make_verl_outcome_advantage_fn("maxrl_unnorm")
    registry["maxrl_norm"] = _make_verl_outcome_advantage_fn("maxrl_norm")
    registry["reinforce_baseline"] = _make_verl_outcome_advantage_fn("reinforce_baseline")
    return True


def _run_verl_main() -> None:
    trainer_mod = importlib.import_module("verl.trainer.main_ppo")
    entry = getattr(trainer_mod, "main", None)
    if callable(entry):
        entry()
        return

    # Fallback for unusual versions.
    callables = [obj for _, obj in inspect.getmembers(trainer_mod, callable)]
    if not callables:
        raise RuntimeError("Could not locate a callable entrypoint in verl.trainer.main_ppo")
    callables[0]()


def _patch_vllm_run_headless_compat() -> None:
    """
    Some verl versions import ``run_headless`` from vLLM, while older vLLM
    releases only expose ``run_server``. Patch the symbol preemptively.
    """
    try:
        serve_mod = importlib.import_module("vllm.entrypoints.cli.serve")
    except Exception:
        return

    if hasattr(serve_mod, "run_headless"):
        return

    run_server = getattr(serve_mod, "run_server", None)
    if callable(run_server):
        setattr(serve_mod, "run_headless", run_server)


def _rewrite_argv_for_config(argv: list[str], config_path: Path) -> list[str]:
    cfg = _load_yaml_dict(config_path)

    if not _is_experiment4_config(cfg):
        # Native verl/Hydra config file: direct pass-through.
        return rewrite_argv_for_native_verl(argv)

    normalized_overrides = _normalize_extra_overrides(argv)
    legacy_overrides, hydra_passthrough = _split_legacy_and_hydra_overrides(normalized_overrides)
    for key, value in legacy_overrides.items():
        _apply_legacy_override(cfg, key, value)

    trainer_mod = importlib.import_module("verl.trainer.main_ppo")
    trainer_config_dir = Path(trainer_mod.__file__).resolve().parent / "config"
    base_overrides = _build_verl_overrides_from_experiment_cfg(cfg, config_path=config_path)

    return [
        "--config-path",
        str(trainer_config_dir),
        "--config-name",
        "ppo_trainer",
        *base_overrides,
        *hydra_passthrough,
    ]


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    config_path = parse_config_path(argv)
    if config_path is None:
        raise SystemExit(
            "Missing --config <path>. "
            "Usage: python -m experiment4_llm.src.verl_main_ppo --config <cfg.yaml> [...args]"
        )
    if not config_path.exists():
        raise SystemExit(f"Config path does not exist: {config_path}")

    method = load_adv_estimator_from_config(config_path)
    if method in {"maxrl_unnorm", "maxrl_norm", "reinforce_baseline"}:
        register_custom_estimators_in_verl()

    os.environ.setdefault("VLLM_USE_V1", "1")
    _patch_vllm_run_headless_compat()
    rewritten_argv = _rewrite_argv_for_config(argv, config_path=config_path)
    sys.argv = [sys.argv[0], *rewritten_argv]
    _run_verl_main()


if __name__ == "__main__":
    main()
