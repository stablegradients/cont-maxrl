"""Per-step PIE metrics injected via a monkey-patch on
`verl.trainer.ppo.metric_utils.compute_data_metrics`.

The PIE reward function returns per-rollout dicts of {"score", "acc",
"speedup"}; verl's NaiveRewardManager pivots every key in that dict into
`batch.non_tensor_batch[<key>]`. This hook reads acc/speedup from there
and emits the PIE-specific metrics this experiment cares about:

    critic/score/gate_pass_rate           mean(acc)                  — separates "writes compiling, correct code" from "writes faster code"
    critic/score/speedup_rate             frac(score > 0)            — Mode-2 discovery rate
    critic/score/mean_speedup             mean(speedup | acc==1)     — interpretable "average X× speedup"
    critic/score/max_speedup              max(speedup)               — best algorithmic improvement in the batch
    critic/score/fraction_dead_groups     frac(uid-groups all-zero)  — gradient starvation signal
    critic/score/fraction_mode2           frac(score >= 0.3)         — speedup >= 1.23×

Without this hook, the existing critic/score/{mean,max,min,fraction_zero,...}
already give partial coverage, but `fraction_zero` conflates gate-fail and
correct-but-not-faster (since the new reward is `max(0, log2(src/cand))` and
both gate-fail and "correct but same speed" produce score=0). The
gate_pass_rate metric breaks that ambiguity.

This module is a no-op if the hook is not installed (call
`install_pie_metrics_hook()` from train.py under PIE_MODE=1).
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np

_INSTALLED = False


def install_pie_metrics_hook() -> None:
    """Idempotent. Wraps `verl.trainer.ppo.metric_utils.compute_data_metrics`
    so PIE-specific metrics are emitted alongside the verl-built-in ones.

    Patches TWO targets: the module attribute (`metric_utils.compute_data_metrics`)
    AND the already-bound reference in `ray_trainer` (which did
    `from … import compute_data_metrics` and so holds its own local binding).
    Without the second patch, the monkey-patch on the module attribute has
    no effect on the function ray_trainer actually calls — confirmed via
    pie-efficiency commit afce125 which had only the module patch and emitted
    zero PIE metrics in the W&B step log.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from verl.trainer.ppo import metric_utils

    original = metric_utils.compute_data_metrics
    first_call = {"done": False}

    def wrapped(batch, use_critic: bool = True) -> dict[str, Any]:
        metrics = original(batch, use_critic=use_critic)
        try:
            extras = _compute_pie_metrics(batch)
            metrics.update(extras)
            if not first_call["done"]:
                first_call["done"] = True
                print(
                    f"[pie_metrics] wrapped() first call: added "
                    f"{len(extras)} PIE metrics → {sorted(extras)}",
                    flush=True,
                )
        except Exception as e:  # never crash the trainer on metric computation
            import traceback
            print(f"[pie_metrics] WARN: PIE metrics failed: {e}", flush=True)
            traceback.print_exc()
        return metrics

    metric_utils.compute_data_metrics = wrapped

    # ray_trainer.py does `from verl.trainer.ppo.metric_utils import compute_data_metrics`
    # at module load, creating a separate local binding. Patch that too.
    try:
        from verl.trainer.ppo import ray_trainer
        ray_trainer.compute_data_metrics = wrapped
        rt_patched = True
    except ImportError:
        rt_patched = False

    _INSTALLED = True
    print(
        f"[pie_metrics] compute_data_metrics hook installed "
        f"(metric_utils ✓, ray_trainer {'✓' if rt_patched else 'pending'})",
        flush=True,
    )


def _compute_pie_metrics(batch) -> dict[str, float]:
    """Build the PIE-specific metrics dict. Returns empty if non_tensor_batch
    doesn't carry acc/speedup (i.e., we're running outside PIE mode)."""
    ntb = batch.non_tensor_batch
    accs = ntb.get("acc")
    speedups = ntb.get("speedup")
    if accs is None or speedups is None:
        return {}

    # token_level_scores: shape (batch, response_len); reward is at the last
    # response token, sum-over-tokens recovers the scalar reward per rollout.
    rewards = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()

    accs_arr = np.asarray(accs, dtype=float)
    speedups_arr = np.asarray(speedups, dtype=float)
    rewards_arr = np.asarray(rewards, dtype=float)
    n = len(accs_arr)
    if n == 0:
        return {}

    gate_pass_mask = accs_arr >= 1.0
    n_gate_pass = int(gate_pass_mask.sum())

    metrics: dict[str, float] = {
        "critic/score/gate_pass_rate": float(accs_arr.mean()),
        "critic/score/speedup_rate": float((rewards_arr > 0).mean()),
        "critic/score/fraction_mode2": float((rewards_arr >= 0.3).mean()),
        "critic/score/max_speedup": float(speedups_arr.max()),
    }
    if n_gate_pass > 0:
        metrics["critic/score/mean_speedup"] = float(speedups_arr[gate_pass_mask].mean())
    else:
        metrics["critic/score/mean_speedup"] = 0.0

    # fraction_dead_groups: a prompt-group is "dead" iff every rollout in the
    # group has reward 0 (gradient starvation under group-relative estimators).
    # Group key = uid (verl assigns one uid per prompt before .repeat()).
    uids = ntb.get("uid")
    if uids is not None and len(uids) == n:
        groups: dict[Any, list[float]] = defaultdict(list)
        for u, r in zip(uids, rewards_arr):
            groups[u].append(r)
        n_groups = len(groups)
        if n_groups > 0:
            dead = sum(1 for v in groups.values() if all(x == 0 for x in v))
            metrics["critic/score/fraction_dead_groups"] = dead / n_groups
            metrics["critic/score/n_groups"] = float(n_groups)

    return metrics
