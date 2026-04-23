"""verl-compatible reward functions for Reasoning Gym tasks.

Exposes two entry points used by verl's ``custom_reward_function`` machinery:

- ``compute_score_continuous(solution_str, ground_truth, data_source, extra_info)``
  Returns the continuous score_answer value directly (partial credit).

- ``compute_score_binary(solution_str, ground_truth, data_source, extra_info)``
  Returns ``float(continuous_score == 1.0)`` — sparse binary reward.

Both always log the same dict of extras so W&B can compare continuous vs
binary metrics regardless of which mode drove training. See spec §9.2.

Task dispatch uses ``extra_info['task']``. Each parquet row must carry:

    data_source  = "reasoning_gym_{task}"
    reward_model = {"ground_truth": entry['answer']}
    extra_info   = {"task": task, "difficulty": <int|str>, "entry": json.dumps(entry)}

The ``entry`` field is JSON-encoded so it survives parquet roundtripping
(nested dicts in pandas-> pyarrow can be lossy).

K&K reward
----------
reasoning_gym's built-in K&K ``score_answer`` already enforces
``len(predicted) == len(oracle)`` (anti-hack). We deliberately reimplement a
simpler matching-fraction formula per spec §3.2:

    1. Parse both answers into a set of (name, role) tuples via
       reasoning_gym's ``_normalize_answer``.
    2. If ``len(pred) != len(oracle)`` -> 0.0.
    3. Otherwise -> ``len(pred & oracle) / len(oracle)`` (no 0.3 base).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(solution_str: Optional[str]) -> str:
    """Extract content between <answer>...</answer> tags.

    Falls back to the full string if no tags are present (reward will
    typically be low in that case but we do not crash).
    """
    if solution_str is None:
        return ""
    m = _ANSWER_TAG_RE.search(solution_str)
    if m is not None:
        return m.group(1).strip()
    return solution_str.strip()


# ---------------------------------------------------------------------------
# Per-task scorers
# ---------------------------------------------------------------------------


def _knk_fixed_score(answer_str: str, entry: dict) -> float:
    """Spec §3.2 K&K reward: matching/total with strict length check."""
    from reasoning_gym.logic.knights_knaves import KnightsKnavesDataset

    if not isinstance(answer_str, str) or not answer_str:
        return 0.0
    try:
        oracle = KnightsKnavesDataset._normalize_answer(entry["answer"])
        pred = KnightsKnavesDataset._normalize_answer(answer_str)
    except Exception:
        return 0.0
    if not oracle:
        return 0.0
    if len(pred) != len(oracle):
        return 0.0
    return len(pred & oracle) / len(oracle)


_TASK_DATASETS: dict[str, Any] = {}


def _get_scorer(task: str):
    """Lazily construct a dataset instance per task; reused across calls.

    reasoning_gym ``score_answer`` methods only read state from ``entry``,
    so a single dataset instance can score any entry for that task.
    """
    if task in _TASK_DATASETS:
        return _TASK_DATASETS[task]
    import reasoning_gym

    # size/seed irrelevant — we only use the instance's score_answer method.
    _TASK_DATASETS[task] = reasoning_gym.create_dataset(task, size=1, seed=0)
    return _TASK_DATASETS[task]


def _score_continuous(task: str, answer_str: str, entry: dict) -> float:
    """Raw continuous score in [0, 1] for the given task."""
    if task == "knights_knaves":
        return _knk_fixed_score(answer_str, entry)
    # Generic: use reasoning_gym's built-in score_answer
    return float(_get_scorer(task).score_answer(answer_str, entry))


# ---------------------------------------------------------------------------
# verl compute_score entry points
# ---------------------------------------------------------------------------


def _compute_score_impl(solution_str, ground_truth, data_source, extra_info, *, binary: bool):
    """Shared impl: return dict with score, continuous, binary, fraction_perfect, format_ok."""
    extra_info = extra_info or {}
    task = extra_info.get("task")
    raw_entry = extra_info.get("entry")

    if task is None or raw_entry is None:
        # Parquet missing metadata -> crash loudly rather than silently score 0.
        raise ValueError(
            f"reward_manager: extra_info missing 'task' or 'entry' "
            f"(data_source={data_source!r}, extra_info_keys={list(extra_info)})"
        )

    try:
        entry = json.loads(raw_entry) if isinstance(raw_entry, str) else raw_entry
    except Exception as e:
        raise ValueError(f"reward_manager: could not decode entry JSON: {e}") from e

    extracted = extract_answer(solution_str)
    format_ok = 1.0 if (solution_str and "<answer>" in solution_str.lower()) else 0.0

    continuous = _score_continuous(task, extracted, entry)
    continuous = max(0.0, min(1.0, float(continuous)))
    binary_val = 1.0 if continuous == 1.0 else 0.0

    score = binary_val if binary else continuous

    return {
        "score": float(score),
        "continuous": float(continuous),
        "binary": float(binary_val),
        "fraction_perfect": float(binary_val),
        "format_ok": float(format_ok),
    }


def compute_score_continuous(solution_str, ground_truth, data_source=None, extra_info=None):
    """Continuous reward: score = score_answer value (partial credit)."""
    return _compute_score_impl(solution_str, ground_truth, data_source, extra_info, binary=False)


def compute_score_binary(solution_str, ground_truth, data_source=None, extra_info=None):
    """Binary reward: score = 1.0 iff continuous score == 1.0."""
    return _compute_score_impl(solution_str, ground_truth, data_source, extra_info, binary=True)


# Convenience alias used by verl's default loader when
# ``custom_reward_function.name`` is omitted.
compute_score = compute_score_continuous


__all__ = [
    "extract_answer",
    "compute_score",
    "compute_score_continuous",
    "compute_score_binary",
]
