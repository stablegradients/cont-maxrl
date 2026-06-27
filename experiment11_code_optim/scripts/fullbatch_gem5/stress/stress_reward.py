"""Trivial instant reward for the 4B memory/throughput STRESS test only.

Isolates the GPU (actor fwd/bwd + vLLM generation) from the real gem5 reward so
the stress measures pure compute/memory. Returns acc/speedup keys so the PIE
metrics hook does not error. verl auto-detects the *_batch entry."""
from __future__ import annotations


def compute_scores_batch(reward_inputs, **kwargs):
    return [{"score": 0.0, "acc": 0.0, "speedup": 0.0, "reason": "stress"}
            for _ in reward_inputs]


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    return {"score": 0.0, "acc": 0.0, "speedup": 0.0, "reason": "stress"}
