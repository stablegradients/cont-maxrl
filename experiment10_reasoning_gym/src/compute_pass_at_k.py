"""Pre-training pass@k evaluation of Qwen3-0.6B-Base on reasoning_gym tasks.

Uses vLLM to generate ``num_samples`` completions per prompt, then computes
pass@k using the unbiased estimator from Chen et al. 2021 (HumanEval).

Purpose: validate the sparsity claim — at higher difficulty levels, pass@1
should drop sharply, making binary reward very sparse for RL training.

Usage:
    python -m experiment10_reasoning_gym.src.compute_pass_at_k \\
        --task word_sorting --difficulty 10 \\
        --num_prompts 200 --num_samples 64 --k_values 1,4,16,64 \\
        --temperature 1.0

Output:
    experiment10_reasoning_gym/results/pass_at_k/{task}_diff{difficulty}.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from experiment10_reasoning_gym.src.generate_data import (
    ANSWER_INSTRUCTION,
    _difficulty_kwargs,
)
from experiment10_reasoning_gym.src.reward_manager import _score_continuous, extract_answer


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["word_sorting", "sentence_reordering",
                                 "knights_knaves", "game_of_life"])
    parser.add_argument("--difficulty", type=int, required=True)
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--k_values", type=str, default="1,4,16,64")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_response_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--output_root", default="experiment10_reasoning_gym/results/pass_at_k")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    args = parser.parse_args()

    k_values = sorted({int(k) for k in args.k_values.split(",")})
    assert max(k_values) <= args.num_samples, "num_samples must be >= max(k_values)"

    # Build prompts from reasoning_gym (disjoint from train seed = 42)
    import reasoning_gym
    kwargs = _difficulty_kwargs(args.task, args.difficulty)
    ds = reasoning_gym.create_dataset(args.task, size=args.num_prompts,
                                      seed=args.seed, **kwargs)
    prompts = [ds[i]["question"] + ANSWER_INSTRUCTION for i in range(args.num_prompts)]
    entries = [ds[i] for i in range(args.num_prompts)]

    # vLLM
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_tokens,
        seed=args.seed,
    )
    print(f"Generating {args.num_samples} samples x {args.num_prompts} prompts ...")
    results = llm.generate(prompts, sp)

    # Score each sample
    per_prompt_correct = []
    per_prompt_mean = []
    for i, res in enumerate(results):
        entry = entries[i]
        correct = 0
        scores = []
        for out in res.outputs:
            extracted = extract_answer(out.text)
            s = _score_continuous(args.task, extracted, entry)
            scores.append(float(s))
            if s == 1.0:
                correct += 1
        per_prompt_correct.append(correct)
        per_prompt_mean.append(float(np.mean(scores)))

    # pass@k
    n = args.num_samples
    pass_ks = {}
    for k in k_values:
        vals = [pass_at_k(n, c, k) for c in per_prompt_correct]
        pass_ks[f"pass@{k}"] = float(np.mean(vals))
    mean_continuous = float(np.mean(per_prompt_mean))

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_diff{args.difficulty}.json"
    payload = {
        "task": args.task,
        "difficulty": args.difficulty,
        "num_prompts": args.num_prompts,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "model": args.model,
        "pass_at_k": pass_ks,
        "mean_continuous_reward": mean_continuous,
        "per_prompt_correct": per_prompt_correct,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n== {args.task} diff={args.difficulty} ==")
    for k, v in pass_ks.items():
        print(f"  {k} = {v:.4f}")
    print(f"  mean_continuous = {mean_continuous:.4f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
