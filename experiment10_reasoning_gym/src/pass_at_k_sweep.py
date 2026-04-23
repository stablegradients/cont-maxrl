"""Multi-config pass@k sweep that shares a single vLLM engine across configs.

Amortizes the ~40s model-load + compile overhead. Intended for running
one shard of a larger sweep per GPU.

Usage:
    python -m experiment10_reasoning_gym.src.pass_at_k_sweep \\
        --configs 'word_sorting:5,word_sorting:6,knights_knaves:4' \\
        --num_prompts 100 --num_samples 32 \\
        --max_response_tokens 1024

Writes one JSON per config to experiment10_reasoning_gym/results/pass_at_k/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import reasoning_gym

from experiment10_reasoning_gym.src.generate_data import (
    ANSWER_INSTRUCTION,
    _difficulty_kwargs,
)
from experiment10_reasoning_gym.src.reward_manager import (
    _score_continuous,
    extract_answer,
)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True,
                   help="Comma-separated list of 'task:difficulty' pairs")
    p.add_argument("--num_prompts", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--k_values", type=str, default="1,4,16,32")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_response_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--output_root", default="experiment10_reasoning_gym/results/pass_at_k")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = p.parse_args()

    configs = []
    for s in args.configs.split(","):
        task, diff = s.split(":")
        configs.append((task.strip(), int(diff)))

    k_values = sorted({int(k) for k in args.k_values.split(",")})
    assert max(k_values) <= args.num_samples

    # Load vLLM once
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

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task, difficulty in configs:
        kwargs = _difficulty_kwargs(task, difficulty)
        ds = reasoning_gym.create_dataset(task, size=args.num_prompts,
                                          seed=args.seed, **kwargs)
        prompts = [ds[i]["question"] + ANSWER_INSTRUCTION for i in range(args.num_prompts)]
        entries = [ds[i] for i in range(args.num_prompts)]

        print(f"\n>>> {task} diff={difficulty}: {args.num_samples}x{args.num_prompts}")
        results = llm.generate(prompts, sp)

        per_prompt_correct = []
        per_prompt_mean = []
        for i, res in enumerate(results):
            entry = entries[i]
            correct = 0
            scores = []
            for out in res.outputs:
                extracted = extract_answer(out.text)
                s = _score_continuous(task, extracted, entry)
                scores.append(float(s))
                if s == 1.0:
                    correct += 1
            per_prompt_correct.append(correct)
            per_prompt_mean.append(float(np.mean(scores)))

        pass_ks = {}
        for k in k_values:
            vals = [pass_at_k(args.num_samples, c, k) for c in per_prompt_correct]
            pass_ks[f"pass@{k}"] = float(np.mean(vals))
        mean_continuous = float(np.mean(per_prompt_mean))

        out_path = out_dir / f"{task}_diff{difficulty}.json"
        payload = {
            "task": task, "difficulty": difficulty,
            "num_prompts": args.num_prompts, "num_samples": args.num_samples,
            "temperature": args.temperature, "model": args.model,
            "pass_at_k": pass_ks,
            "mean_continuous_reward": mean_continuous,
            "per_prompt_correct": per_prompt_correct,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  pass@k={pass_ks}  mean_continuous={mean_continuous:.4f}  -> {out_path}")


if __name__ == "__main__":
    main()
