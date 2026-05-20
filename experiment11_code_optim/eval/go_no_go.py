"""
§9 GO/NO-GO check on the SFT cold-start (spec v9 §9).

Loads the SFT-tuned actor, generates 16 completions per problem on
pie_val_200.parquet, grades them via PIE's compute_scores_batch, and
reports:

    GO     if  ≥ 20 % of the 200 problems have ≥ 8 / 16 completions
              passing the strict gate (acc == 1.0)
    NO-GO  otherwise

A NO-GO means the SFT didn't produce a learnable policy — the user must
either retrain with more data / epochs, or pick a different base model
before launching the (expensive) RL chains.

Run:
    PIE_TEST_CASE_DIR=/work/nvme/bgmo/.../pie_data/tests \\
    PIE_REFERENCE_TABLE_DIR=/work/nvme/bgmo/.../pie_data/reference_table \\
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        python -m experiment11_code_optim.eval.go_no_go \\
        --sft_ckpt /work/nvme/bgmo/.../sft_ckpts/.../final \\
        --val_parquet /work/nvme/bgmo/.../pie_val_200.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--n_completions", type=int, default=16,
                    help="Spec §9: G=16, must match RL rollout n.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Spec §9: training rollout temperature, NOT eval 0.6.")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=8192,
                    help="vLLM context cap. Prompts that exceed this get DROPPED "
                         "(see scheduler.py:1106). val_200 max prompt is 3330 tok, "
                         "so 8192 = 3330 + 4096 + headroom fits everything.")
    ap.add_argument("--max_num_seqs", type=int, default=128,
                    help="vLLM max concurrent sequences. 128 = ~user-requested batch size.")
    ap.add_argument("--stop_strings", type=str, nargs="*",
                    default=None,
                    help="Optional sampling stop strings (recommended for base models that "
                         "don't terminate naturally, e.g., ['\\n```\\n', '\\n\\n###']).")
    ap.add_argument("--pass_threshold", type=int, default=8,
                    help="A problem 'passes' if ≥ this many of n_completions pass.")
    ap.add_argument("--go_fraction", type=float, default=0.20,
                    help="GO if fraction-of-passing-problems ≥ this.")
    ap.add_argument("--output_json", default=None,
                    help="Write the per-problem detail + verdict here.")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Start row in val parquet (for data-parallel sharding).")
    ap.add_argument("--end_idx", type=int, default=None,
                    help="End row in val parquet (exclusive).")
    args = ap.parse_args()

    # ---- Load val problems ----
    df = pd.read_parquet(args.val_parquet)
    if args.end_idx is None:
        args.end_idx = len(df)
    df = df.iloc[args.start_idx:args.end_idx].reset_index(drop=True)
    print(f"[go_no_go] shard rows [{args.start_idx}, {args.end_idx})")
    prompts: list[str] = []
    ground_truths: list[str] = []
    pids: list[str] = []
    for _, row in df.iterrows():
        prompt_text = next(
            (m["content"] for m in row["prompt"] if m.get("role") == "user"),
            None,
        )
        if prompt_text is None:
            continue
        prompts.append(prompt_text)
        gt = row["reward_model"]["ground_truth"]
        ground_truths.append(gt)
        try:
            pids.append(json.loads(gt)["problem_id"])
        except (json.JSONDecodeError, KeyError):
            pids.append("?")
    n_problems = len(prompts)
    print(f"[go_no_go] {n_problems} val problems loaded")

    # ---- Generate ----
    from vllm import LLM, SamplingParams
    print(f"[go_no_go] Loading SFT actor from {args.sft_ckpt}...")
    llm = LLM(
        model=args.sft_ckpt,
        tokenizer=args.sft_ckpt,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
    )
    sp_kwargs = dict(
        n=args.n_completions,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    if args.stop_strings:
        sp_kwargs["stop"] = args.stop_strings
    sp = SamplingParams(**sp_kwargs)
    t0 = time.monotonic()
    print(f"[go_no_go] Generating {args.n_completions} x {n_problems} completions "
          f"@ T={args.temperature}...")
    outputs = llm.generate(prompts, sp)
    gen_s = time.monotonic() - t0
    print(f"[go_no_go] generation in {gen_s:.0f}s "
          f"({args.n_completions*n_problems/gen_s:.0f} samples/s)")

    # ---- Score ----
    from experiment11_code_optim.reward.pie_reward_fn import (
        compute_scores_batch, preload_test_cases,
    )
    test_case_dir = os.environ["PIE_TEST_CASE_DIR"]
    reference_table_dir = os.environ.get("PIE_REFERENCE_TABLE_DIR")
    preload_test_cases(test_case_dir)

    flat_inputs = []
    owner = []  # (problem_idx, completion_idx)
    for i, out in enumerate(outputs):
        for j, comp in enumerate(out.outputs):
            flat_inputs.append({
                "solution_str": comp.text,
                "ground_truth": ground_truths[i],
            })
            owner.append((i, j))

    t0 = time.monotonic()
    print(f"[go_no_go] Scoring {len(flat_inputs)} completions...")
    scores = compute_scores_batch(
        flat_inputs,
        test_case_dir=test_case_dir,
        reference_table_dir=reference_table_dir,
    )
    print(f"[go_no_go] scoring in {time.monotonic()-t0:.0f}s")

    # ---- Aggregate per problem ----
    per_problem_passes = [0] * n_problems
    per_problem_avg_score = [0.0] * n_problems
    for (i, _), s in zip(owner, scores):
        if s.get("acc", 0.0) == 1.0:
            per_problem_passes[i] += 1
        per_problem_avg_score[i] += s.get("score", 0.0)
    per_problem_avg_score = [s / args.n_completions for s in per_problem_avg_score]

    n_passing_problems = sum(1 for c in per_problem_passes if c >= args.pass_threshold)
    fraction = n_passing_problems / max(n_problems, 1)
    verdict = "GO" if fraction >= args.go_fraction else "NO-GO"

    # Histogram of pass-counts
    from collections import Counter
    hist = Counter(per_problem_passes)
    print()
    print("=" * 60)
    print(f"§9 GO/NO-GO RESULT: {verdict}")
    print("=" * 60)
    print(f"  Problems passing threshold (≥{args.pass_threshold}/{args.n_completions}): "
          f"{n_passing_problems}/{n_problems} = {100*fraction:.1f}%")
    print(f"  Required: ≥ {100*args.go_fraction:.0f}%")
    print()
    print(f"  Pass-count distribution (passes/{args.n_completions}: # problems):")
    for k in range(args.n_completions + 1):
        c = hist.get(k, 0)
        if c == 0:
            continue
        bar = "#" * min(60, c)
        print(f"    {k:>2}/{args.n_completions}  ({c:>3}) {bar}")
    print()
    print(f"  Mean per-problem avg score: {sum(per_problem_avg_score)/max(n_problems,1):.3f}")
    n_pass_at_least_1 = sum(1 for c in per_problem_passes if c >= 1)
    print(f"  Problems with ≥1 pass:       {n_pass_at_least_1}/{n_problems} = "
          f"{100*n_pass_at_least_1/max(n_problems,1):.1f}%")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "verdict": verdict,
                "n_problems": n_problems,
                "n_completions": args.n_completions,
                "pass_threshold": args.pass_threshold,
                "go_fraction": args.go_fraction,
                "n_passing_problems": n_passing_problems,
                "fraction_passing": fraction,
                "per_problem_passes": per_problem_passes,
                "per_problem_avg_score": per_problem_avg_score,
                "per_problem_pids": pids,
                "pass_count_histogram": dict(hist),
            }, f, indent=2)
        print(f"  -> wrote {args.output_json}")

    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
