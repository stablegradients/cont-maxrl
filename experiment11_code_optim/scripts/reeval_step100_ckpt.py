"""Standalone re-evaluation of a verl-saved ckpt on pie_val_200.

Builds an HF-format model from the FSDP shards + HF snapshot bundled
inside ${ckpt}/actor/, generates 256 completions per problem with vLLM,
grades via PIE's compute_scores_batch (global_timeout=None so we don't
silently zero out like the in-training insitu_eval did), then writes a
metrics JSON and optionally pushes the numbers to W&B under a separate
run for cross-arm comparison.

Usage:
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/scripts/reeval_step100_ckpt.py \\
        --ckpt /work/.../qwen3_1.7b_instruct_pie_maxrl_unnorm_g16_s0/global_step_100 \\
        --arm maxrl_unnorm \\
        --step 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to global_step_N dir (contains actor/huggingface)")
    ap.add_argument("--arm", required=True, help="maxrl_unnorm | grpo | rloo")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--val_parquet", default="/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_val_200.parquet")
    ap.add_argument("--n_completions", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=6656)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    hf_dir = ckpt / "actor" / "huggingface"
    if not hf_dir.is_dir():
        print(f"FATAL: no HF snapshot at {hf_dir}", file=sys.stderr)
        return 2

    os.environ.setdefault("PIE_BACKEND", "perf")
    os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
    os.environ.setdefault("PIE_PER_TEST_TIMEOUT", "30")
    os.environ.setdefault("PIE_TEST_CASE_DIR", "/work/nvme/bgmo/sramasubramanian/pie_data/tests")
    os.environ.setdefault("PIE_TEST_CASE_PICKLE", "/work/nvme/bgmo/sramasubramanian/pie_data/tests.pkl")
    os.environ.setdefault("VLLM_USE_V1", "0")

    # PIE_MODE shim so insitu_eval.score_and_save's hardcoded
    # `from experiment4_llm.src.reward_function import compute_scores_batch`
    # resolves to PIE's reward fn.
    os.environ["PIE_MODE"] = "1"
    from experiment11_code_optim.reward import pie_reward_fn
    sys.modules["experiment4_llm.src.reward_function"] = pie_reward_fn

    # PCH setup for fast compile under -O0
    from experiment11_code_optim.measurement.core import setup_pch
    os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
    print(f"[setup] PCH built (-O0)", flush=True)

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # ---- load val_200 prompts ----
    df = pd.read_parquet(args.val_parquet)
    print(f"[load] {len(df)} val rows", flush=True)
    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)

    prompts = []
    ground_truths = []
    for _, row in df.iterrows():
        msgs = list(row["prompt"])
        rendered = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        # drop overlong (insitu_eval filters at >2560 tokens via
        # data.filter_overlong_prompts=True; mirror that here so prompt counts
        # match training-time eval)
        n_tok = len(tok(rendered, add_special_tokens=False)["input_ids"])
        if n_tok > 2560:
            continue
        prompts.append(rendered)
        ground_truths.append(row["reward_model"]["ground_truth"])
    n_problems = len(prompts)
    print(f"[load] {n_problems} prompts after overlong filter", flush=True)

    # ---- load model (HF snapshot inside ckpt) ----
    print(f"[load] vLLM from {hf_dir}", flush=True)
    t0 = time.monotonic()
    llm = LLM(
        model=str(hf_dir),
        tokenizer=str(hf_dir),
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        max_num_seqs=128,
        enforce_eager=False,
        dtype="bfloat16",
    )
    print(f"[load] vLLM ready in {time.monotonic()-t0:.0f}s", flush=True)

    sp = SamplingParams(
        n=args.n_completions,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    # ---- generate ----
    print(f"[gen] {args.n_completions}×{n_problems} = {args.n_completions*n_problems} completions", flush=True)
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sp)
    gen_s = time.monotonic() - t0
    print(f"[gen] done in {gen_s:.0f}s", flush=True)

    # ---- bundle for score_and_save ----
    completions = []
    for out in outputs:
        completions.append([c.text for c in out.outputs])

    # ---- score with our fixed score_and_save ----
    from experiment4_llm.src.insitu_eval import score_and_save

    label = f"reeval_step{args.step}_{args.arm}"
    out_dir = (
        Path("/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/docs")
        / "reeval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[score] running score_and_save (global_timeout off)", flush=True)
    t0 = time.monotonic()
    metrics = score_and_save(
        completions=completions,
        ground_truths=ground_truths,
        epoch=args.step // 272,
        global_step=args.step,
        output_dir=str(out_dir / args.arm),
        n_target=args.n_completions,
        k_values=[1, 4, 16, 64, 256],
        timeout_per_test=30,
        n_workers=64,
    )
    print(f"[score] done in {time.monotonic()-t0:.0f}s", flush=True)

    summary = metrics.get("summary", {})
    pk = metrics.get("pass_at_k") or {}
    bsk = metrics.get("best_speedup_at_k") or {}
    print()
    print(f"=== {args.arm} step {args.step} ===")
    for k in [1, 4, 16, 64, 256]:
        v = pk.get(str(k))
        bsv = bsk.get(str(k))
        if v is not None:
            print(f"  pass@{k:>3} = {v:.4f}    best_speedup@{k:>3} = {bsv if bsv is None else f'{bsv:.3f}'}")
    print(f"  frac_problems_with_speedup = {summary.get('frac_problems_with_speedup')}")
    print(f"  frac_problems_mode2        = {summary.get('frac_problems_mode2')}")
    print(f"  mean_speedup_given_pass    = {summary.get('mean_speedup_given_pass')}")
    print(f"  n_problems_solved          = {summary.get('n_problems_solved')}/{summary.get('total_problems')}")

    # ---- W&B push: same group as the corresponding training run + same
    # metric prefix that _insitu_eval uses (ray_trainer.py:893, label defaults
    # to "livecodebench_v5") so the reeval point overlays the training curve.
    if not args.no_wandb:
        try:
            import wandb
            arm_to_group = {
                "maxrl_unnorm": "qwen3-1.7b-instruct_pie_maxrl_unnorm_G16",
                "grpo":          "qwen3-1.7b-instruct_pie_grpo_G16",
                "rloo":          "qwen3-1.7b-instruct_pie_rloo_G16",
            }
            group = arm_to_group[args.arm]
            run = wandb.init(
                project="cont-maxrl-pie",
                name=f"{group}_s0_reeval_step{args.step}",
                group=group,
                tags=["reeval", args.arm, f"step{args.step}"],
                config={
                    "arm": args.arm,
                    "step": args.step,
                    "ckpt": str(ckpt),
                    "n_completions": args.n_completions,
                    "n_problems": n_problems,
                },
                reinit=True,
            )
            prefix = "insitu_eval/pie_val_200"
            log = {f"{prefix}/pass_at_{k}": pk.get(str(k))
                   for k in [1, 4, 16, 64, 256] if pk.get(str(k)) is not None}
            log.update({f"{prefix}/best_speedup_at_{k}": bsk.get(str(k))
                        for k in [1, 4, 16, 64, 256] if bsk.get(str(k)) is not None})
            for key in ("mean_pass_rate", "mean_score", "n_problems_solved",
                        "frac_problems_with_speedup", "frac_problems_mode2",
                        "mean_speedup_given_pass"):
                v = summary.get(key)
                if v is not None:
                    log[f"{prefix}/{key}"] = v
            log["training/global_step"] = args.step
            wandb.log(log, step=args.step)
            run.finish()
            print(f"[wandb] logged {len(log)} metrics to {group}_s0_reeval_step{args.step}", flush=True)
        except Exception as e:
            print(f"[wandb] failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
