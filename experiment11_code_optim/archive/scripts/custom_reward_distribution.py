"""Custom reward distribution for untrained Qwen3-0.6B on PIE val.

For each of N val problems:
  1. Measure src_instructions (the prompt's slow code) under perf backend.
  2. Generate K rollouts via vLLM (chat template, enable_thinking=False).
  3. Grade each rollout via pie_reward_fn.compute_scores_batch (perf backend).
  4. For each rollout:
       r = max(0, log_e(src_instr / gen_instr))   if all_pass
       r = 0                                       otherwise

Then plot the reward distribution (histogram) over the full set of
rollouts.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_COMPILE_SCRATCH",
                       f"/tmp/{os.environ.get('USER', 'u')}_pie_scratch")
os.makedirs(os.environ["PIE_COMPILE_SCRATCH"], exist_ok=True)

REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n_problems", type=int, default=8)
    ap.add_argument("--n_rollouts", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir",
                    default=f"{REPO}/experiment11_code_optim/docs/plots/custom_reward_qwen3_06B")
    ap.add_argument("--opt_level", choices=["O0", "O3"], default="O3",
                    help="g++ optimization level. -O0 preserves every "
                         "instruction (interpreter-like) so algorithmic "
                         "differences yield large instruction-count ratios; "
                         "-O3 normalizes most variations away.")
    ap.add_argument("--timeout_per_test", type=float, default=10.0,
                    help="Per-test grading timeout (-O0 needs more time).")
    args = ap.parse_args()

    # Monkey-patch core.COMPILE_FLAGS BEFORE setup_pch so the PCH is also
    # built at the chosen optimization level. -O0 PCH is incompatible with
    # -O3 source compile and vice versa.
    import experiment11_code_optim.measurement.core as core
    core.COMPILE_FLAGS = [f"-{args.opt_level}", "-std=c++17"]
    print(f"[setup] g++ COMPILE_FLAGS={core.COMPILE_FLAGS}", flush=True)

    from experiment11_code_optim.measurement.core import setup_pch
    os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
    print("[setup] PCH built", flush=True)

    from experiment11_code_optim.reward.pie_reward_fn import (
        grade_dedup_batch, preload_test_cases, extract_code,
    )
    # Re-bind for the rest of the function:
    globals()["grade_dedup_batch"] = grade_dedup_batch
    globals()["preload_test_cases"] = preload_test_cases
    globals()["extract_code"] = extract_code

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pick N problems from val that have src_code + reference data
    TEST_CASE_DIR = "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
    preload_test_cases(TEST_CASE_DIR)
    print(f"[setup] test cases preloaded", flush=True)

    val_jsonl = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/val.jsonl"
    picked: list[tuple[str, str]] = []  # (pid, src_code)
    seen_pids: set[str] = set()
    with open(val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("problem_id"); src = r.get("src_code")
            if not pid or not src or pid in seen_pids:
                continue
            seen_pids.add(pid)
            picked.append((pid, src))
            if len(picked) >= args.n_problems:
                break
    print(f"[setup] picked {len(picked)} val problems: "
          f"{[p[0] for p in picked]}", flush=True)

    # ---- measure src_instructions per problem ----
    src_tasks = {}
    src_meta = {}
    for pid, src in picked:
        key = hashlib.md5(f"SRC::{pid}::{src}".encode()).hexdigest()
        src_tasks[key] = {"code": src, "problem_id": pid, "split": "val"}
        src_meta[key] = pid
    print(f"[src] grading {len(src_tasks)} src programs...", flush=True)
    src_results = grade_dedup_batch(
        src_tasks, test_case_dir=TEST_CASE_DIR,
        measurement_max_tests=5, timeout_per_test=args.timeout_per_test,
        num_workers=8,
    )
    src_instr: dict[str, int] = {}
    for key, r in src_results.items():
        pid = src_meta[key]
        if r.get("all_pass") and r.get("elapsed_ns"):
            src_instr[pid] = int(r["elapsed_ns"])
        else:
            print(f"[src] WARNING: pid={pid} src failed to grade: "
                  f"compiled={r.get('compiled')} all_pass={r.get('all_pass')}",
                  flush=True)
    print(f"[src] graded {len(src_instr)}/{len(picked)} src programs", flush=True)
    print(f"[src] instructions per pid: "
          f"{ {p: src_instr.get(p, 'FAIL') for p in seen_pids} }", flush=True)

    # Drop problems with no src baseline
    picked = [(p, s) for p, s in picked if p in src_instr]
    print(f"[setup] {len(picked)} problems usable after src grading", flush=True)

    # ---- generate K rollouts per problem via vLLM ----
    print(f"[gen] loading model {args.model}...", flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        max_num_seqs=256,
        dtype="bfloat16",
        enforce_eager=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
        seed=args.seed,
    )

    # Build prompts using PIE PROMPT_TEMPLATE in chat format
    PROMPT_TEMPLATE = (
        "You are a C++ optimization expert. Below is a working C++ program. "
        "Rewrite it to run faster while producing identical output for all valid "
        "inputs. Keep the same input/output format and only change the algorithm/"
        "implementation.\n\n"
        "### Slow Version:\n"
        "```cpp\n{src_code}\n```\n\n"
        "### Optimized Version:\n"
    )
    formatted_prompts = []
    for pid, src in picked:
        user_content = PROMPT_TEMPLATE.format(src_code=src)
        text = tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        formatted_prompts.append(text)
    print(f"[gen] generating {args.n_rollouts} rollouts × "
          f"{len(formatted_prompts)} problems...", flush=True)

    sp = SamplingParams(
        n=args.n_rollouts, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_tokens=args.max_tokens,
        stop_token_ids=[151645, 151643],  # <|im_end|>, <|endoftext|>
    )
    t0 = time.monotonic()
    outputs = llm.generate(formatted_prompts, sp)
    print(f"[gen] done in {time.monotonic() - t0:.1f}s", flush=True)
    # outputs[i].outputs = list of args.n_rollouts samples
    rollouts: list[tuple[str, str, str]] = []  # (pid, sample_idx, completion_text)
    for i, out in enumerate(outputs):
        pid = picked[i][0]
        for j, s in enumerate(out.outputs):
            rollouts.append((pid, str(j), s.text))
    print(f"[gen] {len(rollouts)} total completions", flush=True)

    # Free GPU memory before CPU-heavy grading
    del llm
    import torch
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ---- grade every rollout under perf ----
    grade_tasks = {}
    grade_meta = {}
    for pid, j, text in rollouts:
        code = extract_code(text)
        if not code:
            continue
        key = hashlib.md5(f"GEN::{pid}::{j}::{code}".encode()).hexdigest()
        grade_tasks[key] = {"code": code, "problem_id": pid, "split": "val"}
        grade_meta[key] = (pid, j)
    print(f"[grade] {len(grade_tasks)} parseable completions to grade "
          f"(out of {len(rollouts)} rollouts)", flush=True)
    t0 = time.monotonic()
    grade_results = grade_dedup_batch(
        grade_tasks, test_case_dir=TEST_CASE_DIR,
        measurement_max_tests=5, timeout_per_test=args.timeout_per_test,
        num_workers=64,
    )
    print(f"[grade] done in {time.monotonic() - t0:.1f}s", flush=True)

    # ---- compute custom reward per rollout ----
    rewards: list[dict] = []  # one entry per ROLLOUT (parseable or not)
    n_pass = 0; n_parse_fail = 0; n_gate_fail = 0
    for pid, j, text in rollouts:
        code = extract_code(text)
        if not code:
            rewards.append({"pid": pid, "j": j, "r": 0.0,
                            "reason": "no parseable code"})
            n_parse_fail += 1
            continue
        key = hashlib.md5(f"GEN::{pid}::{j}::{code}".encode()).hexdigest()
        r = grade_results.get(key)
        if r is None or not r.get("all_pass") or not r.get("elapsed_ns"):
            rewards.append({"pid": pid, "j": j, "r": 0.0,
                            "reason": "gate fail"})
            n_gate_fail += 1
            continue
        gen_instr = int(r["elapsed_ns"])
        src_i = src_instr[pid]
        if gen_instr <= 0:
            rewards.append({"pid": pid, "j": j, "r": 0.0,
                            "reason": "zero instr"})
            continue
        rew = max(0.0, math.log(src_i / gen_instr))
        rewards.append({
            "pid": pid, "j": j, "r": rew,
            "src_instr": src_i, "gen_instr": gen_instr,
            "speedup": src_i / gen_instr,
        })
        n_pass += 1
    print(f"[reward] n_pass={n_pass} parse_fail={n_parse_fail} gate_fail={n_gate_fail}",
          flush=True)

    # Save raw
    out_json = out_dir / "rewards.json"
    with open(out_json, "w") as f:
        json.dump({
            "model": args.model, "n_problems": len(picked),
            "n_rollouts_per_problem": args.n_rollouts,
            "src_instr": src_instr, "rewards": rewards,
            "summary": {"n_pass": n_pass, "n_parse_fail": n_parse_fail,
                        "n_gate_fail": n_gate_fail, "n_total": len(rollouts)},
        }, f)
    print(f"[save] {out_json}", flush=True)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    r_all = [x["r"] for x in rewards]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) Full distribution incl. zeros
    ax = axes[0]
    bins = np.arange(0.0, max(max(r_all), 0.1) + 0.1, 0.1)
    ax.hist(r_all, bins=bins, color="#4c72b0", edgecolor="black",
            linewidth=0.3, alpha=0.9)
    ax.set_xlabel("reward  r = max(0, ln(src_instr / gen_instr))")
    ax.set_ylabel(f"frequency  (out of {len(r_all)} rollouts)")
    ax.set_title(f"{args.model}: full reward distribution\n"
                 f"({len(picked)} val problems × {args.n_rollouts} rollouts; "
                 f"perf backend)")
    ax.grid(alpha=0.25, axis="y")
    n_zero = sum(1 for r in r_all if r == 0.0)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1,
               label=f"r=0 (parse/gate fail): {n_zero}")
    ax.legend()

    # (2) Conditional on r > 0 (pass distribution)
    ax = axes[1]
    r_pos = [r for r in r_all if r > 0.0]
    if r_pos:
        bins = np.arange(0.0, max(r_pos) + 0.1, 0.1)
        ax.hist(r_pos, bins=bins, color="#2ca02c", edgecolor="black",
                linewidth=0.3, alpha=0.9)
        ax.set_xlabel("reward (conditional on r > 0)")
        ax.set_ylabel(f"frequency  (n={len(r_pos)} passing)")
        ax.set_title(f"reward distribution among passing rollouts\n"
                     f"median={np.median(r_pos):.3f} p90={np.percentile(r_pos, 90):.3f} "
                     f"max={max(r_pos):.3f}")
        ax.grid(alpha=0.25, axis="y")
    else:
        ax.text(0.5, 0.5, "no passing rollouts", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out_png = out_dir / "reward_distribution.png"
    plt.savefig(out_png, dpi=120)
    print(f"[plot] {out_png}", flush=True)

    print()
    print(f"=== summary ===")
    print(f"  n_rollouts:     {len(r_all)}")
    print(f"  n_pass (r>0):   {n_pass}")
    print(f"  parse_fail:     {n_parse_fail}")
    print(f"  gate_fail:      {n_gate_fail}")
    print(f"  mean_reward:    {np.mean(r_all):.4f}")
    print(f"  mean_reward (positive only): "
          f"{np.mean(r_pos) if r_pos else 0:.4f}")
    print(f"  fraction_positive: {n_pass/len(r_all):.4f}")


if __name__ == "__main__":
    main()
