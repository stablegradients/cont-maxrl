"""Custom reward probe — but the model is NOT shown the src code.
Instead it sees 2 I/O example pairs from the problem's test cases and is
asked to write a C++ program from scratch.

Reward formula identical: r = max(0, ln(src_instr / gen_instr)) if pass else 0.
The src_instr used for reward is still the original slow src (so we
measure how much faster than src the from-scratch generation is — same
yardstick as the src-given variant for direct comparison).
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PIE_BACKEND", "perf")
os.environ.setdefault("PIE_MEASUREMENT_MAX_TESTS", "5")
os.environ.setdefault("PIE_COMPILE_SCRATCH",
                       f"/tmp/{os.environ.get('USER', 'u')}_pie_scratch_nosrc")
os.makedirs(os.environ["PIE_COMPILE_SCRATCH"], exist_ok=True)

REPO = "/projects/bgmo/sramasubramanian/cont-maxrl"
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_problems", type=int, default=8)
    ap.add_argument("--n_rollouts", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opt_level", choices=["O0", "O3"], default="O0")
    ap.add_argument("--timeout_per_test", type=float, default=30.0)
    ap.add_argument("--n_examples", type=int, default=2,
                    help="How many I/O example pairs to show in the prompt.")
    ap.add_argument("--output_dir", default=f"{REPO}/experiment11_code_optim/docs/plots/custom_reward_nosrc")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import experiment11_code_optim.measurement.core as core
    core.COMPILE_FLAGS = [f"-{args.opt_level}", "-std=c++17"]
    print(f"[setup] COMPILE_FLAGS={core.COMPILE_FLAGS}", flush=True)
    from experiment11_code_optim.measurement.core import setup_pch
    os.environ["PIE_PCH_INCLUDE_PATH"] = setup_pch()
    print("[setup] PCH built", flush=True)

    from experiment11_code_optim.reward.pie_reward_fn import (
        grade_dedup_batch, preload_test_cases, extract_code,
    )
    TEST_CASE_DIR = "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
    preload_test_cases(TEST_CASE_DIR)

    val_jsonl = "/work/nvme/bgmo/sramasubramanian/pie_data/raw/val.jsonl"
    picked: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("problem_id"); src = r.get("src_code")
            if not pid or not src or pid in seen:
                continue
            seen.add(pid)
            picked.append((pid, src))
            if len(picked) >= args.n_problems:
                break
    print(f"[setup] picked pids: {[p[0] for p in picked]}", flush=True)

    # Load test cases per problem
    pid_examples: dict[str, list[tuple[str, str]]] = {}
    for pid, _src in picked:
        d = Path(TEST_CASE_DIR) / pid
        cases: list[tuple[str, str]] = []
        for i in range(args.n_examples):
            ip = d / f"input.{i}.txt"
            op = d / f"output.{i}.txt"
            if ip.exists() and op.exists():
                cases.append((ip.read_text(), op.read_text()))
        pid_examples[pid] = cases
        if not cases:
            print(f"[warn] pid={pid} has no test cases at {d}", flush=True)
    picked = [(p, s) for p, s in picked if pid_examples.get(p)]
    print(f"[setup] {len(picked)} problems with examples", flush=True)

    # Grade src under same -O0 backend (same as reference yardstick)
    src_tasks = {}; src_meta = {}
    for pid, src in picked:
        key = hashlib.md5(f"SRC::{pid}::{src}".encode()).hexdigest()
        src_tasks[key] = {"code": src, "problem_id": pid, "split": "val"}
        src_meta[key] = pid
    src_results = grade_dedup_batch(
        src_tasks, test_case_dir=TEST_CASE_DIR,
        measurement_max_tests=5, timeout_per_test=args.timeout_per_test,
        num_workers=8,
    )
    src_instr: dict[str, int] = {}
    for k, r in src_results.items():
        if r.get("all_pass") and r.get("elapsed_ns"):
            src_instr[src_meta[k]] = int(r["elapsed_ns"])
    print(f"[src] graded {len(src_instr)}/{len(picked)}", flush=True)

    # vLLM
    print(f"[gen] loading {args.model}...", flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model, tokenizer=args.model, trust_remote_code=True,
        gpu_memory_utilization=0.85, tensor_parallel_size=1,
        max_num_seqs=256, dtype="bfloat16", enforce_eager=False,
        enable_chunked_prefill=True, max_num_batched_tokens=32768,
        seed=args.seed,
    )

    def build_prompt_user(pid: str) -> str:
        ex_blocks = []
        for i, (inp, out) in enumerate(pid_examples[pid]):
            ex_blocks.append(f"### Example {i+1}\n\n"
                              f"Input:\n```\n{inp.rstrip()}\n```\n\n"
                              f"Output:\n```\n{out.rstrip()}\n```")
        examples = "\n\n".join(ex_blocks)
        return (
            "You are an expert C++ competitive programmer. Read the input "
            "from standard input and write the output to standard output. "
            "Infer the input/output format and the function to compute from "
            "the examples below. Write an efficient C++ program.\n\n"
            f"{examples}\n\n"
            "### C++ Solution\n"
        )

    formatted_prompts = []
    for pid, _ in picked:
        user_content = build_prompt_user(pid)
        text = tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        formatted_prompts.append(text)

    print(f"[gen] generating {args.n_rollouts} × {len(formatted_prompts)} ...", flush=True)
    sp = SamplingParams(
        n=args.n_rollouts, temperature=args.temperature, top_p=1.0,
        top_k=-1, max_tokens=args.max_tokens,
        stop_token_ids=[151645, 151643],
    )
    t0 = time.monotonic()
    outputs = llm.generate(formatted_prompts, sp)
    print(f"[gen] done in {time.monotonic() - t0:.1f}s", flush=True)
    rollouts = []
    for i, out in enumerate(outputs):
        pid = picked[i][0]
        for j, s in enumerate(out.outputs):
            rollouts.append((pid, str(j), s.text))
    print(f"[gen] {len(rollouts)} completions", flush=True)

    del llm
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    grade_tasks = {}; grade_meta = {}
    for pid, j, text in rollouts:
        code = extract_code(text)
        if not code: continue
        key = hashlib.md5(f"GEN::{pid}::{j}::{code}".encode()).hexdigest()
        grade_tasks[key] = {"code": code, "problem_id": pid, "split": "val"}
        grade_meta[key] = (pid, j)
    print(f"[grade] {len(grade_tasks)} parseable / {len(rollouts)}", flush=True)
    grade_results = grade_dedup_batch(
        grade_tasks, test_case_dir=TEST_CASE_DIR,
        measurement_max_tests=5, timeout_per_test=args.timeout_per_test,
        num_workers=64,
    )

    rewards = []
    n_pass = n_pf = n_gf = 0
    for pid, j, text in rollouts:
        code = extract_code(text)
        if not code:
            rewards.append({"pid": pid, "j": j, "r": 0.0, "reason": "no code"}); n_pf += 1; continue
        key = hashlib.md5(f"GEN::{pid}::{j}::{code}".encode()).hexdigest()
        r = grade_results.get(key)
        if r is None or not r.get("all_pass") or not r.get("elapsed_ns"):
            rewards.append({"pid": pid, "j": j, "r": 0.0, "reason": "gate fail"}); n_gf += 1; continue
        gi = int(r["elapsed_ns"]); si = src_instr.get(pid)
        if not si or gi <= 0:
            rewards.append({"pid": pid, "j": j, "r": 0.0, "reason": "no baseline"}); continue
        rew = max(0.0, math.log(si / gi))
        rewards.append({"pid": pid, "j": j, "r": rew,
                        "src_instr": si, "gen_instr": gi, "speedup": si / gi})
        n_pass += 1
    print(f"[reward] n_pass={n_pass} parse_fail={n_pf} gate_fail={n_gf}", flush=True)

    with open(out_dir / "rewards.json", "w") as f:
        json.dump({"model": args.model, "opt_level": args.opt_level,
                   "prompt_mode": "no_src_io_examples",
                   "n_problems": len(picked), "n_rollouts_per_problem": args.n_rollouts,
                   "src_instr": src_instr, "rewards": rewards,
                   "summary": {"n_pass": n_pass, "n_parse_fail": n_pf,
                               "n_gate_fail": n_gf, "n_total": len(rollouts)}}, f)
    print(f"[save] {out_dir / 'rewards.json'}")

    # Quick plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    r_all = [x["r"] for x in rewards]; r_pos = [r for r in r_all if r > 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(0.0, max(max(r_all), 1.0) + 0.05, 0.05)
    axes[0].hist(r_all, bins=bins, color="#4c72b0", edgecolor="black",
                 linewidth=0.3, alpha=0.9)
    axes[0].set_yscale("log")
    axes[0].set_title(f"NO-SRC prompt — full distribution  (n={len(r_all)})")
    axes[0].set_xlabel("reward"); axes[0].set_ylabel("freq (log)")
    if r_pos:
        bins = np.arange(0.0, max(r_pos)+0.05, 0.05)
        axes[1].hist(r_pos, bins=bins, color="#dd8452", edgecolor="black",
                     linewidth=0.3, alpha=0.9)
        axes[1].set_title(f"positive only  (n={len(r_pos)})\n"
                          f"med={np.median(r_pos):.3f} max={max(r_pos):.3f}")
    fig.suptitle(f"{args.model}  prompt=no_src (I/O examples only)  -{args.opt_level}",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_distribution.png", dpi=120)
    print(f"[plot] {out_dir / 'reward_distribution.png'}")


if __name__ == "__main__":
    main()
