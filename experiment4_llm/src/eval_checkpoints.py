"""
Re-evaluate HE+ / MBPP+ on saved FSDP checkpoints.

1. Merge FSDP shards → HF format (skips if already done)
2. Generate N completions per problem with vLLM
3. Score with reward function, compute pass@k
4. Save metrics JSON (same schema as insitu_eval)
5. Generate pass@k plots across training steps

Usage:
    python -m experiment4_llm.src.eval_checkpoints \
        --results_dir results/qwen3_1.7b_base_deepcoder_grpo_g16_s42 \
        --eval_files data/evalplus_humaneval_test.parquet data/evalplus_mbpp_test.parquet \
        --eval_names humaneval_plus mbpp_plus \
        --base_model Qwen/Qwen3-1.7B-Base \
        --steps 83 166 249 332 415 498 \
        --n_completions 200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from experiment4_llm.src.reward_function import compute_scores_batch, extract_code


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def merge_fsdp_checkpoint(
    actor_dir: str,
    hf_dir: str,
    base_model: str,
    merger_script: str,
) -> None:
    """Merge FSDP shards to HF format."""
    if os.path.isdir(hf_dir) and os.path.isfile(os.path.join(hf_dir, "config.json")):
        print(f"    HF checkpoint exists at {hf_dir}, skipping merge.")
        return
    print(f"    Merging FSDP → HF: {actor_dir} → {hf_dir}")
    subprocess.run(
        [sys.executable, merger_script,
         "--backend", "fsdp",
         "--hf_model_path", base_model,
         "--local_dir", actor_dir,
         "--target_dir", hf_dir],
        check=True,
    )


def generate_completions(
    model_path: str,
    problems: list[dict],
    n_completions: int = 200,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = -1,
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 4,
    max_num_seqs: int = 256,
) -> list[list[str]]:
    """Generate completions using vLLM with chat template."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        dtype="bfloat16",
        enforce_eager=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=32768,
    )
    sampling_params = SamplingParams(
        n=n_completions,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    # Format prompts via chat template (matches training rollout)
    formatted_prompts = []
    for problem in problems:
        messages = list(problem["prompt"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted_prompts.append(text)

    print(f"    Generating {n_completions} completions for {len(formatted_prompts)} problems...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    all_completions = [[o.text for o in output.outputs] for output in outputs]

    # Explicitly delete vLLM to free GPU memory before next checkpoint
    del llm
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    return all_completions


def score_completions(
    completions: list[list[str]],
    problems: list[dict],
    n_target: int = 200,
    k_values: list[int] | None = None,
    timeout_per_test: int = 6,
    n_workers: int = 128,
) -> dict:
    """Score completions and compute pass@k (same schema as insitu_eval)."""
    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50, 100, 200]

    n_problems = len(completions)

    # Build flat list of reward inputs
    reward_inputs = []
    problem_indices = []
    for i in range(n_problems):
        gt = problems[i]["reward_model"]["ground_truth"]
        if isinstance(gt, str):
            pass  # already JSON string
        else:
            gt = json.dumps(gt)
        for code in completions[i][:n_target]:
            reward_inputs.append({"solution_str": code, "ground_truth": gt})
            problem_indices.append(i)

    print(f"    Scoring {len(reward_inputs)} completions with {n_workers} workers...")
    scores = compute_scores_batch(
        reward_inputs, timeout_per_test=timeout_per_test, num_workers=n_workers,
    )

    # Aggregate per problem
    problem_pass_counts: dict[int, int] = defaultdict(int)
    problem_total_counts: dict[int, int] = defaultdict(int)
    problem_scores_list: dict[int, list[float]] = defaultdict(list)
    for prob_idx, score_dict in zip(problem_indices, scores):
        problem_total_counts[prob_idx] += 1
        problem_scores_list[prob_idx].append(score_dict["score"])
        if score_dict["acc"] == 1.0:
            problem_pass_counts[prob_idx] += 1

    # Unique solution rate and output length
    problem_unique_rates: dict[int, float] = {}
    problem_mean_lengths: dict[int, float] = {}
    for i in range(n_problems):
        codes = [extract_code(c) for c in completions[i][:n_target]]
        n_actual = max(len(codes), 1)
        problem_unique_rates[i] = len(set(codes)) / n_actual
        problem_mean_lengths[i] = sum(len(c) for c in completions[i][:n_target]) / n_actual

    per_problem_results = []
    for i in range(n_problems):
        n = problem_total_counts[i]
        c = problem_pass_counts[i]
        ppk = {}
        for k in k_values:
            ppk[str(k)] = pass_at_k(n, c, k) if k <= n else None
        per_problem_results.append({
            "problem_idx": i,
            "n": n,
            "c": c,
            "pass_at_k": ppk,
            "mean_score": sum(problem_scores_list[i]) / max(len(problem_scores_list[i]), 1),
            "unique_solution_rate": problem_unique_rates[i],
            "mean_output_length": problem_mean_lengths[i],
        })

    # Macro-average pass@k
    avg_pass_at_k = {}
    for k in k_values:
        values = [r["pass_at_k"][str(k)] for r in per_problem_results if r["pass_at_k"][str(k)] is not None]
        avg_pass_at_k[str(k)] = sum(values) / len(values) if values else None

    # Summary
    total_passing = sum(r["c"] for r in per_problem_results)
    total_completions = sum(r["n"] for r in per_problem_results)
    all_scores = [s for sl in problem_scores_list.values() for s in sl]
    n_problems_solved = sum(1 for r in per_problem_results if r["c"] > 0)

    all_codes = []
    for i in range(n_problems):
        all_codes.extend(extract_code(c) for c in completions[i][:n_target])
    unique_code_rate_global = len(set(all_codes)) / max(len(all_codes), 1)

    return {
        "pass_at_k": avg_pass_at_k,
        "per_problem": per_problem_results,
        "summary": {
            "total_problems": n_problems,
            "total_completions": total_completions,
            "total_passing": total_passing,
            "n_problems_solved": n_problems_solved,
            "mean_pass_rate": total_passing / max(total_completions, 1),
            "mean_score": sum(all_scores) / max(len(all_scores), 1),
            "mean_unique_solution_rate": sum(problem_unique_rates.values()) / max(len(problem_unique_rates), 1),
            "mean_output_length": sum(problem_mean_lengths.values()) / max(len(problem_mean_lengths), 1),
            "unique_code_rate_global": unique_code_rate_global,
        },
    }


def generate_plots(results_dirs: list[str], run_names: list[str], output_path: str):
    """Generate pass@k plots across training steps for multiple runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect data: {eval_name: {run_name: {step: {k: value}}}}
    all_data: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for results_dir, run_name in zip(results_dirs, run_names):
        # Find all eval_* directories that have reeval_ metrics
        for entry in sorted(os.listdir(results_dir)):
            if not entry.startswith("reeval_"):
                continue
            eval_name = entry.replace("reeval_", "")
            eval_dir = os.path.join(results_dir, entry)
            if not os.path.isdir(eval_dir):
                continue
            for fname in sorted(os.listdir(eval_dir)):
                if not fname.startswith("metrics_step") or not fname.endswith(".json"):
                    continue
                step = int(fname.replace("metrics_step", "").replace(".json", ""))
                with open(os.path.join(eval_dir, fname)) as f:
                    metrics = json.load(f)
                all_data[eval_name][run_name][step] = metrics["pass_at_k"]

    if not all_data:
        print("  No reeval metrics found, skipping plots.")
        return

    # Plot config — auto-detect k values from the first available metrics file
    k_values_to_plot = None
    for eval_name_check in sorted(all_data.keys()):
        for run_data_check in all_data[eval_name_check].values():
            for step_data_check in run_data_check.values():
                k_values_to_plot = sorted(step_data_check.keys(), key=lambda x: int(x))
                break
            if k_values_to_plot:
                break
        if k_values_to_plot:
            break
    if not k_values_to_plot:
        k_values_to_plot = ["1", "4", "16", "64", "256"]
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    colors = {k: palette[i % len(palette)] for i, k in enumerate(k_values_to_plot)}
    run_styles = ["-", "--", "-.", ":"]

    eval_names = sorted(all_data.keys())
    n_evals = len(eval_names)

    fig, axes = plt.subplots(1, n_evals, figsize=(7 * n_evals, 5), squeeze=False)
    fig.suptitle("pass@k Over Training Steps (Re-evaluated with Fixed Prompts)", fontsize=14, y=1.02)

    for col, eval_name in enumerate(eval_names):
        ax = axes[0, col]
        run_data = all_data[eval_name]

        for run_idx, (run_name, step_data) in enumerate(sorted(run_data.items())):
            steps = sorted(step_data.keys())
            style = run_styles[run_idx % len(run_styles)]

            for k in k_values_to_plot:
                values = []
                valid_steps = []
                for s in steps:
                    v = step_data[s].get(k)
                    if v is not None:
                        values.append(v * 100)
                        valid_steps.append(s)
                if values:
                    label = f"{run_name} pass@{k}" if n_evals == 1 or run_idx == 0 else f"{run_name} pass@{k}"
                    ax.plot(valid_steps, values, style, color=colors.get(k, "gray"),
                            label=label, linewidth=1.5, marker="o", markersize=3)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("pass@k (%)")
        ax.set_title(eval_name.replace("_", " ").title())
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {output_path}")
    plt.close(fig)

    # Also generate per-benchmark individual plots with more detail
    for eval_name in eval_names:
        run_data = all_data[eval_name]
        fig, ax = plt.subplots(figsize=(10, 6))

        for run_idx, (run_name, step_data) in enumerate(sorted(run_data.items())):
            steps = sorted(step_data.keys())
            style = run_styles[run_idx % len(run_styles)]

            for k in k_values_to_plot:
                values = []
                valid_steps = []
                for s in steps:
                    v = step_data[s].get(k)
                    if v is not None:
                        values.append(v * 100)
                        valid_steps.append(s)
                if values:
                    ax.plot(valid_steps, values, style, color=colors.get(k, "gray"),
                            label=f"{run_name} pass@{k}", linewidth=2, marker="o", markersize=4)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("pass@k (%)", fontsize=12)
        ax.set_title(f"{eval_name.replace('_', ' ').title()} — pass@k Over Training", fontsize=13)
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        per_bench_path = output_path.replace(".png", f"_{eval_name}.png")
        fig.savefig(per_bench_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {per_bench_path}")
        plt.close(fig)


def _upload_to_wandb(
    results_dirs: list[str],
    wandb_run_ids_list: list[str],
    wandb_project: str,
    wandb_entity: str,
    eval_names: list[str],
) -> None:
    """Upload reeval metrics to existing W&B runs.

    wandb_run_ids_list: one entry per results_dir. Each entry can contain
    multiple run IDs separated by commas (for chained runs). Steps are routed
    to the correct run by querying the step range from W&B history.
    """
    import wandb

    for results_dir, run_ids_str in zip(results_dirs, wandb_run_ids_list):
        run_ids = [r.strip() for r in run_ids_str.split(",")]

        # Collect all reeval metrics keyed by step
        step_metrics: dict[int, dict[str, float]] = defaultdict(dict)
        for eval_name in eval_names:
            reeval_dir = os.path.join(results_dir, f"reeval_{eval_name}")
            if not os.path.isdir(reeval_dir):
                continue
            for fname in sorted(os.listdir(reeval_dir)):
                if not fname.startswith("metrics_step") or not fname.endswith(".json"):
                    continue
                step = int(fname.replace("metrics_step", "").replace(".json", ""))
                with open(os.path.join(reeval_dir, fname)) as f:
                    metrics = json.load(f)
                prefix = f"insitu_eval/{eval_name}"
                pk = metrics["pass_at_k"]
                for k, v in pk.items():
                    if v is not None:
                        step_metrics[step][f"{prefix}/pass_at_{k}"] = v
                summary = metrics.get("summary", {})
                for key in ["mean_score", "mean_pass_rate", "mean_unique_solution_rate",
                            "mean_output_length", "n_problems_solved", "unique_code_rate_global"]:
                    if key in summary:
                        step_metrics[step][f"{prefix}/{key}"] = summary[key]

        if not step_metrics:
            print(f"    No reeval metrics found for {results_dir}, skipping.")
            continue

        # Query step ranges for each run ID
        api = wandb.Api()
        run_ranges: list[tuple[str, int, int]] = []
        for rid in run_ids:
            try:
                r = api.run(f"{wandb_entity}/{wandb_project}/{rid}")
                hist = r.history(keys=["training/global_step"], samples=10000)
                if len(hist) > 0 and "training/global_step" in hist.columns:
                    min_s = int(hist["training/global_step"].min())
                    max_s = int(hist["training/global_step"].max())
                    run_ranges.append((rid, min_s, max_s))
                    print(f"    Run {rid}: steps {min_s}-{max_s}")
            except Exception as e:
                print(f"    WARNING: could not query run {rid}: {e}")

        if not run_ranges:
            print(f"    No valid W&B runs found, skipping upload.")
            continue

        # Group steps by which run they belong to
        run_step_metrics: dict[str, dict[int, dict]] = defaultdict(dict)
        for step, mets in sorted(step_metrics.items()):
            best_run = None
            for rid, min_s, max_s in run_ranges:
                if min_s <= step <= max_s:
                    best_run = rid
                    break
            if best_run is None:
                # Fall back to last run (for steps beyond all ranges)
                best_run = run_ranges[-1][0]
            run_step_metrics[best_run][step] = mets

        # Log to each wandb run
        for rid, steps_mets in run_step_metrics.items():
            print(f"\n  Uploading to W&B run {rid}...")
            run = wandb.init(
                id=rid,
                project=wandb_project,
                entity=wandb_entity,
                resume="allow",
            )
            for step in sorted(steps_mets.keys()):
                run.log(steps_mets[step], step=step)
                print(f"    Logged {len(steps_mets[step])} metrics at step {step}")
            run.finish()
            print(f"    Done uploading to W&B run {rid}")


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints on HE+/MBPP+")
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="Results directories containing global_step_* checkpoints")
    parser.add_argument("--run_names", nargs="+", default=None,
                        help="Display names for each run (defaults to dir basenames)")
    parser.add_argument("--eval_files", nargs="+", required=True,
                        help="Parquet eval files (e.g. evalplus_humaneval_test.parquet)")
    parser.add_argument("--eval_names", nargs="+", required=True,
                        help="Names for each eval file (e.g. humaneval_plus mbpp_plus)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Specific steps to evaluate (default: all available)")
    parser.add_argument("--n_completions", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--timeout_per_test", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=128)
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[1, 2, 5, 10, 20, 50, 100, 200])
    parser.add_argument("--plot_output", type=str, default=None,
                        help="Path for plot output (default: <first_results_dir>/reeval_plots/pass_at_k.png)")
    parser.add_argument("--wandb_run_ids", nargs="+", default=None,
                        help="W&B run IDs to update with reeval metrics (one per results_dir)")
    parser.add_argument("--wandb_project", type=str, default="cont-maxrl-code",
                        help="W&B project name (default: cont-maxrl-code)")
    parser.add_argument("--wandb_entity", type=str, default="stablegradients",
                        help="W&B entity (default: stablegradients)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip eval, only regenerate plots from existing metrics")
    parser.add_argument("--_direct", action="store_true",
                        help=argparse.SUPPRESS)  # internal: run inline, no subprocess
    args = parser.parse_args()

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    merger_script = os.path.join(repo_root, "vendor/verl/scripts/model_merger.py")

    if args.run_names is None:
        args.run_names = [os.path.basename(d.rstrip("/")) for d in args.results_dirs]

    assert len(args.eval_files) == len(args.eval_names), "eval_files and eval_names must match"
    assert len(args.results_dirs) == len(args.run_names), "results_dirs and run_names must match"

    if not args.plot_only:
        # Load eval datasets once
        eval_datasets = {}
        for eval_file, eval_name in zip(args.eval_files, args.eval_names):
            df = pd.read_parquet(eval_file)
            problems = df.to_dict(orient="records")
            eval_datasets[eval_name] = problems
            print(f"Loaded {eval_name}: {len(problems)} problems from {eval_file}")

        # Process each results directory
        for results_dir, run_name in zip(args.results_dirs, args.run_names):
            print(f"\n{'='*60}")
            print(f"Run: {run_name} ({results_dir})")
            print(f"{'='*60}")

            # Find checkpoints
            ckpt_dirs = sorted(
                [d for d in Path(results_dir).glob("global_step_*/actor") if d.is_dir()],
                key=lambda p: int(p.parent.name.split("_")[-1]),
            )
            available_steps = [int(d.parent.name.split("_")[-1]) for d in ckpt_dirs]

            if args.steps:
                selected_steps = [s for s in args.steps if s in available_steps]
                missing = [s for s in args.steps if s not in available_steps]
                if missing:
                    print(f"  WARNING: steps not found: {missing}")
                    print(f"  Available: {available_steps}")
            else:
                selected_steps = available_steps

            print(f"  Evaluating {len(selected_steps)} checkpoints: {selected_steps}")

            for step in selected_steps:
                actor_dir = os.path.join(results_dir, f"global_step_{step}", "actor")
                hf_dir = os.path.join(results_dir, f"global_step_{step}", "actor_hf")

                print(f"\n  --- Step {step} ---")

                # Check if all evals already done for this step
                all_done = all(
                    os.path.isfile(os.path.join(results_dir, f"reeval_{eval_name}", f"metrics_step{step}.json"))
                    for eval_name in args.eval_names
                )
                if all_done:
                    print(f"    All evals already done, skipping.")
                    continue

                # Merge FSDP → HF
                merge_fsdp_checkpoint(actor_dir, hf_dir, args.base_model, merger_script)

                # Generate completions and score for each eval dataset.
                # When TP>1, vLLM's torch.distributed state leaks between
                # engine reinitializations, causing timeouts on the 2nd+ init.
                # Fix: spawn each (step, eval) as a fresh subprocess.
                for eval_name, problems in eval_datasets.items():
                    out_dir = os.path.join(results_dir, f"reeval_{eval_name}")
                    metrics_path = os.path.join(out_dir, f"metrics_step{step}.json")

                    if os.path.isfile(metrics_path):
                        print(f"    {eval_name}: already done, skipping.")
                        continue

                    if args._direct:
                        # Direct mode: run inline (called from subprocess)
                        print(f"    {eval_name}: generating completions...")
                        completions = generate_completions(
                            model_path=hf_dir,
                            problems=problems,
                            n_completions=args.n_completions,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            max_tokens=args.max_tokens,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            tensor_parallel_size=args.tensor_parallel_size,
                            max_num_seqs=256,
                        )

                        metrics = score_completions(
                            completions=completions,
                            problems=problems,
                            n_target=args.n_completions,
                            k_values=args.k_values,
                            timeout_per_test=args.timeout_per_test,
                            n_workers=args.n_workers,
                        )
                        metrics["global_step"] = step

                        os.makedirs(out_dir, exist_ok=True)
                        with open(metrics_path, "w") as f:
                            json.dump(metrics, f, indent=2)
                        print(f"    {eval_name}: saved to {metrics_path}")

                        pk = metrics["pass_at_k"]
                        pk_strs = [f"pass@{k}={pk.get(str(k), 'N/A'):.4f}" for k in args.k_values if pk.get(str(k)) is not None]
                        print(f"    {' '.join(pk_strs)}")
                    else:
                        # Subprocess mode: fresh process avoids vLLM reinit crash
                        eval_file = args.eval_files[args.eval_names.index(eval_name)]
                        print(f"    {eval_name}: launching subprocess for generation + scoring...")

                        sub_cmd = [
                            sys.executable, "-m", "experiment4_llm.src.eval_checkpoints",
                            "--_direct",
                            "--results_dirs", results_dir,
                            "--run_names", run_name,
                            "--eval_files", eval_file,
                            "--eval_names", eval_name,
                            "--base_model", args.base_model,
                            "--steps", str(step),
                            "--n_completions", str(args.n_completions),
                            "--temperature", str(args.temperature),
                            "--top_p", str(args.top_p),
                            "--top_k", str(args.top_k),
                            "--max_tokens", str(args.max_tokens),
                            "--tensor_parallel_size", str(args.tensor_parallel_size),
                            "--gpu_memory_utilization", str(args.gpu_memory_utilization),
                            "--timeout_per_test", str(args.timeout_per_test),
                            "--n_workers", str(args.n_workers),
                            "--k_values", *[str(k) for k in args.k_values],
                        ]
                        result = subprocess.run(sub_cmd, env=os.environ)
                        if result.returncode != 0:
                            print(f"    WARNING: subprocess for {eval_name} step {step} exited with code {result.returncode}")
                        elif os.path.isfile(metrics_path):
                            with open(metrics_path) as f:
                                metrics = json.load(f)
                            pk = metrics["pass_at_k"]
                            pk_strs = [f"pass@{k}={pk[str(k)]:.4f}" for k in args.k_values if pk.get(str(k)) is not None]
                            print(f"    {' '.join(pk_strs)}")

    # Generate plots
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")

    plot_output = args.plot_output
    if plot_output is None:
        plot_output = os.path.join(args.results_dirs[0], "..", "reeval_plots", "pass_at_k.png")
        plot_output = os.path.normpath(plot_output)

    generate_plots(args.results_dirs, args.run_names, plot_output)

    # Upload reeval metrics to W&B runs
    if args.wandb_run_ids and not args._direct:
        assert len(args.wandb_run_ids) == len(args.results_dirs), \
            "wandb_run_ids must match results_dirs (one per run)"
        _upload_to_wandb(
            results_dirs=args.results_dirs,
            wandb_run_ids=args.wandb_run_ids,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            eval_names=args.eval_names,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
