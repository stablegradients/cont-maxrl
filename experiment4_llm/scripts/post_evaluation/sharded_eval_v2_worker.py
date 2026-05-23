"""
Sharded eval v2 worker: 4 GPUs × TP=1, one benchmark per invocation.
Saves both completions.json and metrics.json to the output directory.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_shard(gpu_id, hf_dir, problems_file, metrics_file, completions_file, args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.pie_mode:
        env["PIE_MODE"] = "1"

    # Optional 1-shot preamble for raw_oneshot template (Base eval B2).
    # Stored as a literal string we inject into the heredoc.
    preamble_literal = repr(args.oneshot_preamble) if args.oneshot_preamble else "None"
    stop_ids_literal = repr(list(args.stop_token_ids)) if args.stop_token_ids else "None"

    # Build the PIE_MODE shim that MUST run before any experiment4_llm import.
    # Without it the worker imports DeepCoder's reward function, which scores
    # every C++ program at 0 (silently). See F6 in DEFERRED_FIXES.md.
    shim = (
        "import os, sys\n"
        "if os.environ.get('PIE_MODE') == '1':\n"
        f"    sys.path.insert(0, {args.repo_root!r})\n"
        "    from experiment11_code_optim.reward import pie_reward_fn\n"
        "    sys.modules['experiment4_llm.src.reward_function'] = pie_reward_fn\n"
    ) if args.pie_mode else ""

    cmd = [
        sys.executable, "-c", f"""
{shim}
import json, sys, math
sys.path.insert(0, "{args.repo_root}")
import pandas as pd
from experiment4_llm.src.eval_checkpoints import generate_completions, score_completions
import experiment4_llm.src.reward_function as _rf

_orig_batch = _rf.compute_scores_batch
def _fast_batch(reward_inputs, timeout_per_test=6, memory_limit_mb=512, num_workers=None, global_timeout=15):
    return _orig_batch(reward_inputs, timeout_per_test, memory_limit_mb, num_workers, global_timeout=15)
_rf.compute_scores_batch = _fast_batch

problems = pd.read_parquet("{problems_file}").to_dict(orient="records")
print(f"    GPU {gpu_id}: generating {{len(problems)}} problems...", flush=True)

completions = generate_completions(
    model_path="{hf_dir}",
    problems=problems,
    n_completions={args.n_completions},
    temperature={args.temperature},
    top_p={args.top_p},
    top_k={args.top_k},
    max_tokens={args.max_tokens},
    gpu_memory_utilization={args.gpu_memory_utilization},
    tensor_parallel_size=1,
    max_num_seqs=256,
    template_mode={args.template_mode!r},
    oneshot_preamble={preamble_literal},
    stop_token_ids={stop_ids_literal},
)

metrics = score_completions(
    completions=completions,
    problems=problems,
    n_target={args.n_completions},
    k_values={args.k_values},
    timeout_per_test={args.timeout_per_test},
    n_workers={args.n_workers},
)

with open("{metrics_file}", "w") as f:
    json.dump(metrics, f, indent=2)

with open("{completions_file}", "w") as f:
    json.dump(completions, f)

print(f"    GPU {gpu_id}: done", flush=True)
"""
    ]
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


def merge_shard_metrics(shard_files, k_values):
    all_per_problem = []
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        all_per_problem.extend(data["per_problem"])
    for i, pp in enumerate(all_per_problem):
        pp["problem_idx"] = i

    avg_pass_at_k = {}
    for k in k_values:
        values = [p["pass_at_k"].get(str(k)) for p in all_per_problem
                  if p["pass_at_k"].get(str(k)) is not None]
        avg_pass_at_k[str(k)] = sum(values) / len(values) if values else None

    all_scores = [p["mean_score"] for p in all_per_problem]
    all_unique = [p["unique_solution_rate"] for p in all_per_problem]
    all_lengths = [p["mean_output_length"] for p in all_per_problem]
    total_c = sum(p["c"] for p in all_per_problem)
    total_n = sum(p["n"] for p in all_per_problem)

    return {
        "pass_at_k": avg_pass_at_k,
        "per_problem": all_per_problem,
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "mean_pass_rate": total_c / total_n if total_n else 0,
        "mean_unique_solution_rate": sum(all_unique) / len(all_unique) if all_unique else 0,
        "mean_output_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
        "n_problems_solved": sum(1 for p in all_per_problem if p["c"] > 0),
        "n_problems": len(all_per_problem),
        "n_completions_per_problem": all_per_problem[0]["n"] if all_per_problem else 0,
    }


def merge_shard_completions(shard_files):
    all_completions = []
    for sf in shard_files:
        with open(sf) as f:
            all_completions.extend(json.load(f))
    return all_completions


def shard_parquet(eval_file, n_shards, tmpdir):
    df = pd.read_parquet(eval_file)
    shard_files = []
    shard_size = math.ceil(len(df) / n_shards)
    for i in range(n_shards):
        shard = df.iloc[i * shard_size : (i + 1) * shard_size]
        if len(shard) == 0:
            continue
        path = os.path.join(tmpdir, f"shard_{i}.parquet")
        shard.to_parquet(path)
        shard_files.append(path)
    return shard_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--hf_dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_completions", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--timeout_per_test", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=64)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    # PIE baseline-eval additions:
    parser.add_argument("--pie_mode", action="store_true",
                        help="Install sys.modules shim so experiment4_llm.src.reward_function "
                             "→ pie_reward_fn (C++ compile+gate scoring instead of DeepCoder).")
    parser.add_argument("--template_mode", choices=["chat", "raw", "raw_oneshot"],
                        default="chat",
                        help="chat: apply_chat_template with enable_thinking=False (default). "
                             "raw: use prompt user-content as-is (Base, no anchor). "
                             "raw_oneshot: prepend --oneshot_preamble (Base + 1-shot anchor).")
    parser.add_argument("--oneshot_preamble", type=str, default=None,
                        help="Required when --template_mode=raw_oneshot. Path to a text file "
                             "whose contents are prepended to each user-content prompt.")
    parser.add_argument("--stop_token_ids", type=int, nargs="*", default=None,
                        help="EOS list to pass to vLLM SamplingParams.stop_token_ids. "
                             "Base: 151643. Instruct: 151645 151643.")
    args = parser.parse_args()
    # Resolve oneshot file → literal string before passing to subprocess
    if args.oneshot_preamble:
        with open(args.oneshot_preamble) as f:
            args.oneshot_preamble = f.read()
    args.repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)

    output_dir = args.output_dir
    metrics_path = os.path.join(output_dir, "metrics.json")
    completions_path = os.path.join(output_dir, "completions.json")

    if os.path.isfile(metrics_path):
        print(f"  Already done: {metrics_path}. Skipping.")
        return

    print(f"  === {args.benchmark} (step {args.step}) ===")

    with tempfile.TemporaryDirectory(prefix=f"shard_v2_{args.benchmark}_") as tmpdir:
        shard_parquets = shard_parquet(args.data_file, args.n_gpus, tmpdir)
        n_actual_shards = len(shard_parquets)
        print(f"  Split into {n_actual_shards} shards")

        shard_metrics_files = []
        shard_completions_files = []
        procs = []
        for i, sp in enumerate(shard_parquets):
            mf = os.path.join(tmpdir, f"metrics_shard_{i}.json")
            cf = os.path.join(tmpdir, f"completions_shard_{i}.json")
            shard_metrics_files.append(mf)
            shard_completions_files.append(cf)
            p = run_shard(i, args.hf_dir, sp, mf, cf, args)
            procs.append(p)

        failed = 0
        for i, p in enumerate(procs):
            rc = p.wait()
            if rc != 0:
                print(f"  WARNING: GPU {i} exited with code {rc}")
                failed += 1

        if failed > 0:
            print(f"  {failed} shard(s) failed")
            sys.exit(1)

        missing_m = [f for f in shard_metrics_files if not os.path.isfile(f)]
        missing_c = [f for f in shard_completions_files if not os.path.isfile(f)]
        if missing_m or missing_c:
            print(f"  Missing shard outputs: metrics={missing_m}, completions={missing_c}")
            sys.exit(1)

        merged_metrics = merge_shard_metrics(shard_metrics_files, args.k_values)
        merged_metrics["global_step"] = args.step
        merged_completions = merge_shard_completions(shard_completions_files)

        os.makedirs(output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(merged_metrics, f, indent=2)
        with open(completions_path, "w") as f:
            json.dump(merged_completions, f)

        pk = merged_metrics["pass_at_k"]
        pk_strs = [f"pass@{k}={pk[str(k)]:.4f}" for k in args.k_values if pk.get(str(k)) is not None]
        print(f"  {args.benchmark}: {' '.join(pk_strs)}")
        print(f"  Saved to {output_dir}")


if __name__ == "__main__":
    main()
