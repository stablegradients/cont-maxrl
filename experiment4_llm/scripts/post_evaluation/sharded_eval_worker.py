"""
Sharded eval worker: runs one benchmark at a time, 4 GPUs in parallel (TP=1 each).
Each GPU handles 1/4 of the problems. Results merged and saved as single metrics file.
Benchmarks run sequentially so partial progress is saved on timeout.
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
    """Unbiased estimator for pass@k."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def run_shard(gpu_id, hf_dir, problems_file, out_file, args):
    """Run eval on a shard of problems on a specific GPU. Returns subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-c", f"""
import json, sys
sys.path.insert(0, "{os.path.join(args.repo_root)}")
import pandas as pd
from experiment4_llm.src.eval_checkpoints import generate_completions, score_completions
# Monkey-patch global_timeout from 60s to 15s to avoid pathological completions blocking scoring
import experiment4_llm.src.reward_function as _rf
_orig_batch = _rf.compute_scores_batch
def _fast_batch(reward_inputs, timeout_per_test=6, memory_limit_mb=512, num_workers=None, global_timeout=15):
    return _orig_batch(reward_inputs, timeout_per_test, memory_limit_mb, num_workers, global_timeout=15)
_rf.compute_scores_batch = _fast_batch

problems = pd.read_parquet("{problems_file}").to_dict(orient="records")
print(f"  GPU {gpu_id}: generating {{len(problems)}} problems...", flush=True)

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
)

metrics = score_completions(
    completions=completions,
    problems=problems,
    n_target={args.n_completions},
    k_values={args.k_values},
    timeout_per_test={args.timeout_per_test},
    n_workers={args.n_workers},
)

with open("{out_file}", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  GPU {gpu_id}: done, saved to {out_file}", flush=True)
"""
    ]
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


def merge_shard_metrics(shard_files, k_values):
    """Merge per-shard metrics into a single metrics dict."""
    all_per_problem = []
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        all_per_problem.extend(data["per_problem"])

    # Re-index problem_idx
    for i, pp in enumerate(all_per_problem):
        pp["problem_idx"] = i

    n_problems = len(all_per_problem)

    # Recompute macro-average pass@k
    avg_pass_at_k = {}
    for k in k_values:
        values = [p["pass_at_k"].get(str(k)) for p in all_per_problem
                  if p["pass_at_k"].get(str(k)) is not None]
        avg_pass_at_k[str(k)] = sum(values) / len(values) if values else None

    # Summary stats
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
        "unique_code_rate_global": sum(all_unique) / len(all_unique) if all_unique else 0,
    }


def shard_parquet(eval_file, n_shards, tmpdir):
    """Split a parquet file into n_shards temp files. Returns list of paths."""
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
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_completions", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--timeout_per_test", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=64)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    args = parser.parse_args()
    args.repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)

    benchmarks = [
        ("livecodebench_v5", f"{args.repo_root}/data/livecodebench_v5_test.parquet"),
        ("humaneval_plus", f"{args.repo_root}/data/evalplus_humaneval_test.parquet"),
        ("mbpp_plus", f"{args.repo_root}/data/evalplus_mbpp_test.parquet"),
    ]

    for eval_name, eval_file in benchmarks:
        out_dir = os.path.join(args.results_dir, f"reeval_{eval_name}")
        metrics_path = os.path.join(out_dir, f"metrics_step{args.step}.json")

        if os.path.isfile(metrics_path):
            print(f"\n  {eval_name}: already done, skipping.")
            continue

        print(f"\n  === {eval_name} ===")

        with tempfile.TemporaryDirectory(prefix=f"shard_{eval_name}_") as tmpdir:
            # Split problems into shards
            shard_parquets = shard_parquet(eval_file, args.n_gpus, tmpdir)
            n_actual_shards = len(shard_parquets)
            print(f"  Split into {n_actual_shards} shards")

            # Launch one vLLM per GPU
            shard_out_files = []
            procs = []
            for i, sp in enumerate(shard_parquets):
                out_file = os.path.join(tmpdir, f"metrics_shard_{i}.json")
                shard_out_files.append(out_file)
                p = run_shard(i, args.hf_dir, sp, out_file, args)
                procs.append(p)

            # Wait for all shards
            failed = 0
            for i, p in enumerate(procs):
                rc = p.wait()
                if rc != 0:
                    print(f"  WARNING: GPU {i} exited with code {rc}")
                    failed += 1

            if failed > 0:
                print(f"  {failed} shard(s) failed, skipping merge for {eval_name}")
                continue

            # Check all shard outputs exist
            missing = [f for f in shard_out_files if not os.path.isfile(f)]
            if missing:
                print(f"  Missing shard outputs: {missing}, skipping merge")
                continue

            # Merge and save
            merged = merge_shard_metrics(shard_out_files, args.k_values)
            merged["global_step"] = args.step

            os.makedirs(out_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(merged, f, indent=2)

            pk = merged["pass_at_k"]
            pk_strs = [f"pass@{k}={pk[str(k)]:.4f}" for k in args.k_values if pk.get(str(k)) is not None]
            print(f"  {eval_name}: {' '.join(pk_strs)}")
            print(f"  Saved to {metrics_path}")

    print("\n  All benchmarks complete!")


if __name__ == "__main__":
    main()
