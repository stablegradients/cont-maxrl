"""
Sharded eval with completion saving: same as sharded_eval_worker.py but saves
raw completions + extracted code + per-completion scores for reward hacking analysis.

Output: for each benchmark, saves:
  - completions_step{N}.json: raw completions, extracted code, scores per problem
  - metrics_step{N}.json: same as normal eval (pass@k, etc.)
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
    """Run eval on a shard, saving both metrics and raw completions."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "-c", f"""
import json, sys, math, numpy as np
sys.path.insert(0, "{args.repo_root}")
import pandas as pd
from experiment4_llm.src.eval_checkpoints import generate_completions
from experiment4_llm.src.reward_function import extract_code, compute_scores_batch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

# Monkey-patch global_timeout to 15s
import experiment4_llm.src.reward_function as _rf
_orig_batch = _rf.compute_scores_batch
def _fast_batch(reward_inputs, timeout_per_test=6, memory_limit_mb=512, num_workers=None, global_timeout=15):
    return _orig_batch(reward_inputs, timeout_per_test, memory_limit_mb, num_workers, global_timeout=15)
_rf.compute_scores_batch = _fast_batch

problems = pd.read_parquet("{problems_file}").to_dict(orient="records")
n_problems = len(problems)
print(f"  GPU {gpu_id}: generating {{n_problems}} problems...", flush=True)

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

# Score all completions
n_target = {args.n_completions}
reward_inputs = []
problem_indices = []
for i in range(n_problems):
    gt = problems[i]["reward_model"]["ground_truth"]
    if not isinstance(gt, str):
        gt = json.dumps(gt)
    for code in completions[i][:n_target]:
        reward_inputs.append({{"solution_str": code, "ground_truth": gt}})
        problem_indices.append(i)

print(f"    Scoring {{len(reward_inputs)}} completions with {args.n_workers} workers...", flush=True)
scores = _fast_batch(reward_inputs, timeout_per_test={args.timeout_per_test}, num_workers={args.n_workers})

# Build per-problem results with completions
per_problem_completions = []
per_problem_metrics = []
score_idx = 0
for i in range(n_problems):
    n_actual = min(len(completions[i]), n_target)
    raw_completions = completions[i][:n_actual]
    extracted = [extract_code(c) for c in raw_completions]
    problem_scores = scores[score_idx:score_idx + n_actual]
    score_idx += n_actual

    c = sum(1 for s in problem_scores if float(s["acc"]) == 1.0)
    n = len(problem_scores)
    mean_score = sum(float(s["score"]) for s in problem_scores) / max(n, 1)
    unique_rate = len(set(extracted)) / max(n, 1)
    mean_len = sum(len(r) for r in raw_completions) / max(n, 1)

    ppk = {{}}
    for k in {args.k_values}:
        ppk[str(k)] = pass_at_k(n, c, k) if k <= n else None

    per_problem_completions.append({{
        "problem_idx": i,
        "prompt": problems[i]["prompt"],
        "n": n,
        "c": c,
        "mean_score": mean_score,
        "completions": [
            {{
                "raw": raw_completions[j],
                "extracted_code": extracted[j],
                "score": float(problem_scores[j]["score"]),
                "acc": float(problem_scores[j]["acc"]),
            }}
            for j in range(n_actual)
        ],
    }})

    per_problem_metrics.append({{
        "problem_idx": i,
        "n": n,
        "c": c,
        "pass_at_k": ppk,
        "mean_score": mean_score,
        "unique_solution_rate": unique_rate,
        "mean_output_length": mean_len,
    }})

# Save completions
with open("{completions_file}", "w") as f:
    json.dump(per_problem_completions, f, indent=2, cls=NumpyEncoder)

# Save metrics (same format as normal eval)
with open("{metrics_file}", "w") as f:
    json.dump({{"per_problem": per_problem_metrics}}, f, indent=2, cls=NumpyEncoder)

print(f"  GPU {gpu_id}: done", flush=True)
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

    total_c = sum(p["c"] for p in all_per_problem)
    total_n = sum(p["n"] for p in all_per_problem)

    return {
        "pass_at_k": avg_pass_at_k,
        "per_problem": all_per_problem,
        "n_problems_solved": sum(1 for p in all_per_problem if p["c"] > 0),
        "mean_score": sum(p["mean_score"] for p in all_per_problem) / max(len(all_per_problem), 1),
        "mean_pass_rate": total_c / max(total_n, 1),
    }


def merge_shard_completions(shard_files):
    all_completions = []
    for sf in shard_files:
        with open(sf) as f:
            all_completions.extend(json.load(f))
    for i, pc in enumerate(all_completions):
        pc["problem_idx"] = i
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
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_completions", type=int, default=16)  # fewer completions for inspection
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--timeout_per_test", type=int, default=6)
    parser.add_argument("--n_workers", type=int, default=64)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    args.repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)

    benchmarks = [
        ("livecodebench_v5", f"{args.repo_root}/data/livecodebench_v5_test.parquet"),
        ("humaneval_plus", f"{args.repo_root}/data/evalplus_humaneval_test.parquet"),
        ("mbpp_plus", f"{args.repo_root}/data/evalplus_mbpp_test.parquet"),
    ]

    for eval_name, eval_file in benchmarks:
        out_dir = os.path.join(args.results_dir, f"completions_{eval_name}")
        completions_path = os.path.join(out_dir, f"completions_step{args.step}.json")
        metrics_path = os.path.join(out_dir, f"metrics_step{args.step}.json")

        if os.path.isfile(completions_path):
            print(f"\n  {eval_name}: completions already saved, skipping.")
            continue

        print(f"\n  === {eval_name} (saving completions) ===")

        with tempfile.TemporaryDirectory(prefix=f"shard_compl_{eval_name}_") as tmpdir:
            shard_parquets = shard_parquet(eval_file, args.n_gpus, tmpdir)
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
                print(f"  {failed} shard(s) failed, skipping {eval_name}")
                continue

            missing_m = [f for f in shard_metrics_files if not os.path.isfile(f)]
            missing_c = [f for f in shard_completions_files if not os.path.isfile(f)]
            if missing_m or missing_c:
                print(f"  Missing outputs, skipping merge")
                continue

            merged_metrics = merge_shard_metrics(shard_metrics_files, args.k_values)
            merged_metrics["global_step"] = args.step
            merged_completions = merge_shard_completions(shard_completions_files)

            os.makedirs(out_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(merged_metrics, f, indent=2)
            with open(completions_path, "w") as f:
                json.dump(merged_completions, f, indent=2)

            pk = merged_metrics["pass_at_k"]
            pk_strs = [f"pass@{k}={pk[str(k)]:.4f}" for k in args.k_values if pk.get(str(k)) is not None]
            print(f"  {eval_name}: {' '.join(pk_strs)}")
            print(f"  Saved completions to {completions_path}")

    print("\n  All benchmarks complete!")


if __name__ == "__main__":
    main()
