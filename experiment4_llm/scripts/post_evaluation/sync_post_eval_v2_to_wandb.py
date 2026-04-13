"""
Sync post_eval_v2 metrics to wandb reeval section for each training run.
Uses reeval/{benchmark}/pass_at_{k} as the metric prefix.
Routes steps to the correct wandb run segment based on step ranges.
"""

import json
import os
import sys
from collections import defaultdict

WANDB_ENTITY = "stablegradients"
WANDB_PROJECT = "cont-maxrl-code"
BENCHMARKS = ["humaneval_plus", "mbpp_plus", "livecodebench_v5"]

# Map: results_dir -> list of (wandb_run_id, min_step, max_step)
# Based on queried step ranges from wandb history
RUN_CONFIG = {
    "/work/nvme/bgts/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_grpo_g4_s0": [
        ("ze26yd4n", 1, 525),
        ("9wgrt2j6", 499, 1000),
        ("ijzrcn1i", 831, 998),
    ],
    "/work/nvme/bgts/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_maxrl_g4_s0": [
        ("ov85hpdx", 1, 413),
        ("6qzc5clk", 416, 670),
        ("mr1l8cbw", 665, 998),
    ],
    "/work/nvme/bgtu/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_grpo_g16_s0": [
        ("4vcqjozy", 1, 234),
        ("z3gickll", 167, 315),
        ("61mg27g6", 250, 969),
    ],
    "/work/nvme/bgtu/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_maxrl_g16_s0": [
        ("ntkd5y22", 1, 186),
        ("332pne89", 167, 267),
        ("03eng8j5", 250, 662),
    ],
    "/work/nvme/bgtv/sramasubramanian/cont-maxrl/results/qwen3_1.7b_base_deepcoder_rloo_g16_s0": [
        ("8ltdmw47", 1, 130),
        ("rgzxkgnr", 84, 662),
    ],
    "/work/nvme/bgmo/sramasubramanian/cont-maxrl/results/qwen3_1.7b_base_deepcoder_binary_maxrl_g16_s0": [
        ("dxlq7uga", 1, 145),
        ("yt9ef3my", 84, 740),
    ],
}


def find_best_run(step, run_segments):
    """Route a step to the best wandb run segment."""
    # Find segments that contain this step
    candidates = [(rid, mn, mx) for rid, mn, mx in run_segments if mn <= step <= mx]
    if candidates:
        # Prefer the latest segment (most recent run)
        return candidates[-1][0]
    # No exact match — find closest segment by distance
    best_rid = run_segments[-1][0]
    best_dist = float("inf")
    for rid, mn, mx in run_segments:
        dist = min(abs(step - mn), abs(step - mx))
        if dist < best_dist:
            best_dist = dist
            best_rid = rid
    return best_rid


def collect_metrics(results_dir):
    """Collect all post_eval_v2 metrics keyed by step."""
    step_metrics = defaultdict(dict)
    pev2_dir = os.path.join(results_dir, "post_eval_v2")
    if not os.path.isdir(pev2_dir):
        return step_metrics

    for benchmark in BENCHMARKS:
        bench_dir = os.path.join(pev2_dir, benchmark)
        if not os.path.isdir(bench_dir):
            continue
        for step_dir_name in os.listdir(bench_dir):
            if not step_dir_name.startswith("step_"):
                continue
            step = int(step_dir_name.replace("step_", ""))
            metrics_path = os.path.join(bench_dir, step_dir_name, "metrics.json")
            if not os.path.isfile(metrics_path):
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)

            prefix = f"reeval/{benchmark}"
            pk = metrics.get("pass_at_k", {})
            for k, v in pk.items():
                if v is not None:
                    step_metrics[step][f"{prefix}/pass_at_{k}"] = v

            # Also log summary stats
            for key in ["mean_score", "mean_pass_rate", "mean_unique_solution_rate",
                        "mean_output_length", "n_problems_solved"]:
                if key in metrics:
                    step_metrics[step][f"{prefix}/{key}"] = metrics[key]

    return step_metrics


def main():
    import wandb

    dry_run = "--dry-run" in sys.argv

    for results_dir, run_segments in RUN_CONFIG.items():
        run_name = os.path.basename(results_dir)
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"Dir: {results_dir}")

        step_metrics = collect_metrics(results_dir)
        if not step_metrics:
            print("  No post_eval_v2 metrics found, skipping.")
            continue

        print(f"  Found metrics for {len(step_metrics)} steps: {sorted(step_metrics.keys())}")

        # Group steps by wandb run
        run_step_metrics = defaultdict(dict)
        for step, mets in sorted(step_metrics.items()):
            rid = find_best_run(step, run_segments)
            run_step_metrics[rid][step] = mets

        for rid, steps_mets in run_step_metrics.items():
            steps = sorted(steps_mets.keys())
            n_metrics = sum(len(m) for m in steps_mets.values())
            print(f"\n  -> W&B run {rid}: steps {steps} ({n_metrics} metric values)")

            if dry_run:
                for step in steps:
                    print(f"     step {step}: {list(steps_mets[step].keys())}")
                continue

            run = wandb.init(
                id=rid,
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                resume="allow",
            )
            wandb.define_metric("reeval/*", step_metric="reeval/step")

            for step in steps:
                payload = {"reeval/step": step, **steps_mets[step]}
                run.log(payload)
                print(f"     Logged {len(steps_mets[step])} metrics at step {step}")

            run.finish()
            print(f"     Done uploading to {rid}")

    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
