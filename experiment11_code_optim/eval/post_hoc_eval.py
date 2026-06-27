"""
Post-hoc per-checkpoint best@K / pass@K eval (spec v9 §8.5).

For every saved actor checkpoint under `$PIE_ACTOR_CARVEOUT_DIR/step_<N>/`,
this script:

  1. Merges FSDP shards → HF format via verl's model_merger (skips if already done).
  2. Generates n_completions samples per val problem with vLLM.
  3. Grades each completion via `compute_scores_batch` — gem5 speedup reward when
     PIE_REWARD_IMPL=gem5 (the live runs), else the legacy perf/ref-table reward.
     MUST match how the checkpoint was trained (see score_completions).
  4. Computes pass@K and best@K at K ∈ {1, 4, 16, 64, 256}.
  5. Writes `metrics_step_<N>.json` + an `eval_done.json` sentinel so re-runs are
     idempotent — a checkpoint with `eval_done.json` is skipped.

Idempotency contract: `eval_done.json` is written atomically AFTER the metrics
file. If a run is killed mid-eval, the metrics file may exist but `eval_done.json`
will not, and the next invocation re-evaluates that step. This is important
under the upfront-chaining SLURM pattern (multiple resumes per run).

Run after RL training completes (or in parallel as a separate SLURM job):

    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        python -m experiment11_code_optim.eval.post_hoc_eval \\
        --carveout_dir /work/nvme/.../actor_ckpts/ \\
        --val_parquet /work/nvme/.../pie_val_200.parquet \\
        --base_model Qwen/Qwen3-1.7B-Base \\
        --n_completions 256
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


# ----------------------------------------------------------------------------
# best@K / pass@K
# ----------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator. n = total samples, c = #correct, k = pick size."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def best_at_k(scores: list[float], k: int) -> float:
    """Unbiased estimator of E[max(K random samples from n scores)].

    Sort scores descending; for the i-th score (1-indexed), the probability it
    is the max in a uniform K-subset of n is C(n-i, k-1)/C(n, k). Numerically
    this collapses scores below n-k+1 (probability 0). For k > n, returns max.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    if k >= n:
        return max(scores)
    s = sorted(scores, reverse=True)
    denom = math.comb(n, k)
    total = 0.0
    for i, x in enumerate(s, start=1):
        rem = n - i
        if rem < k - 1:
            break
        total += x * math.comb(rem, k - 1)
    return total / denom


# ----------------------------------------------------------------------------
# Checkpoint discovery / merge
# ----------------------------------------------------------------------------

def discover_carveouts(carveout_dir: Path) -> list[tuple[int, Path]]:
    """Return (step, path) for every complete step_<N>/ under carveout_dir.
    'Complete' = has the `.carveout_complete` sentinel (written by the save hook)."""
    out: list[tuple[int, Path]] = []
    if not carveout_dir.is_dir():
        return out
    for entry in sorted(carveout_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("step_"):
            continue
        try:
            step = int(entry.name[len("step_"):])
        except ValueError:
            continue
        if not (entry / ".carveout_complete").exists():
            print(f"[discover] skipping {entry.name} (no .carveout_complete sentinel)",
                  file=sys.stderr)
            continue
        out.append((step, entry))
    return sorted(out)


def merge_fsdp_to_hf(actor_dir: Path, hf_dir: Path, base_model: str,
                    merger_script: str) -> None:
    """Idempotent shard merge."""
    cfg = hf_dir / "config.json"
    if cfg.is_file():
        return
    hf_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge] {actor_dir} -> {hf_dir}")
    subprocess.run(
        [sys.executable, merger_script,
         "--backend", "fsdp",
         "--hf_model_path", base_model,
         "--local_dir", str(actor_dir),
         "--target_dir", str(hf_dir)],
        check=True,
    )


# ----------------------------------------------------------------------------
# Generation + scoring
# ----------------------------------------------------------------------------

def generate_completions(model_path: str, prompts: list[str], n_completions: int,
                         max_tokens: int = 4096, temperature: float = 0.6,
                         top_p: float = 0.95,
                         tensor_parallel_size: int = 4) -> list[list[str]]:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=max_tokens + 2816,  # headroom for the up-to-2560-token PIE source-laden prompt (+256 slack)
        enforce_eager=True,
    )
    sp = SamplingParams(
        n=n_completions, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outputs]


def score_completions(completions: list[list[str]], ground_truths: list[str],
                      test_case_dir: str, reference_table_dir: str | None,
                      timeout_per_test: int = 6) -> list[list[dict]]:
    """For each problem i and each of its n_completions completions, return the
    score dict. The scorer must MATCH how the checkpoint was trained, else best@K
    is computed under a different reward than the run optimized:
      - PIE_REWARD_IMPL=gem5 -> reward/gem5_reward.py (live gem5 speedup); the
        val parquet's ground_truth must carry usable_case_ids + gem5_src_per_tc_ticks
        (i.e. pass --val_parquet pie_gem5_val_*.parquet). reference_table_dir is unused.
      - else (default) -> legacy pie_reward_fn (perf/ref-table percentile reward)."""
    gem5 = os.environ.get("PIE_REWARD_IMPL") == "gem5"
    if gem5:
        from experiment11_code_optim.reward.gem5_reward import compute_scores_batch
    else:
        from experiment11_code_optim.reward.pie_reward_fn import compute_scores_batch

    # Flatten: pair each completion with its ground truth, score in one big batch.
    flat_inputs = []
    flat_owner = []  # (problem_idx, completion_idx)
    for i, comps in enumerate(completions):
        for j, c in enumerate(comps):
            flat_inputs.append({"solution_str": c, "ground_truth": ground_truths[i]})
            flat_owner.append((i, j))

    if gem5:
        # gem5_reward reads PIE_GEM5_* from env; native gate timeout is PIE_NAT_TIMEOUT
        results = compute_scores_batch(flat_inputs, test_case_dir=test_case_dir)
    else:
        results = compute_scores_batch(
            flat_inputs,
            test_case_dir=test_case_dir,
            reference_table_dir=reference_table_dir,
            timeout_per_test=timeout_per_test,
        )

    # Scatter back into per-problem lists.
    per_problem: list[list[dict]] = [[None] * len(comps) for comps in completions]
    for (i, j), r in zip(flat_owner, results):
        per_problem[i][j] = r
    return per_problem


# ----------------------------------------------------------------------------
# Per-checkpoint orchestration
# ----------------------------------------------------------------------------

def evaluate_checkpoint(
    *,
    step: int,
    actor_dir: Path,
    val_parquet: str,
    base_model: str,
    merger_script: str,
    output_dir: Path,
    test_case_dir: str,
    reference_table_dir: str | None,
    n_completions: int,
    k_values: list[int],
    tensor_parallel_size: int,
) -> dict | None:
    """Eval one checkpoint. Returns metrics dict or None if skipped (already done)."""
    metrics_path = output_dir / f"metrics_step_{step}.json"
    done_path = output_dir / f"step_{step}.eval_done.json"
    if done_path.exists():
        print(f"[step {step}] eval_done.json present, skipping")
        return None

    hf_dir = actor_dir / "huggingface"  # verl's save hook drops one here already
    if not (hf_dir / "config.json").exists():
        # Need to merge FSDP shards
        merge_fsdp_to_hf(actor_dir, hf_dir, base_model, merger_script)

    df = pd.read_parquet(val_parquet)
    # Apply chat template so rendered prompts match what RL trained on.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    def _render(user_content: str) -> str:
        return _tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    prompts: list[str] = []
    ground_truths: list[str] = []
    for _, row in df.iterrows():
        prompt_text = next(
            (m["content"] for m in row["prompt"] if m.get("role") == "user"),
            None,
        )
        if prompt_text is None:
            continue
        prompts.append(_render(prompt_text))
        ground_truths.append(row["reward_model"]["ground_truth"])

    print(f"[step {step}] generating {n_completions} x {len(prompts)} completions...")
    t0 = time.monotonic()
    completions = generate_completions(
        str(hf_dir), prompts, n_completions,
        tensor_parallel_size=tensor_parallel_size,
    )
    print(f"[step {step}] generation {time.monotonic()-t0:.0f}s")

    print(f"[step {step}] scoring...")
    t0 = time.monotonic()
    per_problem_results = score_completions(
        completions, ground_truths, test_case_dir, reference_table_dir,
    )
    print(f"[step {step}] scoring {time.monotonic()-t0:.0f}s")

    # Per-problem stats
    per_problem_pass_at_k = {f"pass@{k}": [] for k in k_values}
    per_problem_best_at_k = {f"best@{k}": [] for k in k_values}
    for i, results in enumerate(per_problem_results):
        scores = [r.get("score", 0.0) for r in results]
        accs = [r.get("acc", 0.0) for r in results]
        n = len(results)
        c = sum(1 for a in accs if a == 1.0)
        for k in k_values:
            if k > n:
                continue
            per_problem_pass_at_k[f"pass@{k}"].append(pass_at_k(n, c, k))
            per_problem_best_at_k[f"best@{k}"].append(best_at_k(scores, k))

    metrics = {
        "step": step,
        "n_problems": len(per_problem_results),
        "n_completions": n_completions,
        "k_values": k_values,
        "pass_at_k": {
            k: (sum(vs) / max(len(vs), 1)) for k, vs in per_problem_pass_at_k.items()
        },
        "best_at_k": {
            k: (sum(vs) / max(len(vs), 1)) for k, vs in per_problem_best_at_k.items()
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    # Idempotency sentinel — written LAST, atomic.
    done_path.write_text(json.dumps({"step": step, "metrics_path": str(metrics_path)}))
    print(f"[step {step}] wrote {metrics_path}")
    return metrics


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carveout_dir", required=True,
                    help="Directory of step_<N>/ checkpoints (PIE_ACTOR_CARVEOUT_DIR).")
    ap.add_argument("--val_parquet", required=True,
                    help="pie_val_200.parquet for the §9 GO/NO-GO set.")
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write metrics_step_<N>.json (default: carveout_dir/post_hoc_eval/).")
    ap.add_argument("--test_case_dir",
                    default=os.environ.get("PIE_TEST_CASE_DIR",
                                           "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"))
    ap.add_argument("--reference_table_dir",
                    default=os.environ.get("PIE_REFERENCE_TABLE_DIR",
                                           "/work/nvme/bgmo/sramasubramanian/pie_data/reference_table"))
    ap.add_argument("--merger_script",
                    default="/projects/bgmo/sramasubramanian/cont-maxrl/vendor/verl/scripts/model_merger.py")
    ap.add_argument("--n_completions", type=int, default=256)
    ap.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="Optional whitelist of step numbers to evaluate.")
    args = ap.parse_args()

    carveout_dir = Path(args.carveout_dir)
    output_dir = Path(args.output_dir or (carveout_dir / "post_hoc_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_dirs = discover_carveouts(carveout_dir)
    if args.steps is not None:
        whitelist = set(args.steps)
        steps_dirs = [(s, p) for s, p in steps_dirs if s in whitelist]
    if not steps_dirs:
        print(f"[main] no checkpoints to evaluate under {carveout_dir}",
              file=sys.stderr)
        return 1

    print(f"[main] {len(steps_dirs)} checkpoints to evaluate -> {output_dir}")
    n_done = n_skip = n_fail = 0
    for step, actor_dir in steps_dirs:
        try:
            r = evaluate_checkpoint(
                step=step,
                actor_dir=actor_dir,
                val_parquet=args.val_parquet,
                base_model=args.base_model,
                merger_script=args.merger_script,
                output_dir=output_dir,
                test_case_dir=args.test_case_dir,
                reference_table_dir=args.reference_table_dir,
                n_completions=args.n_completions,
                k_values=args.k_values,
                tensor_parallel_size=args.tensor_parallel_size,
            )
            if r is None:
                n_skip += 1
            else:
                n_done += 1
        except Exception as e:
            print(f"[step {step}] FAILED: {e}", file=sys.stderr)
            n_fail += 1

    print(f"\n=== SUMMARY ===")
    print(f"  done: {n_done}  skipped: {n_skip}  failed: {n_fail}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
