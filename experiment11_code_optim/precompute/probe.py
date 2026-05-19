"""
§13 probe items 3-6 (the rest of the calibration after item 0).

Item 3: determinism stress test — fixed C++ program × 20, instr_count statistics
        and comparison against "inter-ref gaps" (gap between programs of obviously
        different work, which is the resolving-power floor a reference table relies on).

Item 4: throughput — parallel grading of 16 programs via ThreadPoolExecutor,
        measured programs/sec.

Item 5: precompute extrapolation — 20 random problems from train.jsonl, each graded
        once end-to-end via the unified test archive, projected to full reference-
        table build (~50 distinct programs/problem × 1474 problems × current grading time).

Item 6: step-budget projection — per-rollout reward cost × G=16 × 256 prompts / step,
        compared against DeepCoder's ~7-min step time. Spec §13 decision rule:
            GO                  if reward ≤ 50% of step time
            GO-WITH-MITIGATION  if 50-70%
            NO-GO               if > 70% post-mitigation

Outputs a JSON results file consumed by docs/PROBE_REPORT.md.

Run on a compute node:
    PYTHONPATH=. /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/probe.py
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

from experiment11_code_optim.measurement.core import measure, MEASUREMENT_BACKEND

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
TESTS_ROOT = PIE_DATA_ROOT / "tests"
TRAIN_JSONL = PIE_DATA_ROOT / "raw" / "train.jsonl"
TRAIN_HQ_JSONL = PIE_DATA_ROOT / "raw" / "train_hq_only.jsonl"
DOCS = Path(__file__).resolve().parent.parent / "docs"

# Spec §3.1a: PIE training has 3050 steps total (10 epochs of 305 steps).
# Spec §3.1 hyperparameters:
TRAIN_BATCH_SIZE = 256
ROLLOUT_N = 16  # G=16
STEPS_PER_EPOCH = 305
TOTAL_TRAINING_STEPS = 3050
ROLLOUTS_PER_STEP = TRAIN_BATCH_SIZE * ROLLOUT_N  # 4096
DEEPCODER_STEP_MIN = 7.0  # ~7 min/step per spec §13 item 6 baseline


# ----- fixture programs -----

# A program whose work depends on N read from stdin — defeats g++ -O3
# constant-folding of pure-numeric loops. Same program; different N gives
# different real work, which is exactly what an "inter-ref gap" needs.
N_LOOP_PROGRAM = r"""
#include <iostream>
int main() {
    long long n;
    std::cin >> n;
    long long s = 0;
    for (long long i = 0; i < n; i++) s = (s + i) ^ (i << 1);  // non-trivial mixing
    std::cout << s << "\n";
    return 0;
}
""".strip()


def _expected_n_loop_output(n: int) -> str:
    s = 0
    for i in range(n):
        s = (s + i) ^ (i << 1)
    return f"{s}\n"


# ----- item 3 -----

def probe_determinism() -> dict:
    """Run N_LOOP_PROGRAM with N=10M ×20, summarize instr_count distribution, and
    compare against runs of the SAME program with N=1M and N=100M.

    Using N-from-stdin defeats g++ -O3 constant-folding of pure-numeric loops:
    earlier versions used three literal loops with hard-coded N and the compiler
    collapsed all three to the same arithmetic-progression closed form, masking
    real work differences. Reading N from stdin keeps the loop in the binary.
    """
    n_fixed, n_small, n_large = 10_000_000, 1_000_000, 100_000_000
    expected_fixed = _expected_n_loop_output(n_fixed)
    samples: list[int] = []
    t0 = time.monotonic()
    for _ in range(20):
        r = measure(N_LOOP_PROGRAM, [(f"{n_fixed}\n", expected_fixed)])
        assert r["all_pass"], f"N_LOOP_PROGRAM (N={n_fixed}) failed gate: {r}"
        samples.append(r["instr_count"])
    elapsed = time.monotonic() - t0

    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples)
    spread = max(samples) - min(samples)
    rel_stdev = stdev / mean if mean else 0.0
    rel_spread = spread / mean if mean else 0.0

    r_small = measure(N_LOOP_PROGRAM, [(f"{n_small}\n", _expected_n_loop_output(n_small))])
    r_large = measure(N_LOOP_PROGRAM, [(f"{n_large}\n", _expected_n_loop_output(n_large))])
    assert r_small["all_pass"] and r_large["all_pass"]

    inter_ref_gap_small = abs(mean - r_small["instr_count"])
    inter_ref_gap_large = abs(r_large["instr_count"] - mean)
    min_inter_ref_gap = min(inter_ref_gap_small, inter_ref_gap_large)

    # Spec §13: jitter≈inter-ref gaps → stop+hand back. Use spread (max-min over 20 runs)
    # as the conservative jitter estimate; require spread < gap/100 as "comfortably resolves".
    resolves_comfortably = spread * 100 < min_inter_ref_gap
    return {
        "n_runs": len(samples),
        "elapsed_s": round(elapsed, 3),
        "instr_count_mean": int(mean),
        "instr_count_stdev": int(stdev),
        "instr_count_spread": int(spread),
        "rel_stdev": round(rel_stdev, 6),
        "rel_spread": round(rel_spread, 6),
        "smaller_program_instr_count": r_small["instr_count"],
        "larger_program_instr_count": r_large["instr_count"],
        "inter_ref_gap_to_smaller": int(inter_ref_gap_small),
        "inter_ref_gap_to_larger": int(inter_ref_gap_large),
        "min_inter_ref_gap": int(min_inter_ref_gap),
        "resolves_comfortably": resolves_comfortably,
    }


# ----- helpers for items 4 and 5 -----

def _load_test_cases(problem_id: str) -> list[tuple[str, str]]:
    pdir = TESTS_ROOT / problem_id
    n = sum(1 for f in os.listdir(pdir) if f.startswith("input.") and f.endswith(".txt"))
    cases = []
    for i in range(n):
        with open(pdir / f"input.{i}.txt") as f_in, open(pdir / f"output.{i}.txt") as f_out:
            cases.append((f_in.read(), f_out.read()))
    return cases


def _sample_train_records(n_problems: int, seed: int) -> list[dict]:
    """Pick `n_problems` distinct problem_ids from train.jsonl, return one record per."""
    rng = random.Random(seed)
    by_pid: dict[str, dict] = {}
    with open(TRAIN_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            by_pid.setdefault(rec["problem_id"], rec)
    pids = list(by_pid.keys())
    rng.shuffle(pids)
    chosen = pids[:n_problems]
    return [by_pid[pid] for pid in chosen]


def _grade_one(rec: dict) -> dict:
    pid = rec["problem_id"]
    cases = _load_test_cases(pid)
    t0 = time.monotonic()
    r = measure(rec["tgt_code"], cases)
    return {
        "problem_id": pid,
        "n_test_cases": len(cases),
        "elapsed_s": time.monotonic() - t0,
        "compiled": r["compiled"],
        "all_pass": r["all_pass"],
        "instr_count": r["instr_count"],
    }


# ----- item 4 -----

def probe_throughput(n_workers: int = 16, n_programs: int = 16, seed: int = 0) -> dict:
    """Grade `n_programs` independent records in parallel; report programs/sec.

    NOTE: this uses a flat ThreadPoolExecutor without a per-core lock on counted
    runs. Spec §4.3 requires the per-core lock for production grading; the probe
    here measures *raw* parallel throughput. If the determinism check (item 3)
    is comfortably resolving, lock-free is fine for the probe.
    """
    records = _sample_train_records(n_programs, seed=seed)
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_grade_one, records))
    elapsed = time.monotonic() - t0

    n_compiled = sum(1 for r in results if r["compiled"])
    n_all_pass = sum(1 for r in results if r["all_pass"])
    total_tests = sum(r["n_test_cases"] for r in results)
    return {
        "n_workers": n_workers,
        "n_programs": n_programs,
        "elapsed_s": round(elapsed, 3),
        "programs_per_sec": round(n_programs / elapsed, 2),
        "tests_per_sec": round(total_tests / elapsed, 2),
        "mean_grading_time_s": round(elapsed * n_workers / n_programs, 3),  # serial-equivalent
        "n_compiled": n_compiled,
        "n_all_pass": n_all_pass,
        "total_test_cases": total_tests,
        "mean_tests_per_problem": round(total_tests / n_programs, 1),
    }


# ----- item 5 -----

def probe_precompute_extrapolation(n_problems: int = 20, seed: int = 0,
                                   programs_per_problem_est: int = 50) -> dict:
    """Grade `n_problems` random problems (1 program/problem from train.jsonl), project
    to a full reference-table build assuming ~`programs_per_problem_est` distinct
    correct programs/problem (spec §7 reference population)."""
    records = _sample_train_records(n_problems, seed=seed)
    t0 = time.monotonic()
    per_prob = [_grade_one(rec) for rec in records]
    elapsed = time.monotonic() - t0

    n_all_pass = sum(1 for r in per_prob if r["all_pass"])
    mean_grade_s = elapsed / n_problems
    n_unique_train_problems = 1474  # verified in PARITY_CONFIG.md
    projected_total_s_serial = mean_grade_s * programs_per_problem_est * n_unique_train_problems
    # ARM Neoverse-V2, 72 cores on NUMA-0. Assume ~50% parallel efficiency (counted-run
    # serialisation per spec §4.3 limits concurrency below cores).
    parallel_efficiency = 0.5
    projected_total_h_parallel = (projected_total_s_serial / (72 * parallel_efficiency)) / 3600.0

    return {
        "n_problems_calibrated": n_problems,
        "elapsed_s": round(elapsed, 3),
        "mean_grading_s_per_problem": round(mean_grade_s, 3),
        "n_all_pass": n_all_pass,
        "assumed_programs_per_problem": programs_per_problem_est,
        "projected_total_s_serial": int(projected_total_s_serial),
        "projected_total_h_parallel_50pct": round(projected_total_h_parallel, 2),
        "spec_thresholds": {
            "GO_max_h": 36.0,
            "GO_WITH_MITIGATION_max_h": 44.0,
        },
        "decision_for_this_item": (
            "GO" if projected_total_h_parallel <= 36.0
            else "GO_WITH_MITIGATION" if projected_total_h_parallel <= 44.0
            else "NO_GO"
        ),
    }


# ----- item 6 -----

def probe_parallel_speedup(n_programs: int = 16, seed: int = 42) -> dict:
    """Measure realised parallel speedup at {1, 4, 16, 64} workers on the same workload.
    This is the empirical input to item 6 (replaces the prior 50%-efficiency assumption,
    which the earlier 16-worker throughput test already weakly contradicted)."""
    records = _sample_train_records(n_programs * 2, seed=seed)
    base = records[:n_programs]

    timings = {}
    # Serial baseline
    t0 = time.monotonic()
    for r in base:
        _grade_one(r)
    timings["serial_1"] = time.monotonic() - t0

    for w in (4, 16, 64):
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as pool:
            list(pool.map(_grade_one, base))
        timings[f"parallel_{w}"] = time.monotonic() - t0

    serial_throughput = n_programs / timings["serial_1"]
    speedups = {
        f"{w}_workers": round(timings["serial_1"] / timings[f"parallel_{w}"], 2)
        for w in (4, 16, 64)
    }
    best_speedup = max(speedups.values())
    return {
        "n_programs": n_programs,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "speedups": speedups,
        "best_speedup": best_speedup,
        "serial_progs_per_sec": round(serial_throughput, 3),
        "saturating_workers": (
            None if best_speedup >= 0.9 * 64
            else min((w for w in (4, 16, 64) if speedups[f"{w}_workers"] >= 0.95 * best_speedup),
                     default=64)
        ),
    }


def probe_pass_rate_sensitivity(per_problem_grade_s: float, empirical_speedup: float) -> dict:
    """Compute reward fraction at multiple assumed RL-rollout pass rates.

    Spec §4.2: 'Measure only if all_pass.' Failed rollouts skip the expensive
    perf-measurement phase entirely; their cost is compile + gate-run only.

    The probe's per-problem grade time ≈ compile (~500ms) + 14×gate (~70ms) +
    14×3×perf (~420ms) ≈ 1000ms. So passing rollouts spend ~all 1s; failing
    rollouts spend ~580ms (compile + gate only). Approximate t_fail ≈ 0.6 × t_pass.

    Realistic early-training pass rates: 10-30%. Late-training (if model converges):
    higher. The worst-case 100% projection is the spec NO-GO trigger; the realistic-
    case projection is what training will actually see.
    """
    t_pass = per_problem_grade_s
    t_fail = 0.6 * per_problem_grade_s  # compile + gate, no perf
    out = {}
    for p in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        per_rollout = p * t_pass + (1 - p) * t_fail
        serial_s = ROLLOUTS_PER_STEP * per_rollout
        parallel_min = (serial_s / empirical_speedup) / 60.0
        fraction = parallel_min / DEEPCODER_STEP_MIN
        decision = (
            "GO" if fraction <= 0.50
            else "GO_WITH_MITIGATION" if fraction <= 0.70
            else "NO_GO"
        )
        out[f"pass_rate_{p:g}"] = {
            "per_rollout_s": round(per_rollout, 3),
            "reward_min_per_step": round(parallel_min, 2),
            "reward_fraction": round(fraction, 3),
            "decision": decision,
        }
    return out


def probe_step_budget(per_problem_grade_s: float, empirical_speedup: float) -> dict:
    """Project reward fraction of step time using item 5's per-problem grade time
    and item 4's EMPIRICAL parallel speedup (the earlier hard-coded 36× was a guess;
    the parallel-speedup probe replaces it with measurement).

    A PIE training step grades ROLLOUTS_PER_STEP = 4096 generations (256 prompts × G=16).
    The probe assumes 100% of rollouts get full perf measurement; under realistic PIE
    pass rates (10-30%) the cost drops 3-10× because only all_pass rollouts are
    measured. That mitigation path is documented in the report; the headline number
    here is the *worst-case* (all rollouts measured)."""
    serial_s_per_step = ROLLOUTS_PER_STEP * per_problem_grade_s
    reward_min_per_step = (serial_s_per_step / empirical_speedup) / 60.0
    reward_fraction = reward_min_per_step / DEEPCODER_STEP_MIN

    total_h = TOTAL_TRAINING_STEPS * max(DEEPCODER_STEP_MIN, reward_min_per_step) / 60.0
    chains_needed = math.ceil(total_h / 47.0)

    decision = (
        "GO" if reward_fraction <= 0.50
        else "GO_WITH_MITIGATION" if reward_fraction <= 0.70
        else "NO_GO"
    )

    return {
        "rollouts_per_step": ROLLOUTS_PER_STEP,
        "per_problem_grade_s_input": round(per_problem_grade_s, 3),
        "empirical_speedup_used": round(empirical_speedup, 2),
        "serial_reward_s_per_step": round(serial_s_per_step, 1),
        "parallel_reward_min_per_step": round(reward_min_per_step, 2),
        "deepcoder_step_min_baseline": DEEPCODER_STEP_MIN,
        "reward_fraction_of_step_worst_case": round(reward_fraction, 3),
        "projected_total_training_h_worst_case": round(total_h, 1),
        "chains_needed_47h_each": chains_needed,
        "chain_left_default_in_scripts": 8,
        "decision_for_this_item": decision,
    }


def main():
    print("=== §13 probe items 3-6 ===")
    print(f"backend={MEASUREMENT_BACKEND}  TESTS_ROOT={TESTS_ROOT}")
    results = {"backend": MEASUREMENT_BACKEND, "tests_root": str(TESTS_ROOT)}

    print("\n[item 3] determinism stress test...")
    results["item3_determinism"] = probe_determinism()
    print(json.dumps(results["item3_determinism"], indent=2))

    print("\n[item 4] throughput...")
    results["item4_throughput"] = probe_throughput()
    print(json.dumps(results["item4_throughput"], indent=2))

    print("\n[item 5] precompute extrapolation...")
    results["item5_precompute"] = probe_precompute_extrapolation()
    print(json.dumps(results["item5_precompute"], indent=2))

    print("\n[item 4b] empirical parallel speedup...")
    results["item4b_parallel_speedup"] = probe_parallel_speedup()
    print(json.dumps(results["item4b_parallel_speedup"], indent=2))

    print("\n[item 6] step budget projection (uses empirical speedup)...")
    per_problem_grade_s = results["item5_precompute"]["mean_grading_s_per_problem"]
    empirical_speedup = results["item4b_parallel_speedup"]["best_speedup"]
    results["item6_step_budget"] = probe_step_budget(per_problem_grade_s, empirical_speedup)
    print(json.dumps(results["item6_step_budget"], indent=2))

    print("\n[item 6b] pass-rate sensitivity (realistic early-training is 10-30%)...")
    results["item6b_pass_rate_sensitivity"] = probe_pass_rate_sensitivity(
        per_problem_grade_s, empirical_speedup
    )
    print(json.dumps(results["item6b_pass_rate_sensitivity"], indent=2))

    # Report scratch dir + SLURM context for the record
    results["runtime_context"] = {
        "PIE_COMPILE_SCRATCH": os.environ.get("PIE_COMPILE_SCRATCH"),
        "PIE_SCRATCH_DIR": os.environ.get("PIE_SCRATCH_DIR"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
    }
    print(f"\nruntime_context: {results['runtime_context']}")

    # Overall decision: worst-of items 3, 5, 6
    parts = [
        ("item3_determinism", "GO" if results["item3_determinism"]["resolves_comfortably"] else "NO_GO"),
        ("item5_precompute", results["item5_precompute"]["decision_for_this_item"]),
        ("item6_step_budget", results["item6_step_budget"]["decision_for_this_item"]),
    ]
    rank = {"GO": 0, "GO_WITH_MITIGATION": 1, "NO_GO": 2}
    overall = max(parts, key=lambda kv: rank[kv[1]])[1]
    results["overall_decision"] = overall
    results["decision_breakdown"] = dict(parts)

    out_path = DOCS / "probe_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== overall decision: {overall} ===")
    print(f"breakdown: {dict(parts)}")
    print(f"results written to: {out_path}")


if __name__ == "__main__":
    main()
