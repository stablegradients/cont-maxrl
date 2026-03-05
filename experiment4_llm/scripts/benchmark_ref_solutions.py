#!/usr/bin/env python3
"""
Benchmark reference solutions from the TACO dataset.

Runs ALL test cases for each problem in a SINGLE subprocess (not one per test),
measuring per-test wall-clock time. This is much faster than spawning a process
per test case.

Output: timing summary + JSON results file.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
import time
import multiprocessing as mp
from pathlib import Path

import sys as _sys
_sys.set_int_max_str_digits(100000)  # Python 3.10 safety limit for huge ints in TACO

PROJ = Path(__file__).resolve().parents[2]
PYTHON = "/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
PER_PROBLEM_TIMEOUT = 120   # kill entire problem after 120s
PER_TEST_TIMEOUT = 30       # per-test timeout inside the subprocess
MEMORY_MB = 512
NUM_WORKERS = 30

# ── Runner that executes ALL tests in one process and reports per-test times ──
RUNNER_TEMPLATE = r'''
import sys, io, json, time, resource, signal
sys.set_int_max_str_digits(100000)

mem_bytes = {mem_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except ValueError:
    pass

per_test_timeout = {per_test_timeout}

def timeout_handler(signum, frame):
    raise TimeoutError("test timeout")

with open(sys.argv[1]) as f:
    code = f.read()
with open(sys.argv[2]) as f:
    data = json.load(f)

inputs = data.get("inputs", [])
outputs = data.get("outputs", [])
fn_name = data.get("fn_name", None)
n = min(len(inputs), len(outputs))

# Compile once
try:
    compiled = compile(code, '<solution>', 'exec')
except SyntaxError:
    # All tests fail
    result = {{"times": [0.0]*n, "passed": [False]*n, "errors": ["SyntaxError"]*n}}
    print(json.dumps(result))
    sys.exit(0)

times = []
passed = []
errors = []

if fn_name is None:
    # stdin/stdout mode
    for i in range(n):
        inp = str(inputs[i]) if not isinstance(inputs[i], str) else inputs[i]
        exp = str(outputs[i]) if not isinstance(outputs[i], str) else outputs[i]

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        sys.stdin = io.StringIO(inp)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(per_test_timeout)
        t0 = time.monotonic()
        try:
            ns = {{}}
            exec(compiled, ns)
            elapsed = time.monotonic() - t0
            signal.alarm(0)
            actual = captured.getvalue().strip()
            ok = (actual == exp.strip())
            times.append(round(elapsed, 6))
            passed.append(ok)
            errors.append(None)
        except TimeoutError:
            elapsed = time.monotonic() - t0
            signal.alarm(0)
            times.append(round(elapsed, 6))
            passed.append(False)
            errors.append("timeout")
        except Exception as e:
            elapsed = time.monotonic() - t0
            signal.alarm(0)
            times.append(round(elapsed, 6))
            passed.append(False)
            errors.append(type(e).__name__)
        finally:
            sys.stdout = old_stdout
            sys.stdin = sys.__stdin__
else:
    # function call mode
    ns = {{}}
    try:
        exec(compiled, ns)
    except Exception as e:
        result = {{"times": [0.0]*n, "passed": [False]*n, "errors": [type(e).__name__]*n}}
        print(json.dumps(result))
        sys.exit(0)

    fn = ns.get(fn_name)
    if fn is None:
        result = {{"times": [0.0]*n, "passed": [False]*n, "errors": ["fn_not_found"]*n}}
        print(json.dumps(result))
        sys.exit(0)

    for i in range(n):
        inp_str = str(inputs[i]) if not isinstance(inputs[i], str) else inputs[i]
        inp_str = inp_str.strip()
        try:
            args = json.loads(inp_str)
            if not isinstance(args, list):
                args = [args]
        except (json.JSONDecodeError, ValueError):
            args = [inp_str] if inp_str else []

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(per_test_timeout)
        t0 = time.monotonic()
        try:
            ret = fn(*args)
            elapsed = time.monotonic() - t0
            signal.alarm(0)

            actual_str = json.dumps(ret)
            exp = outputs[i]
            exp_str = str(exp) if not isinstance(exp, str) else exp

            # Compare
            ok = False
            if actual_str.strip() == exp_str.strip():
                ok = True
            else:
                try:
                    exp_val = json.loads(exp_str)
                    if ret == exp_val:
                        ok = True
                    elif isinstance(exp_val, list) and len(exp_val) == 1 and ret == exp_val[0]:
                        ok = True
                except:
                    pass

            times.append(round(elapsed, 6))
            passed.append(ok)
            errors.append(None)
        except TimeoutError:
            elapsed = time.monotonic() - t0
            signal.alarm(0)
            times.append(round(elapsed, 6))
            passed.append(False)
            errors.append("timeout")
        except Exception as e:
            elapsed = time.monotonic() - t0
            signal.alarm(0)
            times.append(round(elapsed, 6))
            passed.append(False)
            errors.append(type(e).__name__)

result = {{"times": times, "passed": passed, "errors": errors}}
print(json.dumps(result))
'''


def benchmark_one(args):
    """Benchmark one problem in a single subprocess."""
    idx, solution, test_json = args
    try:
        tc = json.loads(test_json)
    except (json.JSONDecodeError, TypeError):
        return {"idx": idx, "error": "bad_json", "max_time": 0, "n_tests": 0,
                "passed": 0, "timed_out": 0, "errored": 0, "mean_time": 0, "total_time": 0}

    n = min(len(tc.get("inputs", [])), len(tc.get("outputs", [])))
    if n == 0:
        return {"idx": idx, "error": "no_tests", "max_time": 0, "n_tests": 0,
                "passed": 0, "timed_out": 0, "errored": 0, "mean_time": 0, "total_time": 0}

    tmp_dir = "/tmp"
    code_path = tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=tmp_dir, delete=False)
    data_path = tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=tmp_dir, delete=False)
    runner_path = tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=tmp_dir, delete=False)

    try:
        code_path.write(solution)
        code_path.close()
        data_path.write(test_json)
        data_path.close()
        runner = RUNNER_TEMPLATE.format(mem_mb=MEMORY_MB, per_test_timeout=PER_TEST_TIMEOUT)
        runner_path.write(runner)
        runner_path.close()

        t0 = time.monotonic()
        result = subprocess.run(
            [PYTHON, runner_path.name, code_path.name, data_path.name],
            capture_output=True, text=True, timeout=PER_PROBLEM_TIMEOUT,
            env={"PATH": os.environ.get("PATH", ""), "HOME": "/tmp", "TMPDIR": tmp_dir},
        )
        wall = time.monotonic() - t0

        if result.returncode != 0:
            return {"idx": idx, "error": f"exit_{result.returncode}",
                    "max_time": wall, "n_tests": n, "passed": 0,
                    "timed_out": 0, "errored": n, "mean_time": 0, "total_time": wall}

        out = json.loads(result.stdout)
        times = out["times"]
        p = sum(out["passed"])
        to = sum(1 for e in out["errors"] if e == "timeout")
        er = sum(1 for e in out["errors"] if e and e != "timeout")

        return {
            "idx": idx, "n_tests": n, "passed": p, "timed_out": to, "errored": er,
            "max_time": max(times) if times else 0,
            "mean_time": sum(times)/len(times) if times else 0,
            "total_time": sum(times),
            "p99_time": sorted(times)[int(len(times)*0.99)] if len(times) >= 2 else (max(times) if times else 0),
        }

    except subprocess.TimeoutExpired:
        return {"idx": idx, "error": "problem_timeout", "max_time": PER_PROBLEM_TIMEOUT,
                "n_tests": n, "passed": 0, "timed_out": n, "errored": 0,
                "mean_time": 0, "total_time": PER_PROBLEM_TIMEOUT}
    except Exception as e:
        return {"idx": idx, "error": str(e)[:200], "max_time": 0, "n_tests": n,
                "passed": 0, "timed_out": 0, "errored": n, "mean_time": 0, "total_time": 0}
    finally:
        for p in [code_path, data_path, runner_path]:
            try:
                os.unlink(p.name)
            except OSError:
                pass


def main():
    from datasets import load_dataset
    import numpy as np

    print("Loading TACO dataset...", flush=True)
    ds = load_dataset("BAAI/TACO", split="train",
                      cache_dir="/projects/bgmo/sramasubramanian/cont-maxrl/.cache/huggingface")
    print(f"Loaded {len(ds)} problems", flush=True)

    work = []
    skipped = 0
    for i, row in enumerate(ds):
        try:
            solutions = json.loads(row.get("solutions", "[]"))
        except (json.JSONDecodeError, TypeError):
            solutions = []
        if not solutions or not isinstance(solutions, list) or not solutions[0]:
            skipped += 1
            continue
        work.append((i, solutions[0], row["input_output"]))

    print(f"Benchmarking {len(work)} problems ({skipped} skipped)", flush=True)
    print(f"Workers={NUM_WORKERS}, per_test_timeout={PER_TEST_TIMEOUT}s, "
          f"per_problem_timeout={PER_PROBLEM_TIMEOUT}s", flush=True)

    t_start = time.monotonic()
    results = []
    done = 0

    with mp.Pool(NUM_WORKERS) as pool:
        for r in pool.imap_unordered(benchmark_one, work, chunksize=1):
            results.append(r)
            done += 1
            if done % 100 == 0:
                elapsed = time.monotonic() - t_start
                rate = done / elapsed
                eta = (len(work) - done) / rate if rate > 0 else 0
                # Running stats
                mt = [x["max_time"] for x in results if x["max_time"] > 0]
                cur_max = max(mt) if mt else 0
                print(f"  [{done:5d}/{len(work)}] {elapsed:7.0f}s elapsed, "
                      f"ETA {eta:.0f}s ({rate:.1f}/s) | running max_time={cur_max:.3f}s",
                      flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"\nDone in {total_elapsed:.0f}s", flush=True)

    # Save
    out_path = str(PROJ / "experiment4_llm" / "scripts" / "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out_path}", flush=True)

    # ── Summary ──
    max_times = np.array([r["max_time"] for r in results if r.get("n_tests", 0) > 0])
    # Only consider problems where the reference solution actually passed tests
    passing = [r for r in results if r.get("passed", 0) > 0]
    pass_max = np.array([r["max_time"] for r in passing])

    total_tests = sum(r.get("n_tests", 0) for r in results)
    total_passed = sum(r.get("passed", 0) for r in results)
    total_to = sum(r.get("timed_out", 0) for r in results)
    total_er = sum(r.get("errored", 0) for r in results)
    prob_to = sum(1 for r in results if r.get("error") == "problem_timeout")

    print(f"\n{'='*65}")
    print(f"REFERENCE SOLUTION TIMING — ALL {len(results)} PROBLEMS")
    print(f"{'='*65}")
    print(f"Total test executions: {total_tests}")
    print(f"  Passed:    {total_passed} ({100*total_passed/max(total_tests,1):.1f}%)")
    print(f"  Timed out: {total_to}")
    print(f"  Errors:    {total_er}")
    print(f"  Problem-level timeouts (>{PER_PROBLEM_TIMEOUT}s): {prob_to}")

    for label, arr in [("ALL problems", max_times),
                        ("PASSING problems (>=1 test passed)", pass_max)]:
        if len(arr) == 0:
            continue
        print(f"\n  Max test time distribution — {label} ({len(arr)} problems):")
        print(f"    Max:    {arr.max():.3f}s")
        print(f"    P99.9:  {np.percentile(arr, 99.9):.3f}s")
        print(f"    P99.5:  {np.percentile(arr, 99.5):.3f}s")
        print(f"    P99:    {np.percentile(arr, 99):.3f}s")
        print(f"    P95:    {np.percentile(arr, 95):.3f}s")
        print(f"    P90:    {np.percentile(arr, 90):.3f}s")
        print(f"    P75:    {np.percentile(arr, 75):.3f}s")
        print(f"    Median: {np.median(arr):.3f}s")
        print(f"    Mean:   {arr.mean():.3f}s")
        for t in [0.5, 1, 2, 3, 5, 10, 15, 20, 30]:
            c = (arr > t).sum()
            if c > 0:
                print(f"    > {t:5.1f}s: {c:5d} ({100*c/len(arr):.2f}%)")

    # Top 30 slowest passing problems
    passing_sorted = sorted(passing, key=lambda r: r["max_time"], reverse=True)
    print(f"\nTop 30 slowest PASSING problems:")
    for r in passing_sorted[:30]:
        print(f"  idx={r['idx']:6d}  max={r['max_time']:8.3f}s  "
              f"mean={r['mean_time']:.3f}s  tests={r['n_tests']}  "
              f"passed={r['passed']}/{r['n_tests']}")


if __name__ == "__main__":
    main()
