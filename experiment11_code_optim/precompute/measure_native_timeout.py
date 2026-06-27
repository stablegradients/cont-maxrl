"""Measure native-correctness wall time of SRC programs to ground PIE_NAT_TIMEOUT.

Mirrors the reward's phase-1 native gate: compile src with the screen's
g++ -O0 (_compile), then run ALL usable cases sequentially on ONE core (exactly
what one reward thread does per rollout), timing per-case and per-rollout-total.
Parallel ACROSS src programs (one thread per src, like the reward's pool), so the
reported totals reflect what a single reward worker experiences per rollout.

Reports the distribution so we can set:
  native_timeout_s        (PIE_NAT_TIMEOUT)            >= max single-case wall
  native_rollout_budget_s (PIE_GEM5_REWARD_NATIVE_BUDGET) >= max per-rollout total
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, subprocess, tempfile, shutil
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
from experiment11_code_optim.measurement.core import _compile, _outputs_match, setup_pch  # noqa

RX = re.compile(r"```cpp\n(.*?)\n```", re.S)
TCDIR = os.environ.get("PIE_TEST_CASE_DIR",
                       "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases")


def run_one(src_code, pid, usable, src_ticks, per_case_to, pch):
    tmp = tempfile.mkdtemp(prefix="natto_")
    try:
        exe, cerr = _compile(src_code, tmp, pch_include_path=pch)
        if exe is None:
            exe, cerr = _compile(src_code, tmp, pch_include_path=None)
        if exe is None:
            return {"pid": pid, "compile_fail": True}
        order = sorted(usable, key=lambda c: src_ticks.get(c, 1 << 62))  # cheapest-first
        total = 0.0; mx = 0.0; npass = 0; failed = None
        for c in order:
            ip = os.path.join(TCDIR, pid, f"input.{c}.txt")
            op = os.path.join(TCDIR, pid, f"output.{c}.txt")
            if not (os.path.exists(ip) and os.path.exists(op)):
                continue
            with open(op, "r", errors="replace") as f:
                exp = f.read()
            t0 = time.monotonic()
            try:
                with open(ip, "rb") as fin:
                    r = subprocess.run([exe], stdin=fin, capture_output=True,
                                       timeout=per_case_to)
                w = time.monotonic() - t0
                out = r.stdout.decode("utf-8", "replace")
                ok = (r.returncode == 0) and _outputs_match(out, exp)
            except subprocess.TimeoutExpired:
                w = time.monotonic() - t0; ok = False; failed = f"case{c}:timeout>{per_case_to}s"
            total += w; mx = max(mx, w)
            if ok:
                npass += 1
            else:
                if failed is None:
                    failed = f"case{c}:mismatch_or_rc"
                break
        return {"pid": pid, "n_cases": len(order), "n_pass": npass,
                "total_wall_s": total, "max_case_wall_s": mx, "failed": failed}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet/pie_gem5_train.parquet")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--per_case_timeout", type=float, default=30.0)
    ap.add_argument("--out", default="/projects/bgmo/sramasubramanian/cont-maxrl/experiment11_code_optim/precompute/native_timeout_measure.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    df = df.sample(n=min(a.n, len(df)), random_state=a.seed).reset_index(drop=True)
    scratch = os.environ.get("TMPDIR", "/tmp")
    try:
        pch = setup_pch(scratch)
    except Exception:
        pch = None
    jobs = []
    for _, row in df.iterrows():
        gt = json.loads(row["reward_model"]["ground_truth"])
        m = RX.search(row["prompt"][0]["content"])
        if not m:
            continue
        usable = [int(c) for c in gt["usable_case_ids"]]
        ticks = {int(k): int(v) for k, v in gt["gem5_src_per_tc_ticks"].items()}
        jobs.append((m.group(1), str(gt["problem_id"]), usable, ticks))
    print(f"measuring {len(jobs)} src programs, {a.workers} parallel, per_case_to={a.per_case_timeout}s", flush=True)

    t0 = time.monotonic(); res = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, *j, a.per_case_timeout, pch) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            res.append(f.result())
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(futs)} done ({time.monotonic()-t0:.0f}s)", flush=True)

    ok = [r for r in res if r.get("n_pass", 0) and not r.get("compile_fail")]
    tot = np.array([r["total_wall_s"] for r in ok])
    mxc = np.array([r["max_case_wall_s"] for r in ok])
    cf = sum(1 for r in res if r.get("compile_fail"))
    fl = sum(1 for r in ok if r.get("failed"))
    print(f"\n=== {len(ok)}/{len(res)} measured ok (compile_fail={cf}, native_fail={fl}) ===")
    print("PER-ROLLOUT TOTAL native wall (s):")
    for q in [50, 90, 95, 99, 100]:
        print(f"  p{q:>4}: {np.percentile(tot,q):.2f}")
    print(f"  mean {tot.mean():.2f}  max {tot.max():.2f}")
    print("MAX SINGLE-CASE native wall (s):")
    for q in [50, 90, 95, 99, 100]:
        print(f"  p{q:>4}: {np.percentile(mxc,q):.2f}")
    print(f"  mean {mxc.mean():.2f}  max {mxc.max():.2f}")
    # slowest few
    sl = sorted(ok, key=lambda r: r["total_wall_s"], reverse=True)[:8]
    print("slowest 8 (pid, total_s, max_case_s, n_cases):")
    for r in sl:
        print(f"  {r['pid']}  total={r['total_wall_s']:.2f}  maxcase={r['max_case_wall_s']:.2f}  n={r['n_cases']}")
    json.dump({"n_ok": len(ok), "total_wall": tot.tolist(), "max_case_wall": mxc.tolist(),
               "slowest": sl}, open(a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
