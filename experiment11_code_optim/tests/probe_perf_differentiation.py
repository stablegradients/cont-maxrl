"""Gate 3 probe — perf cross-program differentiation.

Build a set of programs from "trivial loop" to "O(n^3) computation".
Grade each through _run_and_compare_perf. Pass if max/min >= 5x.

Sanity check that the metric actually discriminates between fast and slow
programs — if it didn't, we'd be attaching the wrong PID or counting only
some subset of instructions.

Usage:
    PIE_BACKEND=perf PYTHONPATH=. python experiment11_code_optim/tests/probe_perf_differentiation.py \
        --output-json /path/to/out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiment11_code_optim.measurement.core import (  # noqa: E402
    _compile, _run_and_compare_perf, MEASUREMENT_BACKEND, setup_pch,
)


PROGRAMS = [
    ("trivial_print", """
#include <iostream>
int main() { std::cout << "hi" << "\\n"; return 0; }
""", "", "hi\n"),

    ("loop_1k", """
#include <iostream>
int main() {
    long s = 0;
    for (int i = 0; i < 1000; ++i) s += i;
    std::cout << s << "\\n"; return 0;
}
""", "", "499500\n"),

    ("loop_100k", """
#include <iostream>
int main() {
    long s = 0;
    for (int i = 0; i < 100000; ++i) s += i;
    std::cout << s << "\\n"; return 0;
}
""", "", "4999950000\n"),

    ("loop_10m", """
#include <iostream>
int main() {
    long long s = 0;
    for (long long i = 0; i < 10000000; ++i) s += i;
    std::cout << s << "\\n"; return 0;
}
""", "", "49999995000000\n"),

    ("o_n_squared_5k", """
#include <iostream>
int main() {
    long s = 0;
    for (int i = 0; i < 5000; ++i)
        for (int j = 0; j < 5000; ++j)
            s += (i ^ j);
    std::cout << s << "\\n"; return 0;
}
""", "", None),  # output recorded at runtime

    ("o_n_cubed_300", """
#include <iostream>
int main() {
    long s = 0;
    for (int i = 0; i < 300; ++i)
        for (int j = 0; j < 300; ++j)
            for (int k = 0; k < 300; ++k)
                s += (i ^ j ^ k);
    std::cout << s << "\\n"; return 0;
}
""", "", None),

    ("vector_push_1m", """
#include <iostream>
#include <vector>
int main() {
    std::vector<int> v;
    v.reserve(1000000);
    for (int i = 0; i < 1000000; ++i) v.push_back(i);
    long s = 0;
    for (int x : v) s += x;
    std::cout << s << "\\n"; return 0;
}
""", "", "499999500000\n"),

    ("string_concat_10k", """
#include <iostream>
#include <string>
int main() {
    std::string a;
    for (int i = 0; i < 10000; ++i) a += "x";
    std::cout << a.size() << "\\n"; return 0;
}
""", "", "10000\n"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    if MEASUREMENT_BACKEND != "perf":
        print(f"[gate3] WARNING: MEASUREMENT_BACKEND={MEASUREMENT_BACKEND!r}", flush=True)

    pch = os.environ.get("PIE_PCH_INCLUDE_PATH") or setup_pch()
    os.environ["PIE_PCH_INCLUDE_PATH"] = pch

    results = []
    with tempfile.TemporaryDirectory(
        dir=os.environ.get("PIE_COMPILE_SCRATCH") or None, prefix="pie_g3_"
    ) as td:
        for name, code, inp, exp in PROGRAMS:
            sub = os.path.join(td, name)
            os.makedirs(sub, exist_ok=True)
            exe, err = _compile(code, sub, pch_include_path=pch)
            if exe is None:
                print(f"[gate3] compile FAIL: {name}: {err}", flush=True)
                results.append({"name": name, "ok": False, "err": err})
                continue
            # If expected output is None (compute-bound programs), capture stdout
            # by running once unchecked.
            if exp is None:
                import subprocess
                proc = subprocess.run(
                    [exe], input=inp, capture_output=True, text=True,
                    errors="replace", timeout=30.0,
                    cwd=os.path.dirname(exe),
                )
                if proc.returncode != 0:
                    results.append({"name": name, "ok": False, "err": "first run failed"})
                    continue
                exp = proc.stdout
            # Median of 3 grades (steady-state)
            counts = []
            for _ in range(3):
                passed, c = _run_and_compare_perf(exe, inp, exp, 30.0)
                counts.append((passed, c))
            ok_counts = [c for p, c in counts if p]
            if not ok_counts:
                results.append({"name": name, "ok": False, "err": "all 3 failed gate",
                                "counts": counts})
                continue
            med = sorted(ok_counts)[len(ok_counts) // 2]
            results.append({"name": name, "ok": True, "counts": ok_counts, "median": med})
            print(f"[gate3] {name:20s}  median={med:>15,}", flush=True)

    ok = [r for r in results if r.get("ok")]
    if len(ok) < 2:
        print(f"[gate3] FAIL: only {len(ok)} programs graded", flush=True)
        sys.exit(1)
    medians = [r["median"] for r in ok]
    ratio = max(medians) / min(medians)
    verdict = "PASS" if ratio >= 5.0 else "FAIL"
    print(f"[gate3] max/min ratio = {ratio:.2f}x  (target >= 5.0)  verdict: {verdict}",
          flush=True)

    report = {
        "label": "gate3_perf_differentiation",
        "backend": MEASUREMENT_BACKEND,
        "node": os.uname().nodename,
        "results": results,
        "ratio": round(ratio, 2),
        "min_median": min(medians),
        "max_median": max(medians),
        "verdict": verdict,
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
    if verdict == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
