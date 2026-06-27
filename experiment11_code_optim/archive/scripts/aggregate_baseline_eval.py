"""Aggregate per-(model, template, decoding) baseline eval results into one table.

Walks `experiment11_code_optim/docs/baseline_eval/<OUTPUT_TAG>/metrics.json`,
extracts pass_at_k and headline diagnostics, prints a markdown table sorted
to match the design doc's recommended layout.

Usage:
    python experiment11_code_optim/scripts/aggregate_baseline_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/projects/bgmo/sramasubramanian/cont-maxrl/"
            "experiment11_code_optim/docs/baseline_eval")

# Canonical row order matching the design doc.
ROW_ORDER = [
    ("base_0.6B_chat_n16_t1",      "Qwen3-0.6B-Base",   "chat",       16, 1.0),
    ("base_0.6B_oneshot_n16_t1",   "Qwen3-0.6B-Base",   "raw+1shot",  16, 1.0),
    ("base_0.6B_oneshot_n1_t0",    "Qwen3-0.6B-Base",   "raw+1shot",   1, 0.0),
    ("base_1.7B_chat_n16_t1",      "Qwen3-1.7B-Base",   "chat",       16, 1.0),
    ("base_1.7B_oneshot_n16_t1",   "Qwen3-1.7B-Base",   "raw+1shot",  16, 1.0),
    ("base_1.7B_oneshot_n1_t0",    "Qwen3-1.7B-Base",   "raw+1shot",   1, 0.0),
    ("base_4B_chat_n16_t1",        "Qwen3-4B-Base",     "chat",       16, 1.0),
    ("base_4B_oneshot_n16_t1",     "Qwen3-4B-Base",     "raw+1shot",  16, 1.0),
    ("base_4B_oneshot_n1_t0",      "Qwen3-4B-Base",     "raw+1shot",   1, 0.0),
    ("instruct_0.6B_chat_n16_t1",  "Qwen3-0.6B (Inst)", "chat",       16, 1.0),
    ("instruct_0.6B_chat_n1_t0",   "Qwen3-0.6B (Inst)", "chat",        1, 0.0),
    ("instruct_1.7B_chat_n16_t1",  "Qwen3-1.7B (Inst)", "chat",       16, 1.0),
    ("instruct_1.7B_chat_n1_t0",   "Qwen3-1.7B (Inst)", "chat",        1, 0.0),
    ("instruct_4B_chat_n16_t1",    "Qwen3-4B (Inst)",   "chat",       16, 1.0),
    ("instruct_4B_chat_n1_t0",     "Qwen3-4B (Inst)",   "chat",        1, 0.0),
]


def fmt(x) -> str:
    if x is None:
        return "  —  "
    return f"{x:.4f}"


def main():
    rows: list[tuple[str, str, str, int, float, dict | None]] = []
    missing = []
    for tag, model, tmpl, n, t in ROW_ORDER:
        mpath = ROOT / tag / "metrics.json"
        if not mpath.exists():
            missing.append(tag)
            rows.append((tag, model, tmpl, n, t, None))
            continue
        with open(mpath) as f:
            m = json.load(f)
        rows.append((tag, model, tmpl, n, t, m))

    # Per-row diagnostics
    print()
    print("# Qwen3 baseline zero-shot pass-rate on PIE val (200 problems)")
    print()
    print(f"| {'Model':22s} | {'Template':10s} | {'n':>2s} | {'T':>4s} | "
          f"{'pass@1':>7s} | {'pass@5':>7s} | {'pass@16':>7s} | "
          f"{'mean_pass_rate':>15s} | {'n_solved':>8s} |")
    print(f"|-{'-' * 22}-|-{'-' * 10}-|-{'-' * 2}-|-{'-' * 4}-|-"
          f"{'-' * 7}-|-{'-' * 7}-|-{'-' * 7}-|-{'-' * 15}-|-{'-' * 8}-|")
    for tag, model, tmpl, n, t, m in rows:
        if m is None:
            print(f"| {model:22s} | {tmpl:10s} | {n:>2d} | {t:>4.1f} | "
                  f"{'(missing)':>7s} | {'':>7s} | {'':>7s} | {'':>15s} | {'':>8s} |")
            continue
        pk = m.get("pass_at_k", {})
        p1 = pk.get("1")
        p5 = pk.get("5")
        p16 = pk.get("16")
        mpr = m.get("mean_pass_rate")
        nsolved = m.get("n_problems_solved")
        nprobs = m.get("n_problems")
        print(f"| {model:22s} | {tmpl:10s} | {n:>2d} | {t:>4.1f} | "
              f"{fmt(p1):>7s} | {fmt(p5):>7s} | {fmt(p16):>7s} | "
              f"{fmt(mpr):>15s} | {nsolved or 0:>4d}/{nprobs or 0:<3d} |")
    print()

    if missing:
        print(f"Missing/incomplete results ({len(missing)}):")
        for tag in missing:
            print(f"  - {tag}")

    # Quick summary by model class
    print()
    print("## Headline pass@16 by model class")
    print()
    print(f"| {'Model':22s} | {'best variant':22s} | {'pass@16':>7s} |")
    print(f"|-{'-' * 22}-|-{'-' * 22}-|-{'-' * 7}-|")
    by_model = {}
    for tag, model, tmpl, n, t, m in rows:
        if m is None or n != 16:
            continue
        p16 = m.get("pass_at_k", {}).get("16")
        if p16 is None:
            continue
        key = model
        if key not in by_model or p16 > by_model[key][1]:
            by_model[key] = (f"{tmpl} (T={t})", p16)
    for k, (variant, p16) in sorted(by_model.items()):
        print(f"| {k:22s} | {variant:22s} | {p16:>7.4f} |")
    print()


if __name__ == "__main__":
    main()
