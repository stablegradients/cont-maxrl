"""
Build PIE training + validation parquets for verl (spec v9 §6 + §8).

For each PIE JSONL split, render the (src_code, problem_id) into a verl-
compatible record and write a parquet. Filters out problems that the
§14 step 6 reference-table precompute marked excluded (0 Accepted
programs pass the strict gate) — keeping them would waste RL rollouts
on flat-zero reward signal.

Schema (matches experiment4_llm/src/data_prep.py for compatibility with
verl's reward_kwargs flow):

    data_source: str          # always "pie"
    prompt: list[dict]        # [{"role": "user", "content": <PROMPT_TEMPLATE>}]
    reward_model: dict        # {"ground_truth": json.dumps({"problem_id": ..., "split": ...})}

The prompt template anchors on "### Optimized Version:" so
`pie_reward_fn.extract_code` can parse model completions identically in
RL rollouts and SFT validation. Must match the SFT template in
`sft/cold_start_sft.py` (§11 prompt parity).

Usage:
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/build_pie_parquet.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
RAW_ROOT = PIE_DATA_ROOT / "raw"
REFERENCE_TABLE_ROOT = PIE_DATA_ROOT / "reference_table"
DEFAULT_OUTPUT_ROOT = PIE_DATA_ROOT / "parquet"


# Must match `experiment11_code_optim/sft/cold_start_sft.py::PROMPT_TEMPLATE`
# (modulo the trailing completion). Anchored on "### Optimized Version:" so
# `pie_reward_fn.extract_code` parses completions the same way.
PROMPT_TEMPLATE = (
    "You are a C++ optimization expert. Below is a working C++ program. "
    "Rewrite it to run faster while producing identical output for all valid "
    "inputs. Keep the same input/output format and only change the algorithm/"
    "implementation.\n\n"
    "### Slow Version:\n"
    "```cpp\n{src_code}\n```\n\n"
    "### Optimized Version:\n"
)


def load_excluded_problem_ids(reference_table_root: Path) -> set[str]:
    """Read MANIFEST.json from the reference-table precompute. Return the
    set of "<split>/<pid>" strings that have ZERO passing reference programs.
    These are the problems the reward fn cannot grade meaningfully — drop
    them from RL training data.
    """
    manifest_path = reference_table_root / "MANIFEST.json"
    if not manifest_path.exists():
        print(
            f"WARN: {manifest_path} not found; no problems will be filtered. "
            f"Build the reference table via "
            f"experiment11_code_optim/precompute/build_reference_table.py first.",
            file=sys.stderr,
        )
        return set()
    with open(manifest_path) as f:
        m = json.load(f)
    return set(m.get("excluded", []))


def _format_record(src_code: str, problem_id: str, split: str) -> dict:
    return {
        "data_source": "pie",
        "prompt": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(src_code=src_code)},
        ],
        "reward_model": {
            "ground_truth": json.dumps({"problem_id": problem_id, "split": split}),
        },
    }


def build_split_records(
    jsonl_path: Path,
    split: str,
    excluded: set[str],
    tokenizer,
    max_prompt_tokens: int,
) -> tuple[list[dict], dict]:
    """Read every row, drop excluded `<split>/<pid>` entries AND prompts whose
    tokenized length exceeds `max_prompt_tokens` (pathological 100k-token
    outliers in the raw data). Returns verl records + stats dict."""
    kept = []
    n_total = 0
    n_excluded = 0
    n_missing_src = 0
    n_overlong = 0
    overlong_examples: list[tuple[str, int]] = []
    with open(jsonl_path) as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            pid = r.get("problem_id")
            src = r.get("src_code")
            if not pid or not src:
                n_missing_src += 1
                continue
            if f"{split}/{pid}" in excluded:
                n_excluded += 1
                continue
            # Filter on tokenized prompt length, not character count: that's the
            # number verl's max_prompt_length compares against at training time.
            prompt_text = PROMPT_TEMPLATE.format(src_code=src)
            n_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if n_tokens > max_prompt_tokens:
                n_overlong += 1
                if len(overlong_examples) < 5:
                    overlong_examples.append((pid, n_tokens))
                continue
            kept.append(_format_record(src, pid, split))
    stats = {
        "n_total_rows": n_total,
        "n_kept": len(kept),
        "n_excluded": n_excluded,
        "n_missing_src": n_missing_src,
        "n_overlong": n_overlong,
        "overlong_examples": overlong_examples,
    }
    return kept, stats


def subsample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic random subsample of up to n records from the list."""
    if len(records) <= n:
        return records
    return random.Random(seed).sample(records, n)


def write_parquet(records: list[dict], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--val_subsample", type=int, default=200,
                    help="Subsample val parquet to N rows for in-situ eval "
                         "(spec §8.5: ~200 val problems for GO/NO-GO).")
    ap.add_argument("--val_seed", type=int, default=0,
                    help="Seed for the val subsample so the GO/NO-GO set is "
                         "stable across runs.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base",
                    help="Tokenizer used for prompt-length filtering.")
    ap.add_argument("--max_prompt_tokens", type=int, default=6144,
                    help="Drop rows whose tokenized prompt exceeds this. The "
                         "raw PIE JSONLs contain pathological 100k+ token "
                         "rows (>p99.9) that bloat verl's overlong filter and "
                         "skew batching. 6144 covers >99.9 percent of real "
                         "data while excising the noise.")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    excluded = load_excluded_problem_ids(REFERENCE_TABLE_ROOT)
    print(f"[load] {len(excluded)} (split, problem_id) entries excluded from reference table")

    print(f"[tokenizer] loading {args.tokenizer}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    summary: dict[str, dict] = {}
    for split, in_jsonl, out_name in [
        ("train", "train.jsonl", "pie_train.parquet"),
        ("val", "val.jsonl", "pie_val_full.parquet"),
        ("test", "test.jsonl", "pie_test.parquet"),
    ]:
        in_path = RAW_ROOT / in_jsonl
        out_path = output_root / out_name
        records, stats = build_split_records(
            in_path, split, excluded, tokenizer, args.max_prompt_tokens
        )
        write_parquet(records, out_path)
        summary[split] = {**stats, "out_path": str(out_path)}
        print(
            f"[{split}] kept {stats['n_kept']:>6} / {stats['n_total_rows']:>6} rows "
            f"(excluded={stats['n_excluded']}, missing_src={stats['n_missing_src']}, "
            f"overlong>{args.max_prompt_tokens}={stats['n_overlong']}) -> {out_path}"
        )
        if stats["overlong_examples"]:
            print(f"  overlong examples: {stats['overlong_examples']}")

    # Subsample val to 200 rows for in-situ eval (§9 GO/NO-GO requires this fixed set).
    val_full = output_root / "pie_val_full.parquet"
    import pandas as pd
    df_val = pd.read_parquet(val_full)
    full_records = df_val.to_dict(orient="records")
    sub = subsample(full_records, args.val_subsample, args.val_seed)
    val_200 = output_root / f"pie_val_{args.val_subsample}.parquet"
    write_parquet(sub, val_200)
    print(
        f"[val_subsample] {len(sub)} rows (seed={args.val_seed}) -> {val_200}"
    )
    summary["val_subsample"] = {"n_rows": len(sub), "out_path": str(val_200)}

    # Manifest
    manifest_path = output_root / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "schema_version": 1,
            "excluded_count_in_reference_table": len(excluded),
            "splits": summary,
        }, f, indent=2)
    print(f"\n[done] MANIFEST -> {manifest_path}")

    # Wire into env
    print("\nNext: export these paths for the RL launch")
    print(f"  export PIE_TRAIN_PARQUET={output_root}/pie_train.parquet")
    print(f"  export PIE_VAL_PARQUET={output_root}/pie_val_{args.val_subsample}.parquet")

    # Epoch arithmetic — print suggested env values for spec §3.1a
    n_train = summary["train"]["n_kept"]
    import math
    spe = math.ceil(n_train / 256)
    total = 10 * spe
    eval_steps = [k * spe for k in range(1, 11)]
    print(f"\nSuggested epoch arithmetic (spec §3.1a, with N_train={n_train}):")
    print(f"  export PIE_STEPS_PER_EPOCH={spe}")
    print(f"  export PIE_TOTAL_STEPS={total}")
    print(f"  export PIE_EVAL_STEPS={eval_steps!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
