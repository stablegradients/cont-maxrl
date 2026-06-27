"""
Build PIE training + validation parquets for verl.

For each PIE JSONL split, render the (src_code, problem_id) into a verl-
compatible record and write a parquet. Records whose src_code failed to
compile or pass at -O0 are dropped (no usable instruction-count denominator
for the log2-speedup reward); the per-record src_instr_count is pulled from
the precomputed pickle at experiment11_code_optim/precompute/build_src_instr_table.py
and embedded inside the ground_truth JSON so the reward fn can read it
per-rollout without an extra disk hit.

Schema (matches experiment4_llm/src/data_prep.py for compatibility with
verl's reward_kwargs flow):

    data_source: str          # always "pie"
    prompt: list[dict]        # [{"role": "user", "content": <PROMPT_TEMPLATE>}]
    reward_model: dict        # {"ground_truth": json.dumps({"problem_id": ...,
                              #                              "split": ...,
                              #                              "src_instr_count": <int>})}

The prompt template anchors on "### Optimized Version:" so
`pie_reward_fn.extract_code` can parse model completions identically in
RL rollouts and SFT validation. Must match the SFT template in
`sft/cold_start_sft.py` (§11 prompt parity).

Usage:
    # First build the src_instr_count table (~13 min on one node):
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/build_src_instr_table.py

    # Then rebuild the parquets joining src_instr_count into ground_truth:
    PYTHONPATH=/projects/bgmo/sramasubramanian/cont-maxrl \\
        /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python \\
        experiment11_code_optim/precompute/build_pie_parquet.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

PIE_DATA_ROOT = Path("/work/nvme/bgmo/sramasubramanian/pie_data")
RAW_ROOT = PIE_DATA_ROOT / "raw"
DEFAULT_SRC_INSTR_PICKLE = PIE_DATA_ROOT / "src_instr_table_O0.pkl"
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


def load_src_instr_table(pickle_path: Path) -> dict[str, int | None]:
    """Load {record_key → src_instr_count|None} pickle produced by
    build_src_instr_table.py. Keys are md5(src_code + "::" + problem_id)."""
    if not pickle_path.is_file():
        raise FileNotFoundError(
            f"src_instr_count pickle not found at {pickle_path}. Run "
            f"experiment11_code_optim/precompute/build_src_instr_table.py "
            f"first (~13 min on one ghx4 node)."
        )
    with open(pickle_path, "rb") as f:
        table = pickle.load(f)
    n_with = sum(1 for v in table.values() if v is not None)
    print(f"[src_instr] loaded {len(table)} keys ({n_with} have instr_count, "
          f"{len(table) - n_with} are null and will drop their records)")
    return table


def record_key(src_code: str, problem_id: str) -> str:
    """Mirror build_src_instr_table.record_key — DO NOT diverge."""
    return hashlib.md5(f"{src_code}::{problem_id}".encode()).hexdigest()


def _format_record(src_code: str, problem_id: str, split: str,
                   src_instr_count: int) -> dict:
    return {
        "data_source": "pie",
        "prompt": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(src_code=src_code)},
        ],
        "reward_model": {
            "ground_truth": json.dumps({
                "problem_id": problem_id,
                "split": split,
                "src_instr_count": int(src_instr_count),
            }),
        },
    }


def build_split_records(
    jsonl_path: Path,
    split: str,
    src_instr: dict[str, int | None],
    tokenizer,
    max_prompt_tokens: int,
) -> tuple[list[dict], dict]:
    """Read every row, drop rows whose src_code has no usable instr count at
    -O0 AND drop prompts whose tokenized length exceeds `max_prompt_tokens`
    (pathological 100k-token outliers in the raw data). Returns verl records
    + stats dict."""
    kept = []
    n_total = 0
    n_missing_src = 0
    n_no_instr = 0       # src_code present but precompute marked it null
    n_unknown_key = 0    # src_code present but precompute pickle has no entry
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
            key = record_key(src, pid)
            if key not in src_instr:
                n_unknown_key += 1
                continue
            instr = src_instr[key]
            if instr is None or instr <= 0:
                n_no_instr += 1
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
            kept.append(_format_record(src, pid, split, instr))
    stats = {
        "n_total_rows": n_total,
        "n_kept": len(kept),
        "n_no_instr": n_no_instr,
        "n_unknown_key": n_unknown_key,
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
    ap.add_argument("--src_instr_pickle", default=str(DEFAULT_SRC_INSTR_PICKLE),
                    help="Pickle from build_src_instr_table.py; "
                         "{record_key: instr_count|None}.")
    ap.add_argument("--val_subsample", type=int, default=200,
                    help="Subsample val parquet to N rows for in-situ eval "
                         "(~200 val problems for GO/NO-GO).")
    ap.add_argument("--val_seed", type=int, default=0,
                    help="Seed for the val subsample so the GO/NO-GO set is "
                         "stable across runs.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B",
                    help="Tokenizer used for prompt-length filtering. Default "
                         "is Qwen3-1.7B (the Instruct/chat variant — Qwen3's "
                         "bare repo name IS the instruct model; there is no "
                         "-Instruct suffix on HF) so the count matches what "
                         "verl renders at training time on the chat template.")
    ap.add_argument("--max_prompt_tokens", type=int, default=6144,
                    help="Drop rows whose tokenized prompt exceeds this. The "
                         "raw PIE JSONLs contain pathological 100k+ token "
                         "rows (>p99.9) that bloat verl's overlong filter and "
                         "skew batching. 6144 covers >99.9 percent of real "
                         "data while excising the noise.")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    src_instr = load_src_instr_table(Path(args.src_instr_pickle))

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
            in_path, split, src_instr, tokenizer, args.max_prompt_tokens
        )
        write_parquet(records, out_path)
        summary[split] = {**stats, "out_path": str(out_path)}
        print(
            f"[{split}] kept {stats['n_kept']:>6} / {stats['n_total_rows']:>6} rows "
            f"(no_instr={stats['n_no_instr']}, unknown_key={stats['n_unknown_key']}, "
            f"missing_src={stats['n_missing_src']}, "
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
            "schema_version": 2,  # bumped: ground_truth now carries src_instr_count
            "src_instr_pickle": str(args.src_instr_pickle),
            "tokenizer": args.tokenizer,
            "max_prompt_tokens": args.max_prompt_tokens,
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
