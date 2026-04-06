"""
Prepare datasets for verl training format (parquet).

Supported datasets:
  - apps        : codeparrot/apps (Introductory + Interview)
  - taco        : BAAI/TACO (~25K problems, same format as APPS)
  - code_contests: deepmind/code_contests (competitive programming)
  - livecodebench: livecodebench/code_generation_lite (eval-only, ~880 problems)

Output format (parquet with columns):
  - data_source: str (dataset name)
  - prompt: list[dict] (chat messages)
  - reward_model: dict with "ground_truth" key

Usage:
    python experiment4_llm/src/data_prep.py --dataset apps --output_dir data/
    python experiment4_llm/src/data_prep.py --dataset all  --output_dir data/
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

# Some APPS/TACO test cases contain very large integers (9000+ digits).
# Python 3.10+ limits int-to-str conversion to 4300 digits by default.
sys.set_int_max_str_digits(0)

import pandas as pd
from datasets import load_dataset


# ── Shared constants ──

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

APPS_DIFFICULTIES_ALL = ["introductory", "interview", "competition"]

TACO_DIFFICULTIES_DEFAULT = ["EASY", "MEDIUM", "MEDIUM_HARD"]


# ── Shared helpers ──

def format_prompt(question: str, fn_name: str | None, starter_code: str | None) -> str:
    """Format a problem into a prompt string.

    Uses the official LiveCodeBench prompt template so that evaluation
    results are comparable with published baselines.
    """
    if starter_code and starter_code.strip():
        # Call-based / functional (LeetCode-style): show starter code
        return (
            f"### Question:\n{question.strip()}\n\n"
            f"### Format: You will use the following starter code to write the "
            f"solution to the problem and enclose your code within delimiters.\n"
            f"```python\n{starter_code.strip()}\n```\n\n"
            f"### Answer: (use the provided format with backticks)\n"
        )
    else:
        # stdin/stdout (competitive programming)
        return (
            f"### Question:\n{question.strip()}\n\n"
            f"### Format: Read the inputs from stdin solve the problem and write "
            f"the answer to stdout (do not directly test on the sample inputs). "
            f"Enclose your code within delimiters as follows. Ensure that when the "
            f"python program runs, it reads the inputs, runs the algorithm and "
            f"writes output to STDOUT.\n"
            f"```python\n# YOUR CODE HERE\n```\n\n"
            f"### Answer: (use the provided format with backticks)\n"
        )


def make_record(
    data_source: str,
    prompt_text: str,
    test_cases: dict,
    max_tests: int = 50,
    max_gt_chars: int = 10_000_000,
) -> dict | None:
    """Build the standard verl record dict, capping test cases.

    Adaptively reduces the number of test cases for problems with large I/O
    to stay under max_gt_chars. Returns None only if even a single test case
    exceeds the limit.
    """
    extra_keys = {k: v for k, v in test_cases.items() if k not in ("inputs", "outputs")}

    # Try progressively fewer tests until ground_truth fits
    for cap in [max_tests, 20, 10, 5, 3, 1]:
        capped = {
            "inputs": test_cases["inputs"][:cap],
            "outputs": test_cases["outputs"][:cap],
            **extra_keys,
        }
        gt_json = json.dumps(capped)
        if len(gt_json) <= max_gt_chars:
            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "reward_model": {
                    "ground_truth": gt_json,
                },
            }

    return None


def write_parquet(records: list[dict], output_path: Path) -> None:
    """Write records list to a parquet file."""
    df = pd.DataFrame(records)
    df.to_parquet(output_path)


def parse_test_cases(input_output_str: str) -> dict | None:
    """Parse the input_output JSON field (APPS/TACO format). Returns None if invalid."""
    try:
        data = json.loads(input_output_str)
    except (json.JSONDecodeError, TypeError):
        return None

    inputs = data.get("inputs", [])
    outputs = data.get("outputs", [])

    if not inputs or not outputs or len(inputs) != len(outputs):
        return None

    # Normalize: ensure all are strings.
    # Use json.dumps for non-strings so the sandbox can json.loads them back.
    # str() would produce Python repr (single quotes) which isn't valid JSON.
    inputs = [x if isinstance(x, str) else json.dumps(x) for x in inputs]
    outputs = [x if isinstance(x, str) else json.dumps(x) for x in outputs]

    result = {"inputs": inputs, "outputs": outputs}

    fn_name = data.get("fn_name")
    if fn_name:
        result["fn_name"] = fn_name

    return result


# ── APPS ──

def prepare_apps(
    output_dir: str,
    min_tests: int = 3,
    max_tests: int = 50,
    difficulties: list[str] | None = None,
    hf_cache_dir: str | None = None,
) -> dict[str, int]:
    """Download and prepare APPS dataset as parquet files."""
    if difficulties is None:
        difficulties = ["introductory", "interview"]

    os.makedirs(output_dir, exist_ok=True)
    difficulty_set = set(difficulties)

    stats = {}

    for split in ["train", "test"]:
        print(f"Loading APPS {split} split...")
        ds = load_dataset(
            "codeparrot/apps",
            split=split,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
        )

        records = []
        skipped_difficulty = 0
        skipped_tests = 0
        skipped_parse = 0

        for example in ds:
            difficulty = str(example.get("difficulty", ""))
            if difficulty not in difficulty_set:
                skipped_difficulty += 1
                continue

            test_cases = parse_test_cases(example.get("input_output", ""))
            if test_cases is None:
                skipped_parse += 1
                continue

            if len(test_cases["inputs"]) < min_tests:
                skipped_tests += 1
                continue

            question = example.get("question", "")
            if not question or not question.strip():
                continue

            fn_name = test_cases.get("fn_name")
            starter_code = example.get("starter_code", "")
            # If call-based (fn_name) but no starter_code, synthesize a stub
            # so format_prompt uses the call-based template.
            if fn_name and not (starter_code and starter_code.strip()):
                starter_code = f"def {fn_name}():\n    pass"
            prompt_text = format_prompt(question, fn_name, starter_code)
            record = make_record("apps", prompt_text, test_cases, max_tests)
            if record is not None:
                records.append(record)

        output_path = Path(output_dir) / f"apps_{split}.parquet"
        write_parquet(records, output_path)

        stats[split] = len(records)
        print(
            f"  {split}: {len(records)} problems written to {output_path}"
            f" (skipped: {skipped_difficulty} difficulty, {skipped_tests} too few tests,"
            f" {skipped_parse} parse errors)"
        )

    return stats


# ── TACO-verified ──

def prepare_taco(
    output_dir: str,
    min_tests: int = 5,
    max_tests: int = 50,
    difficulties: list[str] | None = None,
    hf_cache_dir: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Download and prepare TACO-verified dataset as parquet.

    likaixin/TACO-verified is a cleaned subset of BAAI/TACO (~12.9K problems).
    Solutions verified against all test cases, duplicates removed.
    Only has a train split on HF, so we split into train/val ourselves.
    """
    import random

    if difficulties is None:
        difficulties = TACO_DIFFICULTIES_DEFAULT

    os.makedirs(output_dir, exist_ok=True)
    difficulty_set = set(difficulties)

    print("Loading TACO-verified train split...")
    ds = load_dataset(
        "likaixin/TACO-verified",
        split="train",
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )

    records = []
    skipped_difficulty = 0
    skipped_tests = 0
    skipped_parse = 0

    for example in ds:
        difficulty = example.get("difficulty", "")
        if difficulty not in difficulty_set:
            skipped_difficulty += 1
            continue

        test_cases = parse_test_cases(example.get("input_output", ""))
        if test_cases is None:
            skipped_parse += 1
            continue

        if len(test_cases["inputs"]) < min_tests:
            skipped_tests += 1
            continue

        question = example.get("question", "")
        if not question or not question.strip():
            continue

        fn_name = test_cases.get("fn_name")
        starter_code = example.get("starter_code", "")
        # If call-based (fn_name) but no starter_code, synthesize a stub
        # so format_prompt uses the call-based template.
        if fn_name and not (starter_code and starter_code.strip()):
            starter_code = f"def {fn_name}():\n    pass"
        prompt_text = format_prompt(question, fn_name, starter_code)
        record = make_record("taco", prompt_text, test_cases, max_tests)
        if record is not None:
            records.append(record)

    # Deduplicate by prompt text
    seen_prompts = set()
    unique_records = []
    for r in records:
        prompt_key = str(r["prompt"])
        if prompt_key not in seen_prompts:
            seen_prompts.add(prompt_key)
            unique_records.append(r)
    n_dedup = len(records) - len(unique_records)
    if n_dedup:
        print(f"  Removed {n_dedup} duplicate prompts ({len(records)} -> {len(unique_records)})")
    records = unique_records

    # Split into train/val
    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_fraction))
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_path = Path(output_dir) / "taco_train.parquet"
    val_path = Path(output_dir) / "taco_val.parquet"
    write_parquet(train_records, train_path)
    write_parquet(val_records, val_path)

    stats = {"train": len(train_records), "val": len(val_records)}
    print(
        f"  train: {len(train_records)} problems written to {train_path}"
        f"\n  val: {len(val_records)} problems written to {val_path}"
        f"\n  (skipped: {skipped_difficulty} difficulty, {skipped_tests} too few tests,"
        f" {skipped_parse} parse errors)"
    )

    return stats


# ── CodeContests ──

def _merge_code_contests_tests(example: dict) -> dict | None:
    """Merge public_tests, private_tests, generated_tests into unified format.

    CodeContests test fields are dicts: {"input": [str, ...], "output": [str, ...]}.
    All problems are stdin/stdout (no fn_name).
    """
    inputs = []
    outputs = []

    for tier in ["public_tests", "private_tests", "generated_tests"]:
        tests = example.get(tier)
        if tests is None:
            continue
        tier_inputs = tests.get("input", [])
        tier_outputs = tests.get("output", [])
        n = min(len(tier_inputs), len(tier_outputs))
        for i in range(n):
            inputs.append(str(tier_inputs[i]))
            outputs.append(str(tier_outputs[i]))

    if not inputs:
        return None

    return {"inputs": inputs, "outputs": outputs}


def prepare_code_contests(
    output_dir: str,
    min_tests: int = 3,
    max_tests: int = 50,
    hf_cache_dir: str | None = None,
) -> dict[str, int]:
    """Download and prepare CodeContests dataset as parquet files.

    deepmind/code_contests has train/valid/test splits.
    We use train for training and test for evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map our output names to HF split names
    split_map = {"train": "train", "test": "test"}
    stats = {}

    for out_name, hf_split in split_map.items():
        print(f"Loading CodeContests {hf_split} split...")
        ds = load_dataset(
            "deepmind/code_contests",
            split=hf_split,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
        )

        records = []
        skipped_tests = 0
        skipped_parse = 0

        for example in ds:
            test_cases = _merge_code_contests_tests(example)
            if test_cases is None:
                skipped_parse += 1
                continue

            if len(test_cases["inputs"]) < min_tests:
                skipped_tests += 1
                continue

            description = example.get("description", "")
            if not description or not description.strip():
                continue

            # All CodeContests problems are stdin/stdout
            prompt_text = format_prompt(description, fn_name=None, starter_code=None)
            record = make_record("code_contests", prompt_text, test_cases, max_tests)
            if record is not None:
                records.append(record)

        # Deduplicate by prompt text
        seen_prompts = set()
        unique_records = []
        for r in records:
            prompt_key = str(r["prompt"])
            if prompt_key not in seen_prompts:
                seen_prompts.add(prompt_key)
                unique_records.append(r)
        n_dedup = len(records) - len(unique_records)
        if n_dedup:
            print(f"  Removed {n_dedup} duplicate prompts ({len(records)} -> {len(unique_records)})")
        records = unique_records

        output_path = Path(output_dir) / f"code_contests_{out_name}.parquet"
        write_parquet(records, output_path)

        stats[out_name] = len(records)
        print(
            f"  {out_name}: {len(records)} problems written to {output_path}"
            f" (skipped: {skipped_tests} too few tests, {skipped_parse} parse errors)"
        )

    return stats


# ── LiveCodeBench ──

def _decompress_livecodebench_tests(raw: str) -> list[dict] | None:
    """Decode LiveCodeBench private_test_cases field.

    Encoding: base64 -> zlib -> pickle -> JSON string -> list[dict].
    Each dict has keys: input, output, testtype.
    Falls back to plain JSON if the string is not encoded.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        decoded = base64.b64decode(raw.encode("utf-8"))
        decompressed = zlib.decompress(decoded)
        unpickled = pickle.loads(decompressed)  # noqa: S301
        return json.loads(unpickled)
    except Exception:
        return None


def _parse_lcb_functional_input(raw: str) -> list:
    """Parse LiveCodeBench functional input into an args list.

    LCB functional inputs are newline-separated JSON/Python literals,
    one per function argument. E.g.:
        '"leetscode"\\n["leet", "code"]'  ->  ["leetscode", ["leet", "code"]]
        '[12, 9]\\n1'                     ->  [[12, 9], 1]
        '[[7, 2, 1], [6, 4, 2]]'          ->  [[[7, 2, 1], [6, 4, 2]]]
    """
    args = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            # Fallback: try Python literal eval (e.g. True/False/None)
            try:
                import ast
                args.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                args.append(line)  # bare string
    return args


def _json_encode_value(val) -> str:
    """Encode a return value as JSON string for ground_truth comparison."""
    if isinstance(val, str):
        try:
            # If it's already a JSON-encoded value, keep it
            json.loads(val)
            return val
        except (json.JSONDecodeError, ValueError):
            # Bare string — JSON-encode it
            return json.dumps(val)
    return json.dumps(val)


def prepare_livecodebench(
    output_dir: str,
    min_tests: int = 1,
    max_tests: int = 50,
    hf_cache_dir: str | None = None,
    version_tag: str = "release_v5",
) -> dict[str, int]:
    """Download and prepare LiveCodeBench dataset as parquet (eval-only).

    livecodebench/code_generation_lite has only a test split.
    Problems may be stdin or functional style (determined by testtype field).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading LiveCodeBench ({version_tag})...")
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        version_tag=version_tag,
        split="test",
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )

    records = []
    skipped_tests = 0
    skipped_parse = 0

    for example in ds:
        # Decode private test cases
        raw_private = example.get("private_test_cases", "")
        test_list = _decompress_livecodebench_tests(raw_private)
        if test_list is None or not test_list:
            skipped_parse += 1
            continue

        # Also include public test cases
        raw_public = example.get("public_test_cases", "")
        try:
            public_list = json.loads(raw_public) if raw_public else []
        except (json.JSONDecodeError, TypeError):
            public_list = []

        all_tests = public_list + test_list

        # Determine if functional or stdin
        # Use first test's testtype as representative
        testtype = all_tests[0].get("testtype", "stdin")

        # Parse metadata for fn_name
        metadata_str = example.get("metadata", "{}")
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        fn_name = None
        starter_code = example.get("starter_code", "")
        if testtype == "functional":
            fn_name = metadata.get("func_name")

        # Normalize to our format
        inputs = []
        outputs = []
        for tc in all_tests:
            inp = tc.get("input", "")
            out = tc.get("output") or tc.get("expected_output", "")

            if fn_name is not None:
                # Functional: inputs are newline-separated JSON literals
                # e.g. '"leetscode"\n["leet", "code"]' -> two args
                # Convert to a JSON args list: '["leetscode", ["leet", "code"]]'
                args = _parse_lcb_functional_input(str(inp))
                inputs.append(json.dumps(args))
                # Output is a single return value — wrap as JSON
                outputs.append(_json_encode_value(out))
            else:
                inputs.append(str(inp))
                outputs.append(str(out))

        if len(inputs) < min_tests:
            skipped_tests += 1
            continue

        question = example.get("question_content", "")
        if not question or not question.strip():
            continue

        prompt_text = format_prompt(question, fn_name, starter_code)

        test_cases = {"inputs": inputs, "outputs": outputs}
        if fn_name:
            test_cases["fn_name"] = fn_name

        record = make_record("livecodebench", prompt_text, test_cases, max_tests)
        if record is not None:
            records.append(record)

    # Use version tag in filename for clarity
    if version_tag == "release_v5":
        fname = "livecodebench_test.parquet"
    else:
        fname = f"livecodebench_{version_tag}_test.parquet"
    output_path = Path(output_dir) / fname
    write_parquet(records, output_path)

    stats = {"test": len(records)}
    print(
        f"  test: {len(records)} problems written to {output_path}"
        f" (skipped: {skipped_tests} too few tests, {skipped_parse} parse errors)"
    )

    return stats


# ── DeepCoder (minus LCB) ──

_DEEPCODER_PROBLEM_PREFIX = (
    "Solve the following coding problem using the programming language python:\n\n"
)


def _strip_deepcoder_prefix(problem: str) -> str:
    """Strip the DeepCoder-specific prompt prefix to get the raw problem statement."""
    if problem.startswith(_DEEPCODER_PROBLEM_PREFIX):
        return problem[len(_DEEPCODER_PROBLEM_PREFIX):]
    return problem


def prepare_deepcoder(
    output_dir: str,
    min_tests: int = 3,
    max_tests: int = 50,
    hf_cache_dir: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    deduplicate: bool = False,
) -> dict[str, int]:
    """Download and prepare DeepCoder dataset (minus LCB subset) as parquet.

    Uses taco + primeintellect configs from agentica-org/DeepCoder-Preview-Dataset.
    The lcbv5 config is excluded entirely to avoid contamination with LiveCodeBench eval.
    """
    import random

    os.makedirs(output_dir, exist_ok=True)

    records = []
    skipped_tests = 0
    skipped_parse = 0

    # ── TACO config: tests are {"inputs": [...], "outputs": [...]} ──
    print("Loading DeepCoder taco config...")
    ds_taco = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "taco",
        split="train",
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )

    for example in ds_taco:
        test_cases = parse_test_cases(example.get("tests", ""))
        if test_cases is None:
            skipped_parse += 1
            continue

        if len(test_cases["inputs"]) < min_tests:
            skipped_tests += 1
            continue

        question = example.get("problem", "")
        if not question or not question.strip():
            continue

        fn_name = test_cases.get("fn_name")
        # DeepCoder taco has no starter_code field. For call-based problems
        # (fn_name set), synthesize a minimal stub so format_prompt uses the
        # call-based template and the model sees the expected function name.
        # The problem text usually shows usage examples with the function name,
        # so the model can infer the proper signature.
        starter_code = None
        if fn_name:
            starter_code = f"def {fn_name}():\n    pass"
        prompt_text = format_prompt(question, fn_name, starter_code)
        record = make_record("deepcoder", prompt_text, test_cases, max_tests)
        if record is not None:
            records.append(record)

    n_taco = len(records)
    print(f"  taco: {n_taco} problems")

    # ── PrimeIntellect config: tests are [{"type": "stdin_stdout", "input": ..., "output": ...}] ──
    print("Loading DeepCoder primeintellect config...")
    ds_pi = load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset",
        "primeintellect",
        split="train",
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )

    n_before_pi = len(records)
    for example in ds_pi:
        raw_tests = example.get("tests", "")
        try:
            test_list = json.loads(raw_tests)
        except (json.JSONDecodeError, TypeError):
            skipped_parse += 1
            continue

        if not isinstance(test_list, list) or not test_list:
            skipped_parse += 1
            continue

        inputs = []
        outputs = []
        for tc in test_list:
            inp = tc.get("input", "")
            out = tc.get("output") or tc.get("expected_output", "")
            inputs.append(str(inp))
            outputs.append(str(out))

        if len(inputs) < min_tests:
            skipped_tests += 1
            continue

        question = _strip_deepcoder_prefix(example.get("problem", ""))
        if not question or not question.strip():
            continue

        # All PrimeIntellect problems are stdin/stdout
        prompt_text = format_prompt(question, fn_name=None, starter_code=None)
        test_cases = {"inputs": inputs, "outputs": outputs}
        record = make_record("deepcoder", prompt_text, test_cases, max_tests)
        if record is not None:
            records.append(record)

    n_pi = len(records) - n_before_pi
    print(f"  primeintellect: {n_pi} problems")

    # Optionally deduplicate by prompt text AND by ground truth (test cases).
    # The taco and primeintellect configs can describe the same problem
    # with different wording, so we dedup by both to be safe.
    # Disabled by default to keep parity with other papers using DeepCoder.
    if deduplicate:
        seen_prompts = set()
        seen_ground_truths = set()
        unique_records = []
        for r in records:
            prompt_key = json.dumps(r["prompt"], sort_keys=True)
            gt_key = r["reward_model"]["ground_truth"]
            if prompt_key not in seen_prompts and gt_key not in seen_ground_truths:
                seen_prompts.add(prompt_key)
                seen_ground_truths.add(gt_key)
                unique_records.append(r)
        n_dedup = len(records) - len(unique_records)
        if n_dedup:
            print(f"  Removed {n_dedup} duplicates by prompt+ground_truth ({len(records)} -> {len(unique_records)})")
        records = unique_records

    # Write single combined train set (no val split — matches original DeepCoder)
    rng = random.Random(seed)
    rng.shuffle(records)

    train_path = Path(output_dir) / "deepcoder_train.parquet"
    write_parquet(records, train_path)

    stats = {"train": len(records)}
    print(
        f"  train: {len(records)} problems written to {train_path}"
        f"\n  (skipped: {skipped_tests} too few tests, {skipped_parse} parse errors)"
        f"\n  NOTE: lcbv5 config excluded to avoid LiveCodeBench contamination"
    )

    return stats


# ── EvalPlus (HumanEval+ & MBPP+) ──


def prepare_evalplus(
    output_dir: str,
    max_tests: int = 50,
) -> dict[str, int]:
    """Download and prepare EvalPlus (HumanEval+ & MBPP+) as parquet (eval-only).

    Uses the evalplus package to get problems with base + plus test inputs,
    runs the canonical solution to compute expected outputs.
    """
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    os.makedirs(output_dir, exist_ok=True)

    skipped = 0
    per_dataset_records = {}

    for dataset_name, get_fn in [("humaneval", get_human_eval_plus),
                                  ("mbpp", get_mbpp_plus)]:
        problems = get_fn()
        print(f"  {dataset_name}+: {len(problems)} problems")
        records = []

        for task_id, problem in problems.items():
            entry_point = problem["entry_point"]
            prompt_code = problem["prompt"]
            canonical = problem["canonical_solution"]

            # Combine base + plus inputs
            base = problem.get("base_input", [])
            plus = problem.get("plus_input", [])
            if not isinstance(base, list):
                base = []
            if not isinstance(plus, list):
                plus = []
            all_inputs = base + plus
            if not all_inputs:
                skipped += 1
                continue

            # Run canonical solution to get expected outputs
            full_code = prompt_code + canonical
            namespace = {}
            try:
                exec(full_code, namespace)
            except Exception:
                skipped += 1
                continue

            fn = namespace.get(entry_point)
            if fn is None:
                skipped += 1
                continue

            inputs = []
            outputs = []
            for args in all_inputs:
                try:
                    result = fn(*args)
                    # Convert non-serializable types (e.g. set) to sorted list
                    if isinstance(result, set):
                        result = sorted(result)
                    inp_str = json.dumps(args)
                    out_str = json.dumps(result)
                    inputs.append(inp_str)
                    outputs.append(out_str)
                except Exception:
                    continue

            if len(inputs) < 1:
                skipped += 1
                continue

            # Format prompt — functional/call-based template with starter code
            prompt_text = format_prompt(
                f"Complete the following Python function:\n\n```python\n{prompt_code.strip()}\n```",
                fn_name=entry_point,
                starter_code=prompt_code,
            )

            test_cases = {"inputs": inputs, "outputs": outputs, "fn_name": entry_point}
            record = make_record("evalplus", prompt_text, test_cases, max_tests)
            if record is not None:
                records.append(record)

        per_dataset_records[dataset_name] = records

    # Write separate parquets for each subset
    stats = {}
    for dataset_name, records in per_dataset_records.items():
        output_path = Path(output_dir) / f"evalplus_{dataset_name}_test.parquet"
        write_parquet(records, output_path)
        stats[dataset_name] = len(records)
        print(f"  {dataset_name}+: {len(records)} problems written to {output_path}")

    print(f"  (skipped: {skipped})")

    return stats


# ── CLI ──

ALL_DATASETS = ["apps", "taco", "code_contests", "livecodebench", "deepcoder", "evalplus"]

PREPARE_FNS = {
    "apps": prepare_apps,
    "taco": prepare_taco,
    "code_contests": prepare_code_contests,
    "livecodebench": prepare_livecodebench,
    "deepcoder": prepare_deepcoder,
    "evalplus": prepare_evalplus,
}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for verl training/evaluation"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=ALL_DATASETS + ["all"],
        help="Which dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--min_tests", type=int, default=3,
        help="Minimum number of test cases per problem",
    )
    parser.add_argument(
        "--max_tests", type=int, default=50,
        help="Maximum test cases per problem (caps large test suites for speed)",
    )
    parser.add_argument(
        "--difficulties", nargs="+", default=None,
        help="Difficulty levels to include (apps: introductory/interview/competition, "
             "taco: EASY/MEDIUM/MEDIUM_HARD/HARD/VERY_HARD)",
    )
    parser.add_argument(
        "--hf_cache_dir", type=str, default=None,
        help="HuggingFace cache directory (default: HF_HOME env var)",
    )
    parser.add_argument(
        "--lcb_version", type=str, default="release_v5",
        help="LiveCodeBench version tag (default: release_v5)",
    )
    args = parser.parse_args()

    if args.hf_cache_dir is None:
        args.hf_cache_dir = os.environ.get("HF_HOME")

    datasets_to_prepare = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    all_stats = {}
    for ds_name in datasets_to_prepare:
        print(f"\n{'='*60}")
        print(f"Preparing: {ds_name}")
        print(f"{'='*60}")

        if ds_name == "apps":
            difficulties = args.difficulties or ["introductory", "interview"]
            stats = prepare_apps(
                output_dir=args.output_dir,
                min_tests=args.min_tests,
                max_tests=args.max_tests,
                difficulties=difficulties,
                hf_cache_dir=args.hf_cache_dir,
            )
        elif ds_name == "taco":
            difficulties = args.difficulties or TACO_DIFFICULTIES_DEFAULT
            stats = prepare_taco(
                output_dir=args.output_dir,
                min_tests=args.min_tests,
                max_tests=args.max_tests,
                difficulties=difficulties,
                hf_cache_dir=args.hf_cache_dir,
            )
        elif ds_name == "code_contests":
            stats = prepare_code_contests(
                output_dir=args.output_dir,
                min_tests=args.min_tests,
                max_tests=args.max_tests,
                hf_cache_dir=args.hf_cache_dir,
            )
        elif ds_name == "livecodebench":
            stats = prepare_livecodebench(
                output_dir=args.output_dir,
                min_tests=max(1, args.min_tests),
                max_tests=args.max_tests,
                hf_cache_dir=args.hf_cache_dir,
                version_tag=args.lcb_version,
            )
        elif ds_name == "deepcoder":
            stats = prepare_deepcoder(
                output_dir=args.output_dir,
                min_tests=args.min_tests,
                max_tests=args.max_tests,
                hf_cache_dir=args.hf_cache_dir,
            )
        elif ds_name == "evalplus":
            stats = prepare_evalplus(
                output_dir=args.output_dir,
                max_tests=args.max_tests,
            )
        else:
            print(f"Unknown dataset: {ds_name}")
            continue

        all_stats[ds_name] = stats

    print(f"\n{'='*60}")
    print("Summary:")
    for ds_name, stats in all_stats.items():
        print(f"  {ds_name}: {stats}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
