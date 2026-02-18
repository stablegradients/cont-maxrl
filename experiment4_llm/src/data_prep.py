from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_DIFFICULTIES = ("introductory", "interview")


def normalize_difficulty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        mapping = {
            0: "introductory",
            1: "interview",
            2: "competition",
        }
        return mapping.get(int(value), str(value).strip().lower())
    return str(value).strip().lower()


def parse_input_output_field(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        data = dict(raw)
    elif isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {"inputs": [], "outputs": [], "fn_name": None}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # APPS occasionally contains Python-literal style payloads.
            parsed = ast.literal_eval(raw)
            if not isinstance(parsed, Mapping):
                raise ValueError("input_output payload must decode to a mapping")
            data = dict(parsed)
    else:
        raise TypeError(f"Unsupported input_output type: {type(raw)!r}")

    raw_inputs = data.get("inputs", data.get("input", []))
    raw_outputs = data.get("outputs", data.get("output", []))
    raw_fn_name = data.get("fn_name", data.get("function_name", data.get("entry_point")))

    inputs = list(raw_inputs) if isinstance(raw_inputs, list) else [raw_inputs]
    outputs = list(raw_outputs) if isinstance(raw_outputs, list) else [raw_outputs]
    fn_name = str(raw_fn_name).strip() if raw_fn_name else None

    return {
        "inputs": inputs,
        "outputs": outputs,
        "fn_name": fn_name,
    }


def format_problem_prompt(record: Mapping[str, Any]) -> str:
    question = (record.get("question") or "").strip()
    starter_code = (record.get("starter_code") or "").rstrip()

    parts = [
        "You are solving a Python coding task.",
        "Return only valid Python code.",
        "",
        "Problem:",
        question,
    ]

    if starter_code:
        parts.extend(
            [
                "",
                "Starter code (optional):",
                "```python",
                starter_code,
                "```",
            ]
        )

    parts.extend(
        [
            "",
            "Write a complete Python solution now.",
        ]
    )

    return "\n".join(parts).strip() + "\n"


def convert_apps_record_to_verl(
    record: Mapping[str, Any],
    min_test_cases: int = 5,
) -> dict[str, Any] | None:
    raw_input_output = record.get("input_output")
    if raw_input_output is None:
        return None

    try:
        parsed_tests = parse_input_output_field(raw_input_output)
    except Exception:
        return None

    inputs = parsed_tests["inputs"]
    outputs = parsed_tests["outputs"]
    fn_name = parsed_tests.get("fn_name")

    if len(inputs) != len(outputs):
        return None
    if len(inputs) < min_test_cases:
        return None

    reward_payload: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if fn_name:
        reward_payload["fn_name"] = fn_name

    return {
        "problem_id": record.get("problem_id", record.get("id", "")),
        "difficulty": normalize_difficulty(record.get("difficulty", "")),
        "prompt": format_problem_prompt(record),
        "reward_model": {
            "ground_truth": json.dumps(reward_payload, ensure_ascii=False),
        },
    }


def filter_apps_records(
    records: Iterable[Mapping[str, Any]],
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES,
    min_test_cases: int = 5,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    allowed = {d.strip().lower() for d in difficulties}
    out: list[dict[str, Any]] = []

    for record in records:
        diff = normalize_difficulty(record.get("difficulty", ""))
        if allowed and diff not in allowed:
            continue

        converted = convert_apps_record_to_verl(record, min_test_cases=min_test_cases)
        if converted is None:
            continue

        out.append(converted)
        if max_records is not None and len(out) >= max_records:
            break

    return out


def load_apps_dataset_records(split: str = "train") -> list[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for APPS conversion. Install with: pip install datasets"
        ) from exc

    dataset = load_dataset("codeparrot/apps", split=split)
    return [dataset[idx] for idx in range(len(dataset))]


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert APPS to verl JSONL format.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train/test")
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=list(DEFAULT_DIFFICULTIES),
        help="Difficulty levels to keep",
    )
    parser.add_argument("--min-test-cases", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output JSONL path for converted records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_apps_dataset_records(split=args.split)
    converted = filter_apps_records(
        records=records,
        difficulties=tuple(args.difficulties),
        min_test_cases=args.min_test_cases,
        max_records=args.max_records,
    )
    write_jsonl(converted, path=args.output_path)
    print(
        f"Wrote {len(converted)} records to {args.output_path} "
        f"(split={args.split}, difficulties={args.difficulties})"
    )


if __name__ == "__main__":
    main()
