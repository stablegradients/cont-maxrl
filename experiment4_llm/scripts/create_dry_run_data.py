"""Create a tiny parquet dataset for dry-run testing."""
import json
import os

import pandas as pd

PROBLEMS = [
    {
        "question": "Read an integer n from stdin and print n * 2.",
        "inputs": ["5\n", "10\n", "0\n"],
        "outputs": ["10\n", "20\n", "0\n"],
    },
    {
        "question": "Read two integers a and b from stdin (one per line) and print their sum.",
        "inputs": ["3\n4\n", "10\n20\n", "0\n0\n"],
        "outputs": ["7\n", "30\n", "0\n"],
    },
    {
        "question": "Read a string from stdin and print it reversed.",
        "inputs": ["hello\n", "abc\n", "a\n"],
        "outputs": ["olleh\n", "cba\n", "a\n"],
    },
    {
        "question": "Read an integer n from stdin and print 'even' if it's even, 'odd' otherwise.",
        "inputs": ["4\n", "7\n", "0\n"],
        "outputs": ["even\n", "odd\n", "even\n"],
    },
    {
        "question": "Read an integer n from stdin and print the sum of integers from 1 to n.",
        "inputs": ["5\n", "10\n", "1\n"],
        "outputs": ["15\n", "55\n", "1\n"],
    },
    {
        "question": "Read a string from stdin and print its length.",
        "inputs": ["hello\n", "ab\n", "x\n"],
        "outputs": ["5\n", "2\n", "1\n"],
    },
    {
        "question": "Read an integer n from stdin and print 'yes' if it's positive, 'no' otherwise.",
        "inputs": ["5\n", "-3\n", "0\n"],
        "outputs": ["yes\n", "no\n", "no\n"],
    },
    {
        "question": "Read two integers from stdin and print the larger one.",
        "inputs": ["3\n5\n", "10\n2\n", "7\n7\n"],
        "outputs": ["5\n", "10\n", "7\n"],
    },
]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dry_run")
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for p in PROBLEMS:
        test_cases = {"inputs": p["inputs"], "outputs": p["outputs"]}
        prompt_text = (
            p["question"]
            + "\nYour solution should read from stdin and write to stdout."
            + "\nProvide your complete Python solution inside a code block:\n```python\n"
        )
        record = {
            "data_source": "apps",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            "reward_model": {"ground_truth": json.dumps(test_cases)},
        }
        records.append(record)

    df = pd.DataFrame(records)
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    df.to_parquet(train_path)
    df.to_parquet(test_path)
    print(f"Created {train_path} ({len(df)} problems)")
    print(f"Created {test_path} ({len(df)} problems)")


if __name__ == "__main__":
    main()
