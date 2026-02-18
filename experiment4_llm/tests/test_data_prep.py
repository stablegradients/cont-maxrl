from __future__ import annotations

import json
import unittest

from experiment4_llm.src.data_prep import (
    convert_apps_record_to_verl,
    filter_apps_records,
    normalize_difficulty,
    parse_input_output_field,
)


class DataPrepTests(unittest.TestCase):
    def test_normalize_difficulty(self) -> None:
        self.assertEqual(normalize_difficulty("Interview"), "interview")
        self.assertEqual(normalize_difficulty(0), "introductory")
        self.assertEqual(normalize_difficulty(1), "interview")
        self.assertEqual(normalize_difficulty(2), "competition")

    def test_parse_input_output_field(self) -> None:
        raw = json.dumps({"inputs": [[1], [2]], "outputs": ["2", "4"], "fn_name": "double"})
        parsed = parse_input_output_field(raw)
        self.assertEqual(parsed["fn_name"], "double")
        self.assertEqual(len(parsed["inputs"]), 2)

    def test_parse_input_output_field_python_literal_and_alt_keys(self) -> None:
        raw = "{'input': [[1], [2]], 'output': ['2', '4'], 'entry_point': 'double'}"
        parsed = parse_input_output_field(raw)
        self.assertEqual(parsed["fn_name"], "double")
        self.assertEqual(parsed["inputs"], [[1], [2]])
        self.assertEqual(parsed["outputs"], ["2", "4"])

    def test_convert_and_filter(self) -> None:
        records = [
            {
                "problem_id": "p1",
                "difficulty": "introductory",
                "question": "double x",
                "starter_code": "",
                "input_output": json.dumps(
                    {
                        "inputs": [[1], [2], [3], [4], [5]],
                        "outputs": ["2", "4", "6", "8", "10"],
                        "fn_name": "double",
                    }
                ),
            },
            {
                "problem_id": "p2",
                "difficulty": "competition",
                "question": "ignore",
                "starter_code": "",
                "input_output": json.dumps(
                    {
                        "inputs": [[1], [2], [3], [4], [5]],
                        "outputs": ["1", "2", "3", "4", "5"],
                    }
                ),
            },
        ]

        one = convert_apps_record_to_verl(records[0], min_test_cases=5)
        assert one is not None
        self.assertIn("prompt", one)
        self.assertIn("reward_model", one)

        filtered = filter_apps_records(records, difficulties=("introductory",), min_test_cases=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["problem_id"], "p1")


if __name__ == "__main__":
    unittest.main()
