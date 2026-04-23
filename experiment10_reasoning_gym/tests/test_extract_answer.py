"""Tests for extract_answer — the response parser.

Covers edge cases that a base LLM can plausibly hit:
- trailing whitespace / newlines
- nested tags (we take the first match)
- upper-case / mixed-case tags
- empty <answer></answer>
- ragged truncation (closing tag missing)
- multiple answer blocks (use first)
"""
from __future__ import annotations

from experiment10_reasoning_gym.src.reward_manager import extract_answer


def test_simple_tagged():
    assert extract_answer("<answer>hello</answer>") == "hello"


def test_tags_with_leading_trailing_whitespace():
    assert extract_answer("<answer>  hi  </answer>") == "hi"


def test_tags_multiline_with_newlines():
    assert extract_answer("<answer>\nline1\nline2\n</answer>") == "line1\nline2"


def test_no_tags_returns_stripped_input():
    assert extract_answer("just text") == "just text"
    assert extract_answer("  spaced  ") == "spaced"


def test_empty_tags_returns_empty():
    assert extract_answer("<answer></answer>") == ""


def test_truncated_missing_close_tag_falls_back_to_full():
    # Regex won't match without a close tag; we fall through to full-string strip.
    s = "<answer>partial but truncated"
    out = extract_answer(s)
    assert "<answer>" in out  # full fallback preserved


def test_multiple_answer_blocks_uses_first():
    s = "<answer>first</answer> then <answer>second</answer>"
    assert extract_answer(s) == "first"


def test_case_insensitive_tags():
    assert extract_answer("<ANSWER>X</ANSWER>") == "X"
    assert extract_answer("<Answer>Y</Answer>") == "Y"


def test_none_input():
    assert extract_answer(None) == ""


def test_empty_string_input():
    assert extract_answer("") == ""


def test_tag_content_with_commas_preserved():
    assert extract_answer("<answer>a, b, c</answer>") == "a, b, c"


def test_chain_of_thought_before_answer():
    s = "Let me think... first X then Y. So the answer is:\n<answer>final</answer>\nDone."
    assert extract_answer(s) == "final"
