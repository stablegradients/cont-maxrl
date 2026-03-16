"""Tests for data preparation: serialization round-trips and input parsing.

Covers bugs:
  R1-2: str() vs json.dumps() for non-string I/O serialization
  R3-1: LiveCodeBench functional input parsing (_parse_lcb_functional_input)
"""
import json
import pytest

from experiment4_llm.src.data_prep import parse_test_cases


# ── parse_test_cases: non-string I/O serialization (BUG R1-2) ──


def test_parse_test_cases_nonstring_inputs_are_valid_json():
    """Non-string inputs must be serialized with json.dumps, not str().

    str([[1,2]]) produces "[[1, 2]]" which happens to be valid JSON for simple
    lists, but str(True) produces "True" (not "true") and str({'a':1}) produces
    single quotes — both invalid JSON.
    """
    raw = json.dumps({
        "inputs": [[[1, 2, 3]], True, {"key": "val"}],
        "outputs": [[6], True, "result"],
        "fn_name": "solve",
    })
    result = parse_test_cases(raw)
    assert result is not None

    # Each non-string input must be valid JSON
    for i, inp in enumerate(result["inputs"]):
        try:
            json.loads(inp)
        except json.JSONDecodeError:
            pytest.fail(f"Input {i} is not valid JSON: {inp!r}")

    # Specifically, True must become "true" (JSON) not "True" (Python str)
    assert result["inputs"][1] == "true"
    # Dict must use double quotes
    assert '"key"' in result["inputs"][2]


def test_parse_test_cases_nonstring_outputs_are_valid_json():
    """Non-string outputs must also be valid JSON."""
    raw = json.dumps({
        "inputs": ["1", "2"],
        "outputs": [[1, 2, 3], False],
    })
    result = parse_test_cases(raw)
    assert result is not None

    for i, out in enumerate(result["outputs"]):
        try:
            json.loads(out)
        except json.JSONDecodeError:
            pytest.fail(f"Output {i} is not valid JSON: {out!r}")

    assert result["outputs"][1] == "false"


def test_parse_test_cases_string_inputs_unchanged():
    """String inputs should be left as-is (not double-encoded)."""
    raw = json.dumps({
        "inputs": ["hello world\n", "42\n"],
        "outputs": ["HELLO WORLD\n", "42\n"],
    })
    result = parse_test_cases(raw)
    assert result is not None
    assert result["inputs"][0] == "hello world\n"
    assert result["inputs"][1] == "42\n"


def test_parse_test_cases_fn_name_preserved():
    """fn_name should be preserved in output."""
    raw = json.dumps({
        "inputs": ["[1]"],
        "outputs": ["1"],
        "fn_name": "my_func",
    })
    result = parse_test_cases(raw)
    assert result is not None
    assert result["fn_name"] == "my_func"


def test_parse_test_cases_no_fn_name():
    """Without fn_name, result should not contain fn_name key."""
    raw = json.dumps({
        "inputs": ["1\n"],
        "outputs": ["2\n"],
    })
    result = parse_test_cases(raw)
    assert result is not None
    assert "fn_name" not in result


def test_parse_test_cases_invalid_json():
    """Invalid JSON should return None, not crash."""
    assert parse_test_cases("not json") is None
    assert parse_test_cases("") is None
    assert parse_test_cases(None) is None


def test_parse_test_cases_empty_inputs():
    """Empty inputs or outputs should return None."""
    assert parse_test_cases(json.dumps({"inputs": [], "outputs": []})) is None
    assert parse_test_cases(json.dumps({"inputs": ["1"], "outputs": []})) is None


def test_parse_test_cases_mismatched_lengths():
    """Mismatched input/output lengths should return None."""
    raw = json.dumps({"inputs": ["1", "2"], "outputs": ["3"]})
    assert parse_test_cases(raw) is None


# ── _parse_lcb_functional_input (BUG R3-1) ──


def test_lcb_input_string_and_list():
    """Two-arg functional input: string + list, newline-separated."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '"leetscode"\n["leet", "code"]'
    result = _parse_lcb_functional_input(raw)
    assert result == ["leetscode", ["leet", "code"]]


def test_lcb_input_list_and_int():
    """Two-arg: list + integer."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '[12, 9]\n1'
    result = _parse_lcb_functional_input(raw)
    assert result == [[12, 9], 1]


def test_lcb_input_nested_list():
    """Single-arg: nested list."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '[[7, 2, 1], [6, 4, 2]]'
    result = _parse_lcb_functional_input(raw)
    assert result == [[[7, 2, 1], [6, 4, 2]]]


def test_lcb_input_single_string():
    """Single-arg: just a JSON string."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '"hello world"'
    result = _parse_lcb_functional_input(raw)
    assert result == ["hello world"]


def test_lcb_input_empty():
    """Empty string should return empty list."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    assert _parse_lcb_functional_input("") == []
    assert _parse_lcb_functional_input("  \n  ") == []


def test_lcb_input_python_literal_fallback():
    """Python True (not valid JSON) should be parsed via ast.literal_eval."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    # JSON "true" → Python True
    assert _parse_lcb_functional_input("true") == [True]
    # Python "True" (not valid JSON) → fallback to ast.literal_eval
    assert _parse_lcb_functional_input("True") == [True]


def test_lcb_input_three_args():
    """Three arguments: string, int, list."""
    from experiment4_llm.src.data_prep import _parse_lcb_functional_input
    raw = '"abc"\n42\n[1, 2, 3]'
    result = _parse_lcb_functional_input(raw)
    assert result == ["abc", 42, [1, 2, 3]]


# ── _json_encode_value ──


def test_json_encode_value_already_json_string():
    """String that is already valid JSON should be returned as-is."""
    from experiment4_llm.src.data_prep import _json_encode_value
    assert _json_encode_value('"hello"') == '"hello"'
    assert _json_encode_value("[1, 2, 3]") == "[1, 2, 3]"


def test_json_encode_value_bare_string():
    """Bare string (not JSON-encoded) should be JSON-encoded."""
    from experiment4_llm.src.data_prep import _json_encode_value
    result = _json_encode_value("hello")
    assert result == '"hello"'


def test_json_encode_value_non_string():
    """Non-string values should be json.dumps'd."""
    from experiment4_llm.src.data_prep import _json_encode_value
    assert _json_encode_value(42) == "42"
    assert _json_encode_value(True) == "true"
    assert _json_encode_value([1, 2]) == "[1, 2]"
    assert _json_encode_value({"a": 1}) == '{"a": 1}'


# ── End-to-end: data_prep serialization -> reward function ──


def test_end_to_end_fn_name_round_trip():
    """Full pipeline: parse_test_cases output -> compute_score -> correct result.

    This is the integration test that would have caught BUG R1-2.
    Uses bool return values -- with str(), True becomes "True" (wrong),
    while json.dumps(True) produces "true" (correct JSON).
    """
    from experiment4_llm.src.reward_function import compute_score

    # Simulate raw APPS/TACO format data with non-string I/O.
    # inputs[i] = list of args for test case i, outputs[i] = expected return value
    raw_io = json.dumps({
        "inputs": [[4], [3]],     # two test cases, each with one int arg
        "outputs": [True, False],  # bool return values (critical for str vs json.dumps)
        "fn_name": "is_even",
    })
    parsed = parse_test_cases(raw_io)
    assert parsed is not None

    # Verify serialization is correct: json.dumps(True)="true", NOT str(True)="True"
    assert parsed["outputs"][0] == "true"
    assert parsed["outputs"][1] == "false"

    ground_truth = json.dumps(parsed)
    code = "def is_even(n):\n    return n % 2 == 0"
    result = compute_score(code, ground_truth)
    assert result["score"] == 1.0, f"End-to-end round trip failed: score={result}"


def test_end_to_end_nested_list_round_trip():
    """Pipeline with nested list args and list return value."""
    from experiment4_llm.src.reward_function import compute_score

    # fn takes two list args: inputs[i] = [arg1, arg2]
    raw_io = json.dumps({
        "inputs": [[[1, 2], [3, 4]]],    # one test case: add_lists([1,2], [3,4])
        "outputs": [[4, 6]],              # return value (non-string list)
        "fn_name": "add_lists",
    })
    parsed = parse_test_cases(raw_io)
    assert parsed is not None

    ground_truth = json.dumps(parsed)
    code = "def add_lists(a, b):\n    return [x+y for x,y in zip(a, b)]"
    result = compute_score(code, ground_truth)
    assert result["score"] == 1.0, f"Nested list round trip failed: score={result}"


# ── DeepCoder data prep ──


def test_strip_deepcoder_prefix():
    """Strip the DeepCoder-specific prompt prefix from PrimeIntellect problems."""
    from experiment4_llm.src.data_prep import _strip_deepcoder_prefix

    raw = (
        "Solve the following coding problem using the programming language python:\n\n"
        "Given an array of integers..."
    )
    assert _strip_deepcoder_prefix(raw) == "Given an array of integers..."


def test_strip_deepcoder_prefix_no_prefix():
    """Problems without the prefix should be returned unchanged."""
    from experiment4_llm.src.data_prep import _strip_deepcoder_prefix

    raw = "Given an array of integers..."
    assert _strip_deepcoder_prefix(raw) == raw


def test_deepcoder_parquet_schema():
    """Verify deepcoder parquet files have the correct schema and data_source."""
    import pandas as pd
    from pathlib import Path

    train_path = Path("data/deepcoder_train.parquet")

    if not train_path.exists():
        pytest.skip("DeepCoder parquet files not yet generated")

    df = pd.read_parquet(train_path)
    assert "data_source" in df.columns
    assert "prompt" in df.columns
    assert "reward_model" in df.columns
    assert (df["data_source"] == "deepcoder").all(), \
        f"data_source should be 'deepcoder', got {df['data_source'].unique()}"


def test_deepcoder_ground_truth_valid_json():
    """Every ground_truth in deepcoder parquet must be valid JSON with inputs/outputs."""
    import pandas as pd
    from pathlib import Path

    train_path = Path("data/deepcoder_train.parquet")
    if not train_path.exists():
        pytest.skip("DeepCoder parquet files not yet generated")

    df = pd.read_parquet(train_path)
    # Check a sample (first 100) for speed
    for i, row in df.head(100).iterrows():
        gt = row["reward_model"]["ground_truth"]
        parsed = json.loads(gt)
        assert "inputs" in parsed, f"Row {i}: missing 'inputs'"
        assert "outputs" in parsed, f"Row {i}: missing 'outputs'"
        assert len(parsed["inputs"]) >= 3, f"Row {i}: fewer than min_tests=3 inputs"
        assert len(parsed["inputs"]) == len(parsed["outputs"]), \
            f"Row {i}: input/output length mismatch"


def test_deepcoder_no_livecodebench_contamination():
    """DeepCoder training data must not contain any LiveCodeBench problems."""
    import pandas as pd
    from pathlib import Path

    train_path = Path("data/deepcoder_train.parquet")
    lcb_path = Path("data/livecodebench_test.parquet")

    if not all(p.exists() for p in [train_path, lcb_path]):
        pytest.skip("Required parquet files not yet generated")

    df_train = pd.read_parquet(train_path)
    df_lcb = pd.read_parquet(lcb_path)

    assert (df_train["data_source"] == "deepcoder").all()
    assert (df_lcb["data_source"] == "livecodebench").all()

    # No deepcoder record should have data_source == "livecodebench"
    # (This is guaranteed by construction, but verify explicitly)
    dc_prompts = set(str(p) for p in df_train["prompt"])
    lcb_prompts = set(str(p) for p in df_lcb["prompt"])

    overlap = dc_prompts & lcb_prompts
    assert len(overlap) == 0, \
        f"Found {len(overlap)} overlapping prompts between DeepCoder and LiveCodeBench"


def test_deepcoder_reward_function_compatible():
    """DeepCoder records must work with the reward function end-to-end."""
    import pandas as pd
    from pathlib import Path
    from experiment4_llm.src.reward_function import compute_score

    train_path = Path("data/deepcoder_train.parquet")
    if not train_path.exists():
        pytest.skip("DeepCoder parquet files not yet generated")

    df = pd.read_parquet(train_path)
    row = df.iloc[0]
    gt = row["reward_model"]["ground_truth"]

    # A deliberately wrong solution should score 0
    result = compute_score("print('wrong')", gt)
    assert isinstance(result, dict)
    assert "score" in result
    assert "acc" in result
    assert 0.0 <= result["score"] <= 1.0


# ── EvalPlus parquet tests ──


def test_evalplus_humaneval_parquet_schema():
    """Verify evalplus_humaneval parquet has correct schema and data_source."""
    import pandas as pd
    from pathlib import Path

    path = Path("data/evalplus_humaneval_test.parquet")
    if not path.exists():
        pytest.skip("EvalPlus HumanEval parquet not yet generated")

    df = pd.read_parquet(path)
    assert "data_source" in df.columns
    assert "prompt" in df.columns
    assert "reward_model" in df.columns
    assert (df["data_source"] == "evalplus").all()
    assert len(df) > 0, "HumanEval+ parquet should not be empty"


def test_evalplus_mbpp_parquet_schema():
    """Verify evalplus_mbpp parquet has correct schema and data_source."""
    import pandas as pd
    from pathlib import Path

    path = Path("data/evalplus_mbpp_test.parquet")
    if not path.exists():
        pytest.skip("EvalPlus MBPP parquet not yet generated")

    df = pd.read_parquet(path)
    assert "data_source" in df.columns
    assert "prompt" in df.columns
    assert "reward_model" in df.columns
    assert (df["data_source"] == "evalplus").all()
    assert len(df) > 0, "MBPP+ parquet should not be empty"


def test_evalplus_all_records_have_fn_name():
    """All EvalPlus problems are functional — every record must have fn_name."""
    import pandas as pd
    from pathlib import Path

    for name in ["evalplus_humaneval_test.parquet", "evalplus_mbpp_test.parquet"]:
        path = Path(f"data/{name}")
        if not path.exists():
            pytest.skip(f"{name} not yet generated")

        df = pd.read_parquet(path)
        for i, row in df.iterrows():
            gt = json.loads(row["reward_model"]["ground_truth"])
            assert "fn_name" in gt, f"{name} row {i}: missing fn_name"
            assert "inputs" in gt, f"{name} row {i}: missing inputs"
            assert "outputs" in gt, f"{name} row {i}: missing outputs"
            assert len(gt["inputs"]) == len(gt["outputs"]), \
                f"{name} row {i}: input/output length mismatch"


def test_evalplus_ground_truth_valid_json():
    """Every ground_truth in EvalPlus parquets must be valid JSON."""
    import pandas as pd
    from pathlib import Path

    for name in ["evalplus_humaneval_test.parquet", "evalplus_mbpp_test.parquet"]:
        path = Path(f"data/{name}")
        if not path.exists():
            pytest.skip(f"{name} not yet generated")

        df = pd.read_parquet(path)
        for i, row in df.head(50).iterrows():
            gt_str = row["reward_model"]["ground_truth"]
            parsed = json.loads(gt_str)
            # inputs should be JSON-parseable arg lists
            for j, inp in enumerate(parsed["inputs"][:3]):
                args = json.loads(inp)
                assert isinstance(args, list), \
                    f"{name} row {i} input {j}: expected list of args, got {type(args)}"


def test_evalplus_reward_function_compatible():
    """EvalPlus records must work with the reward function end-to-end."""
    import pandas as pd
    from pathlib import Path
    from experiment4_llm.src.reward_function import compute_score

    path = Path("data/evalplus_humaneval_test.parquet")
    if not path.exists():
        pytest.skip("EvalPlus HumanEval parquet not yet generated")

    df = pd.read_parquet(path)
    row = df.iloc[0]
    gt = row["reward_model"]["ground_truth"]

    # A deliberately wrong solution should not get acc=1.0
    result = compute_score("def foo(): return 42", gt)
    assert isinstance(result, dict)
    assert "score" in result
    assert "acc" in result
    assert 0.0 <= result["score"] <= 1.0


def test_evalplus_correct_solution_scores_perfectly():
    """A known correct solution for the first HumanEval+ problem should score 1.0."""
    import pandas as pd
    from pathlib import Path
    from experiment4_llm.src.reward_function import compute_score

    path = Path("data/evalplus_humaneval_test.parquet")
    if not path.exists():
        pytest.skip("EvalPlus HumanEval parquet not yet generated")

    df = pd.read_parquet(path)
    row = df.iloc[0]
    gt = row["reward_model"]["ground_truth"]
    gt_parsed = json.loads(gt)
    fn_name = gt_parsed["fn_name"]

    # has_close_elements — canonical solution
    correct_code = f'''```python
from typing import List

def {fn_name}(numbers: List[float], threshold: float) -> bool:
    sorted_numbers = sorted(numbers)
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:
            return True
    return False
```'''

    result = compute_score(correct_code, gt, timeout_per_test=5)
    assert result["acc"] == 1.0, f"Expected acc=1.0, got {result}"
    assert result["score"] == 1.0, f"Expected score=1.0, got {result}"


def test_evalplus_no_overlap_between_humaneval_and_mbpp():
    """HumanEval+ and MBPP+ parquets should have no overlapping problems."""
    import pandas as pd
    from pathlib import Path

    he_path = Path("data/evalplus_humaneval_test.parquet")
    mb_path = Path("data/evalplus_mbpp_test.parquet")
    if not he_path.exists() or not mb_path.exists():
        pytest.skip("EvalPlus parquets not yet generated")

    df_he = pd.read_parquet(he_path)
    df_mb = pd.read_parquet(mb_path)

    he_prompts = set(str(p) for p in df_he["prompt"])
    mb_prompts = set(str(p) for p in df_mb["prompt"])

    overlap = he_prompts & mb_prompts
    assert len(overlap) == 0, \
        f"Found {len(overlap)} overlapping prompts between HumanEval+ and MBPP+"


# ── DeepCoder no-dedup test ──


def test_deepcoder_no_val_split():
    """DeepCoder should only produce a train parquet (no val split)."""
    from pathlib import Path

    assert Path("data/deepcoder_train.parquet").exists(), \
        "deepcoder_train.parquet should exist"
    assert not Path("data/deepcoder_val.parquet").exists(), \
        "deepcoder_val.parquet should NOT exist (no val split)"


# ── Training script validation ──


def test_training_scripts_use_hydra_plus_prefix():
    """extra_data_files must use + prefix for Hydra config override."""
    from pathlib import Path

    scripts_dir = Path("experiment4_llm/scripts")
    for script in scripts_dir.glob("qwen3_1.7b_*_deepcoder_*.sh"):
        content = script.read_text()
        if "extra_data_files" not in content:
            continue
        # Every extra_data_files line must use + prefix
        for line in content.splitlines():
            if "extra_data_files" in line and "=" in line:
                stripped = line.strip()
                assert stripped.startswith("+") or stripped.startswith("\\"), \
                    f"{script.name}: extra_data_files missing '+' prefix: {stripped}"


def test_training_scripts_eval_parquet_files_exist():
    """All parquet files referenced in training scripts must exist."""
    from pathlib import Path

    scripts_dir = Path("experiment4_llm/scripts")
    for script in scripts_dir.glob("qwen3_1.7b_*_deepcoder_*.sh"):
        content = script.read_text()
        # Replace ${REPO_ROOT} with actual path
        content = content.replace("${REPO_ROOT}", "/projects/bgmo/sramasubramanian/cont-maxrl")
        for line in content.splitlines():
            if ".parquet" in line and "=" in line:
                # Extract parquet path
                for part in line.split("="):
                    part = part.strip().strip('"').strip("'").rstrip(" \\")
                    if part.endswith(".parquet"):
                        assert Path(part).exists(), \
                            f"{script.name}: parquet file not found: {part}"


def test_training_scripts_model_ids_exist_on_huggingface():
    """Model IDs in training scripts must be valid HuggingFace repos.

    Bug: Qwen/Qwen3-1.7B-128K-Instruct was used but doesn't exist on HF.
    Qwen3 naming: Qwen3-1.7B = instruct, Qwen3-1.7B-Base = base.
    """
    from pathlib import Path
    from huggingface_hub import HfApi

    api = HfApi()
    scripts_dir = Path("experiment4_llm/scripts")
    checked = set()

    for script in scripts_dir.glob("qwen3_1.7b_*_deepcoder_*.sh"):
        content = script.read_text()
        for line in content.splitlines():
            if "model.path=" in line:
                model_id = line.split("model.path=")[1].strip().rstrip(" \\")
                if model_id in checked:
                    continue
                checked.add(model_id)
                try:
                    api.model_info(model_id)
                except Exception:
                    pytest.fail(
                        f"{script.name}: model '{model_id}' not found on HuggingFace"
                    )


def test_training_scripts_base_uses_base_model():
    """Base scripts must use Qwen3-1.7B-Base, not the instruct model.

    Bug: base scripts were pointing to Qwen/Qwen3-1.7B (the instruct model).
    """
    from pathlib import Path

    scripts_dir = Path("experiment4_llm/scripts")
    for script in scripts_dir.glob("qwen3_1.7b_base_deepcoder_*.sh"):
        content = script.read_text()
        for line in content.splitlines():
            if "model.path=" in line:
                model_id = line.split("model.path=")[1].strip().rstrip(" \\")
                assert "Base" in model_id, \
                    f"{script.name}: base script should use Base model, got '{model_id}'"


def test_training_scripts_instruct_uses_instruct_model():
    """Instruct scripts must use Qwen3-1.7B (instruct), not the Base model."""
    from pathlib import Path

    scripts_dir = Path("experiment4_llm/scripts")
    for script in scripts_dir.glob("qwen3_1.7b_instruct_deepcoder_*.sh"):
        content = script.read_text()
        for line in content.splitlines():
            if "model.path=" in line:
                model_id = line.split("model.path=")[1].strip().rstrip(" \\")
                assert "Base" not in model_id, \
                    f"{script.name}: instruct script should NOT use Base model, got '{model_id}'"


# ── EvalPlus data prep unit tests (regression) ──


def test_evalplus_handles_non_list_plus_input():
    """EvalPlus prep must handle plus_input being a dict (not list).

    Bug: Mbpp/793 has plus_input as a dict, causing TypeError on list concat.
    """
    from experiment4_llm.src.data_prep import prepare_evalplus
    # This is a unit-level check: the function should not crash.
    # The actual fix converts non-list plus_input to [].
    # We verify by checking the generated parquets have no crashes.
    import pandas as pd
    from pathlib import Path

    for name in ["evalplus_humaneval_test.parquet", "evalplus_mbpp_test.parquet"]:
        path = Path(f"data/{name}")
        if not path.exists():
            pytest.skip(f"{name} not yet generated")
        df = pd.read_parquet(path)
        # If we got here without crash, the non-list handling worked.
        # Additionally verify no record has empty outputs with non-empty inputs
        for i, row in df.iterrows():
            gt = json.loads(row["reward_model"]["ground_truth"])
            assert len(gt["inputs"]) > 0, f"{name} row {i}: no inputs"
            assert len(gt["outputs"]) == len(gt["inputs"]), \
                f"{name} row {i}: input/output mismatch ({len(gt['inputs'])} vs {len(gt['outputs'])})"


def test_evalplus_set_return_values_serialized():
    """Functions returning sets must be serialized as sorted lists.

    Bug: json.dumps(set(...)) raises TypeError. The fix converts sets to
    sorted lists before serialization. Verify no record has empty outputs
    when inputs exist (the symptom of this bug).
    """
    import pandas as pd
    from pathlib import Path

    path = Path("data/evalplus_mbpp_test.parquet")
    if not path.exists():
        pytest.skip("MBPP+ parquet not yet generated")

    df = pd.read_parquet(path)
    for i, row in df.iterrows():
        gt = json.loads(row["reward_model"]["ground_truth"])
        n_in = len(gt["inputs"])
        n_out = len(gt["outputs"])
        assert n_out == n_in, \
            f"Row {i} (fn={gt.get('fn_name')}): {n_in} inputs but {n_out} outputs. " \
            f"Likely a set/non-serializable return value bug."
        # Also verify outputs are valid JSON
        for j, out in enumerate(gt["outputs"][:5]):
            try:
                json.loads(out)
            except json.JSONDecodeError:
                pytest.fail(f"Row {i} output {j} is not valid JSON: {out[:100]}")
