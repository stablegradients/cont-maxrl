from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .sandbox import ExecutionResult, execute_in_sandbox, outputs_match


@dataclass(frozen=True)
class RewardConfig:
    timeout_seconds: int = 5
    memory_limit_mb: int = 256


@dataclass(frozen=True)
class TestCases:
    inputs: list[Any]
    outputs: list[Any]
    fn_name: str | None = None


def parse_test_cases(raw: str | Mapping[str, Any] | TestCases) -> TestCases:
    if isinstance(raw, TestCases):
        return raw

    if isinstance(raw, str):
        data = json.loads(raw)
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        raise TypeError(f"Unsupported test case payload type: {type(raw)!r}")

    inputs = list(data.get("inputs", []))
    outputs = list(data.get("outputs", []))
    fn_name = data.get("fn_name")

    if len(inputs) != len(outputs):
        raise ValueError(
            f"Mismatched test cases: {len(inputs)} inputs vs {len(outputs)} outputs"
        )

    return TestCases(inputs=inputs, outputs=outputs, fn_name=fn_name)


def run_single_test_case(
    generated_code: str,
    input_data: Any,
    expected_output: Any,
    fn_name: str | None,
    config: RewardConfig,
) -> tuple[bool, ExecutionResult]:
    result = execute_in_sandbox(
        code=generated_code,
        input_data=input_data,
        fn_name=fn_name,
        timeout_seconds=config.timeout_seconds,
        memory_limit_mb=config.memory_limit_mb,
    )
    passed = (
        not result.timed_out
        and result.returncode == 0
        and outputs_match(result.stdout, expected_output)
    )
    return passed, result


def compute_reward(
    generated_code: str,
    test_cases: str | Mapping[str, Any] | TestCases,
    config: RewardConfig | None = None,
) -> float:
    """
    Compute continuous outcome reward: fraction of test cases passed in [0, 1].
    """
    if config is None:
        config = RewardConfig()

    parsed = parse_test_cases(test_cases)
    num_total = len(parsed.inputs)
    if num_total == 0:
        return 0.0

    num_passed = 0
    for input_data, expected_output in zip(parsed.inputs, parsed.outputs):
        try:
            passed, _ = run_single_test_case(
                generated_code=generated_code,
                input_data=input_data,
                expected_output=expected_output,
                fn_name=parsed.fn_name,
                config=config,
            )
            if passed:
                num_passed += 1
        except Exception:
            # Any failure during execution/evaluation counts as failed test.
            continue

    return num_passed / num_total


def compute_reward_with_details(
    generated_code: str,
    test_cases: str | Mapping[str, Any] | TestCases,
    config: RewardConfig | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if config is None:
        config = RewardConfig()

    parsed = parse_test_cases(test_cases)
    num_total = len(parsed.inputs)
    if num_total == 0:
        return 0.0, []

    details: list[dict[str, Any]] = []
    num_passed = 0

    for idx, (input_data, expected_output) in enumerate(zip(parsed.inputs, parsed.outputs)):
        passed = False
        result = ExecutionResult("", "", returncode=-1, timed_out=False, memory_exceeded=False)
        try:
            passed, result = run_single_test_case(
                generated_code=generated_code,
                input_data=input_data,
                expected_output=expected_output,
                fn_name=parsed.fn_name,
                config=config,
            )
        except Exception as exc:
            result = ExecutionResult(
                stdout="",
                stderr=str(exc),
                returncode=-1,
                timed_out=False,
                memory_exceeded=False,
            )

        if passed:
            num_passed += 1

        details.append(
            {
                "test_index": idx,
                "passed": passed,
                "input": input_data,
                "expected": expected_output,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "memory_exceeded": result.memory_exceeded,
            }
        )

    return num_passed / num_total, details


def compute_batch_rewards(
    generated_codes: list[str],
    test_cases_batch: list[str | Mapping[str, Any] | TestCases],
    config: RewardConfig | None = None,
) -> list[float]:
    if len(generated_codes) != len(test_cases_batch):
        raise ValueError("generated_codes and test_cases_batch must have the same length")

    if config is None:
        config = RewardConfig()

    rewards: list[float] = []
    for code, tests in zip(generated_codes, test_cases_batch):
        rewards.append(compute_reward(code, tests, config=config))
    return rewards
