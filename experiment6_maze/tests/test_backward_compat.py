"""Tier 0: Backward-compatibility tests. Protects Qwen3 pipeline."""
import hashlib
import inspect
import subprocess

import pytest

EXPECTED_HASHES = {
    "grpo_advantage": "393400bdc869079ccbea2de7666372255bb3def9007ca61d9f0675ab10bf4c5b",
    "maxrl_unnormalized_advantage": "139d1c587723a1912cb6fec1254ef2734315d12bf0e9e76b8947943d17d5b3db",
    "maxrl_normalized_advantage": "a02a4f12966802e40bfc64c9ebf6dab961cdc14ddb09af52bc3ab1ed16121380",
    "reinforce_baseline_advantage": "0df01994fa83fec4afc65f1519a291c75d8a8be1f8ce6f2c48a26a948daf445d",
    "rloo_advantage": "b6b5a1be9173713b19485948715bd821b5d3087e97700daf562516df80fe73b8",
}

BASELINE_PIP_HASH = "926654ed0b2933faf816127813e6f0485688a0fbe4f7449713ec2885ea17f792"

PROTECTED_FILES = [
    "experiment4_llm/src/reward_function.py",
    "experiment4_llm/src/data_prep.py",
    "experiment4_llm/src/insitu_eval.py",
    "experiment4_llm/src/train.py",
    "experiment4_llm/src/sandbox.py",
    "experiment4_llm/src/evaluate.py",
]

PROTECTED_FUNCTIONS = [
    "grpo_advantage",
    "maxrl_unnormalized_advantage",
    "maxrl_normalized_advantage",
    "reinforce_baseline_advantage",
    "rloo_advantage",
]


def test_no_edits_to_protected_files():
    """Protected files must be byte-identical to maze-exp-base tag."""
    for path in PROTECTED_FILES:
        result = subprocess.run(
            ["git", "diff", "maze-exp-base", "HEAD", "--", path],
            capture_output=True, text=True,
        )
        assert result.stdout == "", f"{path} was modified:\n{result.stdout[:300]}"


def test_existing_advantage_functions_unchanged():
    """Existing advantage functions must have unchanged source."""
    from experiment4_llm.src import advantages
    for fn_name in PROTECTED_FUNCTIONS:
        fn = getattr(advantages, fn_name)
        source = inspect.getsource(fn)
        actual_hash = hashlib.sha256(source.encode()).hexdigest()
        expected = EXPECTED_HASHES[fn_name]
        assert actual_hash == expected, (
            f"{fn_name} source changed! Expected: {expected[:12]}, Got: {actual_hash[:12]}"
        )


def test_existing_custom_estimators_preserved():
    from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
    required = {"maxrl_unnorm", "maxrl_norm", "reinforce_baseline", "rloo"}
    assert required.issubset(set(CUSTOM_ESTIMATORS.keys()))


def test_binary_maxrl_added():
    from experiment4_llm.src.verl_register import CUSTOM_ESTIMATORS
    assert "binary_maxrl" in CUSTOM_ESTIMATORS


def _canonical_pip_freeze() -> str:
    """Return a deterministic pip freeze string.

    Editable installs and the local cont-maxrl package produce different
    output depending on how pip is invoked (bash vs subprocess).  We exclude
    those lines and sort the rest so the hash is reproducible.
    """
    result = subprocess.run(
        ["/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/pip", "freeze"],
        capture_output=True, text=True,
    )
    lines = sorted(
        line for line in result.stdout.splitlines()
        if line
        and not line.startswith("-e")
        and "cont-maxrl" not in line
        and "cont_maxrl" not in line
    )
    return "\n".join(lines) + "\n"


def test_pip_freeze_unchanged():
    """Conda env must not be modified."""
    canonical = _canonical_pip_freeze()
    current_hash = hashlib.sha256(canonical.encode()).hexdigest()
    assert current_hash == BASELINE_PIP_HASH, (
        f"pip freeze changed! Baseline: {BASELINE_PIP_HASH[:12]}, Current: {current_hash[:12]}"
    )
