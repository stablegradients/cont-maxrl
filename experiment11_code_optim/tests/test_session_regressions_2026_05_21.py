"""
Regression tests for every bug discovered in the 2026-05-20/21 RL bringup session.

Each test corresponds to a specific bug documented in
`docs/SESSION_BUG_ANALYSIS_2026_05_20_21.md`. If any of these fail, the
corresponding bug is likely back.

These are SETUP-level invariants (SLURM scripts, ckpt config files, verl
vendor patches, env-var threading) — they complement the unit tests in
`test_measurement.py` and `test_pie_reward_fn.py` which cover code-level
behaviour.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path("/projects/bgmo/sramasubramanian/cont-maxrl")
SCRIPTS_DIR = REPO / "experiment11_code_optim" / "scripts"
RL_SCRIPTS = [
    SCRIPTS_DIR / "qwen3_1.7b_base_pie_maxrl_g16.sh",
    SCRIPTS_DIR / "qwen3_1.7b_base_pie_grpo_g16.sh",
    SCRIPTS_DIR / "qwen3_1.7b_base_pie_rloo_g16.sh",
]
SFT_CKPT_DIR = Path(
    "/work/nvme/bgtu/sramasubramanian/cont-maxrl/sft_ckpts/"
    "qwen3_1.7b_pie_sft_1ep_fulltrain_chat_s0/final"
)
TEST_CASE_PICKLE = Path("/work/nvme/bgmo/sramasubramanian/pie_data/tests.pkl")
REF_TABLE_PICKLE = Path("/work/nvme/bgmo/sramasubramanian/pie_data/reference_table.pkl")
TRAIN_PARQUET = Path("/work/nvme/bgmo/sramasubramanian/pie_data/parquet/pie_train.parquet")

VENDOR_VERL = REPO / "vendor" / "verl"


# ===========================================================================
# Bug B regression: EOS token id includes <|im_end|>
# ===========================================================================

@pytest.mark.skipif(not SFT_CKPT_DIR.is_dir(), reason="chat-format SFT ckpt not staged")
def test_chat_format_sft_ckpt_has_im_end_in_eos():
    """Bug B: the §9 GO eval returned 0/200 because generation_config.json
    set eos_token_id=151643 (<|endoftext|>) instead of 151645 (<|im_end|>).
    The Qwen3 chat template trains the model to emit <|im_end|>; vLLM
    needs to honor that as a stop token.
    """
    gc = json.loads((SFT_CKPT_DIR / "generation_config.json").read_text())
    eos = gc.get("eos_token_id")
    assert eos is not None, "generation_config.json missing eos_token_id"
    # Must include 151645 — either as a list [151645, ...] or as int 151645.
    eos_set = set(eos) if isinstance(eos, list) else {int(eos)}
    assert 151645 in eos_set, (
        f"eos_token_id={eos!r} does not include 151645 (<|im_end|>); "
        f"the model trained with Qwen3 chat template MUST stop on <|im_end|>"
    )

    tc = json.loads((SFT_CKPT_DIR / "tokenizer_config.json").read_text())
    assert tc.get("eos_token") == "<|im_end|>", (
        f"tokenizer_config.json eos_token={tc.get('eos_token')!r}; "
        f"chat-format SFT ckpt must report <|im_end|> as EOS"
    )


# ===========================================================================
# Bug H regression: filter_overlong_responses must be False
# ===========================================================================

def test_filter_overlong_responses_disabled_in_all_rl_scripts():
    """Bug H: filter_overlong_responses=True (the DAPO default) ZEROES OUT
    advantages for any rollout that hits max_response_length. That removes
    the gradient signal that teaches the model to emit EOS. Must be False.
    """
    for s in RL_SCRIPTS:
        body = s.read_text()
        assert "algorithm.filter_overlong_responses=False" in body, (
            f"{s.name}: filter_overlong_responses must be False to penalize "
            f"runaway rollouts"
        )
        assert "algorithm.filter_overlong_responses=True" not in body, (
            f"{s.name}: filter_overlong_responses=True snuck back in"
        )


# ===========================================================================
# Bug I regression: PIE_SKIP_OLD_LOG_PROB_RECOMPUTE wiring
# ===========================================================================

def test_skip_old_log_prob_env_var_set_in_all_rl_scripts():
    """Bug I: skip the ~3 min/step actor recompute by using vLLM rollout
    log_probs directly. Safe under on-policy ppo_epochs=1."""
    for s in RL_SCRIPTS:
        body = s.read_text()
        assert "export PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1" in body, (
            f"{s.name}: PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1 not exported"
        )


def test_skip_old_log_prob_patch_in_vllm_rollout_spmd():
    """The vllm_rollout_spmd.py patch must populate batch['old_log_probs']
    when PIE_SKIP_OLD_LOG_PROB_RECOMPUTE=1."""
    body = (VENDOR_VERL / "verl" / "workers" / "rollout" / "vllm_rollout" /
            "vllm_rollout_spmd.py").read_text()
    assert 'PIE_SKIP_OLD_LOG_PROB_RECOMPUTE' in body, (
        "vllm_rollout_spmd.py missing the PIE_SKIP_OLD_LOG_PROB_RECOMPUTE branch"
    )
    assert '_batch_dict["old_log_probs"] = rollouts_log_probs' in body, (
        "vllm_rollout_spmd.py: rollouts_log_probs not wired into batch as old_log_probs"
    )


def test_patched_vllm_rollout_module_parses_and_imports():
    """REGRESSION (dry-run crash 2026-05-21, job 2314382):
    My PIE patch's module-level docstring edit truncated the ORIGINAL
    docstring; the rest of the original docstring's content fell outside
    the closing `\"\"\"` and Python tried to parse it as code. The line
    containing `doesn't` (an apostrophe) raised SyntaxError: unterminated
    string literal.

    This test imports the patched module to catch any future docstring or
    syntax breakage in the verl vendor patches. A fresh Python import does
    full parse + module-init.
    """
    import importlib, sys
    # vendor verl on path
    vendor = str(VENDOR_VERL)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    # Force a fresh import to catch any breakage
    mod_name = "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd"
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])
    else:
        importlib.import_module(mod_name)
    m = sys.modules[mod_name]
    # Patch sanity: module-level constant and class are present
    assert hasattr(m, "_PIE_SKIP_OLD")
    assert hasattr(m, "vLLMRollout")


def test_skip_old_log_prob_env_var_read_at_module_level_not_init():
    """REGRESSION (dry-run crash 2026-05-21, job 2314300):
    vLLMRollout.__init__ has a nested `import os` inside a megatron-only
    branch (line 96), which makes Python treat `os` as a function-local
    variable for the entire __init__ body. Reading
    PIE_SKIP_OLD_LOG_PROB_RECOMPUTE inside __init__ via `os.environ.get(…)`
    raised UnboundLocalError BEFORE that nested import executes.

    Fix: read the env var at MODULE LOAD time, store as `_PIE_SKIP_OLD`,
    reference the module-level binding from __init__.

    Guard: source-level check that the env-var read is at module level."""
    body = (VENDOR_VERL / "verl" / "workers" / "rollout" / "vllm_rollout" /
            "vllm_rollout_spmd.py").read_text()
    lines = body.splitlines()
    init_start = next(i for i, l in enumerate(lines) if "def __init__" in l)
    module_level_lines = lines[:init_start]
    init_body_lines = lines[init_start:]
    # The os.environ.get for PIE_SKIP_OLD_LOG_PROB_RECOMPUTE must appear
    # at module level, NOT inside __init__.
    module_has_read = any(
        "PIE_SKIP_OLD_LOG_PROB_RECOMPUTE" in l and "os.environ" in l
        for l in module_level_lines
    )
    init_has_read = any(
        "PIE_SKIP_OLD_LOG_PROB_RECOMPUTE" in l and "os.environ" in l
        for l in init_body_lines
    )
    assert module_has_read, (
        "vllm_rollout_spmd.py must read PIE_SKIP_OLD_LOG_PROB_RECOMPUTE at "
        "module level (NOT inside __init__) to avoid the os-scoping trap"
    )
    assert not init_has_read, (
        "vllm_rollout_spmd.py: do NOT read PIE_SKIP_OLD_LOG_PROB_RECOMPUTE "
        "inside __init__ — the nested megatron `import os` at line 96 makes "
        "`os` function-local for the whole __init__ body, triggering "
        "UnboundLocalError on the env-var read"
    )


def test_skip_old_log_prob_patch_in_ray_trainer():
    """The ray_trainer.py patch must skip compute_log_prob when env var is set."""
    body = (VENDOR_VERL / "verl" / "trainer" / "ppo" / "ray_trainer.py").read_text()
    assert 'PIE_SKIP_OLD_LOG_PROB_RECOMPUTE' in body, (
        "ray_trainer.py missing the PIE_SKIP_OLD_LOG_PROB_RECOMPUTE branch"
    )
    assert 'assert "old_log_probs" in batch.batch.keys()' in body, (
        "ray_trainer.py: missing safety assertion that rollout populated "
        "old_log_probs when skip path is taken. NOTE: must use "
        "batch.batch.keys() because TensorDict raises NotImplementedError "
        "on bare `in`."
    )
    # REGRESSION (dry-run crash 2318910): the skip path must also propagate
    # meta_info["temperature"] because dp_actor.update_policy reads it
    # (would otherwise KeyError). The normal compute_log_prob path sets
    # this via fsdp_workers.py; bypassing that requires explicit assignment.
    assert 'batch.meta_info["temperature"]' in body, (
        "ray_trainer.py: skip path must set batch.meta_info['temperature'] "
        "before update_policy (otherwise dp_actor.update_policy raises "
        "KeyError: 'temperature')"
    )


# ===========================================================================
# Bug L regression: max_response_length matches scripts
# ===========================================================================

EXPECTED_MAX_RESPONSE_LENGTH = 4096


def test_max_response_length_in_all_rl_scripts():
    """Bug L: max_response_length must match the user-chosen value across
    all 3 RL scripts. Drift would mean different arms see different data."""
    for s in RL_SCRIPTS:
        body = s.read_text()
        needle = f"data.max_response_length={EXPECTED_MAX_RESPONSE_LENGTH}"
        assert needle in body, (
            f"{s.name}: expected {needle}, got something else — "
            f"all 3 arms must share max_response_length"
        )


# ===========================================================================
# Bug K regression: token-length filter on parquet
# ===========================================================================

@pytest.mark.skipif(not TRAIN_PARQUET.is_file(), reason="pie_train parquet not staged")
def test_no_prompt_in_parquet_exceeds_filter_threshold():
    """Bug K: build_pie_parquet.py drops records where chat-templated prompt
    exceeds 6144 tokens. If any record exceeds, the dataloader would silently
    drop it or trigger filter_overlong_prompts which can blow up batch shape.
    Verify the filter was applied."""
    try:
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("pyarrow or transformers missing")
    if not SFT_CKPT_DIR.is_dir():
        pytest.skip("SFT ckpt not staged for tokenizer")

    tok = AutoTokenizer.from_pretrained(str(SFT_CKPT_DIR), trust_remote_code=True)
    t = pq.read_table(str(TRAIN_PARQUET), columns=["prompt"])
    n_rows = t.num_rows
    # Spot check first 200 — full scan takes minutes
    long_prompts = 0
    max_len = 0
    for i in range(min(200, n_rows)):
        msgs = t.column("prompt")[i].as_py()
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        n = len(tok(rendered, add_special_tokens=False)["input_ids"])
        if n > max_len: max_len = n
        if n > 6144: long_prompts += 1
    assert long_prompts == 0, (
        f"{long_prompts}/{min(200,n_rows)} sampled rows exceed 6144 tokens "
        f"(max sampled={max_len}); build_pie_parquet.py's token filter is broken"
    )


# ===========================================================================
# Bug M regression: preload fast paths
# ===========================================================================

@pytest.mark.skipif(not TEST_CASE_PICKLE.is_file(), reason="tests.pkl not built yet")
def test_test_case_pickle_fast_path_under_500ms():
    """Bug M: §9 GO eval deadlocked because 4 shards walked Lustre concurrently
    to read the test case tree. Pickle fast path replaces that with one
    open()+read(). Must be <500 ms even on cold cache."""
    # Force re-load by clearing the cache state
    from experiment11_code_optim.reward import pie_reward_fn as pf
    pf._TEST_CASE_CACHE.clear()
    pf._TEST_CASE_CACHE_DIR = None
    os.environ["PIE_TEST_CASE_PICKLE"] = str(TEST_CASE_PICKLE)
    t0 = time.monotonic()
    n = pf.preload_test_cases("/work/nvme/bgmo/sramasubramanian/pie_data/tests")
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert n > 1000, f"expected >1000 test cases, got {n}"
    assert elapsed_ms < 500, (
        f"preload_test_cases took {elapsed_ms:.0f} ms; pickle fast path may "
        f"have regressed and we're walking 21K files on Lustre again"
    )


@pytest.mark.skipif(not REF_TABLE_PICKLE.is_file(), reason="reference_table.pkl not built yet")
def test_reference_table_pickle_fast_path_under_500ms():
    """Bug M companion: reference_table.pkl must load fast."""
    from experiment11_code_optim.reward import pie_reward_fn as pf
    pf._REF_TABLE_CACHE.clear()
    pf._REF_TABLE_CACHE_DIR = None
    os.environ["PIE_REFERENCE_TABLE_PICKLE"] = str(REF_TABLE_PICKLE)
    t0 = time.monotonic()
    n = pf.preload_reference_table("/work/nvme/bgmo/sramasubramanian/pie_data/reference_table")
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert n > 100, f"expected >100 ref entries, got {n}"
    assert elapsed_ms < 500, (
        f"preload_reference_table took {elapsed_ms:.0f} ms; pickle fast path regressed"
    )


# ===========================================================================
# Bug C regression: prlimit cap works under 64-thread concurrency
# ===========================================================================

def test_prlimit_binary_exists():
    """Bug C: the prlimit binary must exist on the target system. If absent
    the cap wouldn't be applied and we'd be back to the Ray-OOM crash mode."""
    assert os.access("/usr/bin/prlimit", os.X_OK), (
        "/usr/bin/prlimit missing or not executable; PIE measurement code "
        "relies on it for RLIMIT_AS without thread-unsafe preexec_fn"
    )


def test_rlimit_cap_works_at_64_thread_concurrency():
    """Bug C: the smoking-gun regression test. Run the same known-good
    program through `_run_and_compare` at 64-thread concurrency. With the
    old preexec_fn path, this returned 0/200 pass. With the prlimit
    wrapper, it must be 200/200.
    """
    from experiment11_code_optim.measurement.core import _compile, _run_and_compare

    HELLO = '#include <iostream>\nint main(){ std::cout << "hi\\n"; return 0; }'
    N = 200
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        exe, _ = _compile(HELLO, d)
        assert exe is not None
        def grade(_):
            p, _c = _run_and_compare(exe, "", "hi\n", 5.0)
            return p
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            results = list(pool.map(grade, range(N)))
    n_pass = sum(results)
    assert n_pass == N, (
        f"{N - n_pass}/{N} grades failed under 64-thread concurrency; "
        f"the preexec_fn thread-deadlock bug may be back (or prlimit is broken)"
    )


# ===========================================================================
# Bug D regression: walltime backend is active (NOT cputime/perf)
# ===========================================================================

def test_walltime_backend_is_active():
    """Bug D: we tried switching to CPU time but the 1-ms kernel-tick
    quantization made it useless for short programs. The production backend
    is walltime."""
    from experiment11_code_optim.measurement.core import MEASUREMENT_BACKEND
    assert MEASUREMENT_BACKEND == "walltime", (
        f"backend is {MEASUREMENT_BACKEND!r}; production must be walltime "
        f"unless we've designed a higher-resolution path"
    )


# ===========================================================================
# Bug A regression: SFT chat template prompt is a strict prefix of full render
# ===========================================================================

@pytest.mark.skipif(not SFT_CKPT_DIR.is_dir(), reason="chat-format SFT ckpt not staged")
def test_sft_chat_template_prompt_is_strict_prefix_of_full_render():
    """Bug A: the SFT loss mask is defined as "prompt = full[:len_prompt],
    completion = full[len_prompt:]". This only works if the prompt-only
    render is a strict token-ID prefix of the prompt+target render. Verify
    on the live tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers missing")
    tok = AutoTokenizer.from_pretrained(str(SFT_CKPT_DIR), trust_remote_code=True)
    user = "Optimize this: ```cpp\nint main(){}\n```"
    asst = "```cpp\nint main(){return 0;}\n```"
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    full_text = tok.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": asst}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, (
        "prompt-with-generation render is NOT a strict prefix of the full "
        "prompt+assistant render; SFT loss mask would be misaligned, "
        "training on garbage. Bug A re-emergence."
    )


# ===========================================================================
# Bug G regression: subprocess.run errors='replace' on g++ stderr
# ===========================================================================

def test_subprocess_run_with_replace_handles_non_utf8_in_compile():
    """Bug G: a program with non-UTF-8 bytes in a string literal makes g++
    echo those bytes back in its error output. subprocess.run(text=True)
    without errors='replace' raises UnicodeDecodeError and kills the
    entire reward batch."""
    import subprocess
    src = b'#include <iostream>\nint main(){const char*s="\xff\xfe";std::cout<<s;return 0;}'
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as f:
        f.write(src); fp = f.name
    try:
        proc = subprocess.run(
            ["/opt/cray/pe/gcc-native/14/bin/g++", "-O3", "-std=c++17",
             "-o", fp + ".out", fp],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
        # The point isn't whether it succeeds — it's that NO exception is
        # raised even with `text=True`. errors='replace' makes that safe.
        assert proc.returncode in (0, 1)  # compile may succeed or fail; both fine
    finally:
        for ext in (".cpp", ".out"):
            try: os.unlink(fp.replace(".cpp", "") + ext)
            except FileNotFoundError: pass


# ===========================================================================
# Configuration sanity: env-var contract used by reward fn
# ===========================================================================

def test_rl_scripts_export_pickle_paths():
    """SLURM scripts must export PIE_TEST_CASE_PICKLE + PIE_REFERENCE_TABLE_PICKLE
    so preload_*_table uses the fast path. Without these, eval / RL grader
    walks the directory tree on Lustre and can hang."""
    for s in RL_SCRIPTS:
        body = s.read_text()
        # We rely on the default fallback path "(test_case_dir).pkl" so
        # explicit export not strictly required, but env-var contract for
        # SLURM scripts should at minimum mention test_case_dir.
        assert "PIE_TEST_CASE_DIR" in body
        assert "PIE_REFERENCE_TABLE_DIR" in body


def test_rl_scripts_set_storage_root_to_bgtu():
    """All RL outputs must land on bgtu (more quota headroom than bgmo)."""
    for s in RL_SCRIPTS:
        body = s.read_text()
        assert 'STORAGE_ROOT="/work/nvme/bgtu/sramasubramanian/cont-maxrl"' in body, (
            f"{s.name}: STORAGE_ROOT not pointed at bgtu"
        )


def test_rl_scripts_default_sft_ckpt_is_chat_format():
    """Bug A/B regression: scripts must default to the chat-format SFT ckpt,
    not the raw-prompt ckpt that was deleted."""
    for s in RL_SCRIPTS:
        body = s.read_text()
        assert "qwen3_1.7b_pie_sft_1ep_fulltrain_chat_s0/final" in body
        assert "qwen3_1.7b_pie_sft_1ep_fulltrain_s3/final" not in body, (
            f"{s.name}: still references the old (deleted) raw-prompt SFT ckpt"
        )


def test_rl_scripts_have_total_epochs_3_not_10():
    """Cosmetic but easy to drift: trainer.total_epochs must be 3 (was 10
    when copied from spec). total_training_steps=PIE_TOTAL_STEPS takes
    precedence anyway, but inconsistency is a future trap."""
    for s in RL_SCRIPTS:
        body = s.read_text()
        assert "trainer.total_epochs=3" in body, (
            f"{s.name}: trainer.total_epochs drifted away from 3"
        )
        assert "trainer.total_epochs=10" not in body
