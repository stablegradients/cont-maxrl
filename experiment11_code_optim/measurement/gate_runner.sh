#!/bin/bash
# gate_runner.sh — batched gate runner for the PIE measurement core.
#
# Usage: gate_runner.sh <exe_path> <test_case_dir> <problem_id> <per_test_timeout_s>
#
# Runs <exe_path> against every input.<N>.txt in <test_case_dir>/<problem_id>/,
# compares stdout to output.<N>.txt with per-line trailing-whitespace
# normalisation (spec §4.2), and emits one "PASS <N>" or "FAIL <N>" line to
# stdout per test case. Exit code 0 always — the result is in stdout, not the
# exit code. A test that hits the per-test timeout, segfaults, or has a
# non-zero return code is reported as FAIL.
#
# Replaces N sequential Python subprocess.run() calls (one per test case)
# with one fork+exec at the Python level. The shell still spawns the binary
# per test, but the dominant cost (Python ThreadPoolExecutor + GIL +
# subprocess.Popen overhead) is paid once.

set -u  # explicit errors on undefined vars; do NOT use -e — partial failures are expected.

EXE="$1"
TEST_DIR="$2/$3"
TIMEOUT="$4"

if [[ ! -x "${EXE}" ]]; then
    echo "ERROR exe not executable: ${EXE}" >&2
    exit 2
fi
if [[ ! -d "${TEST_DIR}" ]]; then
    echo "ERROR test_dir missing: ${TEST_DIR}" >&2
    exit 2
fi

for inp in "${TEST_DIR}"/input.*.txt; do
    [[ -e "$inp" ]] || continue  # no matches → loop body skipped
    idx="${inp##*input.}"
    idx="${idx%.txt}"
    expected_file="${TEST_DIR}/output.${idx}.txt"
    if [[ ! -e "${expected_file}" ]]; then
        echo "FAIL ${idx}"
        continue
    fi
    # Capture stdout; trap timeout/non-zero exit.
    if ! actual_raw=$(timeout "${TIMEOUT}" "${EXE}" < "${inp}" 2>/dev/null); then
        echo "FAIL ${idx}"
        continue
    fi
    # Per-line trailing-whitespace normalisation on both sides, then compare.
    # Note: command substitution $() strips trailing newlines from the captured
    # output — that mirrors Python's `.rstrip("\n")` step in core._normalize().
    norm_actual=$(printf '%s\n' "${actual_raw}" | sed 's/[[:space:]]*$//')
    norm_expected=$(sed 's/[[:space:]]*$//' "${expected_file}")
    if [[ "${norm_actual}" == "${norm_expected}" ]]; then
        echo "PASS ${idx}"
    else
        echo "FAIL ${idx}"
    fi
done
