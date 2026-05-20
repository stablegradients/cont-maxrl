"""
PIE isolation guard (spec v9 §10).

Assert at training entry that PIE-specific configuration has NOT drifted
into DeepCoder defaults (or vice versa). Each check corresponds to a
drift item D1-D10 documented in `PARITY_CONFIG.md` — if any assertion
fails, the run aborts loudly with a precise message. Hard-fail beats
silent miscompare.

Call from the SLURM script BEFORE launching `train.py`, or wire as a
Hydra entrypoint hook. The checks are read-only against the runtime
environment and config, no network calls, ~milliseconds.

Usage:
    python -m experiment11_code_optim.guards.isolation \\
        --train_parquet $PIE_TRAIN_PARQUET \\
        --val_parquet $PIE_VAL_PARQUET \\
        --reference_table_dir $PIE_REFERENCE_TABLE_DIR \\
        --test_case_dir $PIE_TEST_CASE_DIR \\
        --total_steps $PIE_TOTAL_STEPS \\
        --steps_per_epoch $PIE_STEPS_PER_EPOCH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


class IsolationViolation(RuntimeError):
    """Raised when a D1-D10 drift item is detected. Hard-stops training."""


def _assert(cond: bool, dN: str, msg: str) -> None:
    if not cond:
        raise IsolationViolation(f"[{dN}] {msg}")


def check_d1_slurm_account() -> None:
    """D1: PIE runs on the bgmo account, never on a DeepCoder account."""
    acct = os.environ.get("SLURM_JOB_ACCOUNT") or os.environ.get("SBATCH_ACCOUNT")
    if not acct:
        return  # not under SLURM; nothing to check
    _assert(
        acct.startswith("bgmo"),
        "D1",
        f"PIE runs must use bgmo* SLURM account; got {acct!r}",
    )


def check_d2_epoch_arithmetic(total_steps: int, steps_per_epoch: int, n_train: int) -> None:
    """D2: PIE epoch arithmetic must be computed from N_pie_train,
    NOT inherited from DeepCoder's 83/1000/996.
    """
    import math
    expected_spe = math.ceil(n_train / 256)
    _assert(
        steps_per_epoch == expected_spe,
        "D2",
        f"steps_per_epoch={steps_per_epoch} != ceil({n_train}/256)={expected_spe}",
    )
    _assert(
        total_steps == 10 * steps_per_epoch,
        "D2",
        f"total_steps={total_steps} != 10 * steps_per_epoch={steps_per_epoch} = "
        f"{10*steps_per_epoch}",
    )
    # DeepCoder values that MUST NOT leak in
    _assert(
        total_steps not in (996, 1000),
        "D2",
        f"total_steps={total_steps} matches DeepCoder's value — epoch arithmetic "
        f"drifted from PIE-computed back to DeepCoder default",
    )


def check_d4_wandb_key() -> None:
    """D4: WANDB_API_KEY must come from env, never hardcoded."""
    _assert(
        bool(os.environ.get("WANDB_API_KEY")),
        "D4",
        "WANDB_API_KEY not set in env; PIE scripts must source it from env, "
        "never inline the key (spec §3.5 / SECURITY_NOTE.md)",
    )


def check_d6_pie_mode_for_rl(is_rl: bool) -> None:
    """F6 (also D9): the PIE_MODE shim must be active when running RL so
    insitu_eval routes through pie_reward_fn. Without it, every C++ rollout
    in insitu_eval silently scores 0."""
    if not is_rl:
        return
    _assert(
        os.environ.get("PIE_MODE") == "1",
        "D9/F6",
        "PIE_MODE=1 must be exported before the RL launch so insitu_eval's "
        "hardcoded reward_function import is redirected to pie_reward_fn. "
        "Without this, every C++ rollout in insitu_eval scores 0 silently.",
    )


def check_d8_reference_table(reference_table_dir: str | None) -> None:
    """D8: the reference table must exist when the percentile-rank reward is
    active. Missing tables → stub 0.5 → no learning signal from the percentile
    head; the reward becomes the binary 0/0.5 gate, defeating PIE's purpose."""
    if reference_table_dir is None:
        raise IsolationViolation(
            "[D8] PIE_REFERENCE_TABLE_DIR not set; reference-table percentile "
            "reward cannot run. Build via "
            "precompute/build_reference_table.py before launching RL."
        )
    p = Path(reference_table_dir) / "MANIFEST.json"
    _assert(p.exists(), "D8", f"reference table MANIFEST.json missing at {p}")


def check_d8_test_case_dir(test_case_dir: str | None) -> None:
    """D8: unified test-case archive must exist; reward path cannot grade
    without it."""
    if test_case_dir is None:
        raise IsolationViolation(
            "[D8] PIE_TEST_CASE_DIR not set; gate cannot run."
        )
    p = Path(test_case_dir) / "MANIFEST.json"
    _assert(
        p.exists(),
        "D8",
        f"test-case archive MANIFEST.json missing at {p}. Build via "
        f"precompute/build_unified_test_cases.py.",
    )


def check_d10_go_probe_config() -> None:
    """D10 (latent §13 GO config): the speed-fix envs must be live."""
    _assert(
        bool(os.environ.get("PIE_MEASUREMENT_MAX_TESTS")),
        "D10",
        "PIE_MEASUREMENT_MAX_TESTS not set; the K-cap that brought reward "
        "time under the 50% GATE B threshold is off — training will be "
        "uneconomically slow.",
    )
    _assert(
        bool(os.environ.get("PIE_PCH_INCLUDE_PATH")),
        "D10",
        "PIE_PCH_INCLUDE_PATH not set; PCH is not used; per-compile time is "
        "2.65× slower than under the GO config.",
    )


def assert_all_pie_invariants(
    *,
    train_parquet: str | None = None,
    val_parquet: str | None = None,
    reference_table_dir: str | None = None,
    test_case_dir: str | None = None,
    total_steps: int | None = None,
    steps_per_epoch: int | None = None,
    is_rl: bool = True,
) -> None:
    """Run every applicable D1-D10 check. Raises IsolationViolation on first
    failure with a precise message."""
    check_d1_slurm_account()
    check_d4_wandb_key()
    check_d6_pie_mode_for_rl(is_rl)
    check_d8_reference_table(reference_table_dir)
    check_d8_test_case_dir(test_case_dir)
    check_d10_go_probe_config()

    if total_steps is not None and steps_per_epoch is not None and train_parquet is not None:
        # n_train read lazily — we don't want to depend on pyarrow at module load.
        try:
            import pyarrow.parquet as pq
            n_train = pq.read_metadata(train_parquet).num_rows
        except Exception as e:
            raise IsolationViolation(
                f"[D2] could not read num_rows from {train_parquet!r}: {e}"
            )
        check_d2_epoch_arithmetic(total_steps, steps_per_epoch, n_train)


def main() -> int:
    ap = argparse.ArgumentParser(description="PIE D1-D10 isolation guard")
    ap.add_argument("--train_parquet", default=os.environ.get("PIE_TRAIN_PARQUET"))
    ap.add_argument("--val_parquet", default=os.environ.get("PIE_VAL_PARQUET"))
    ap.add_argument("--reference_table_dir",
                    default=os.environ.get("PIE_REFERENCE_TABLE_DIR"))
    ap.add_argument("--test_case_dir", default=os.environ.get("PIE_TEST_CASE_DIR"))
    ap.add_argument("--total_steps", type=int,
                    default=int(os.environ.get("PIE_TOTAL_STEPS", "0")) or None)
    ap.add_argument("--steps_per_epoch", type=int,
                    default=int(os.environ.get("PIE_STEPS_PER_EPOCH", "0")) or None)
    ap.add_argument("--sft", action="store_true",
                    help="Run with SFT-mode invariants (skips RL-only checks like PIE_MODE).")
    args = ap.parse_args()

    try:
        assert_all_pie_invariants(
            train_parquet=args.train_parquet,
            val_parquet=args.val_parquet,
            reference_table_dir=args.reference_table_dir,
            test_case_dir=args.test_case_dir,
            total_steps=args.total_steps,
            steps_per_epoch=args.steps_per_epoch,
            is_rl=not args.sft,
        )
    except IsolationViolation as e:
        print(f"ISOLATION GUARD FAIL: {e}", file=sys.stderr)
        return 1

    print("ISOLATION GUARD PASS: D1-D10 invariants OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
