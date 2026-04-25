#!/usr/bin/env python3
"""
Watchdog: kill rgym binary-reward runs that aren't improving.

Policy:
  For each W&B run in the three reasoning_gym_{task} projects:
    if run name contains '_binary_'
    and run state == 'running'
    and latest logged _step >= min_step (default 100)
    and (latest critic/score/mean - baseline) < min_improvement (default 0.04):
      scancel the matching SLURM job (by name), killing parent + dependents.

  "baseline" is the first logged critic/score/mean in run.history (earliest step).
  A run must improve by >= 4 percentage points over that baseline by step 100 to
  be considered learning.

The SLURM job name is 'rgym_<experiment_name>', which matches the W&B run
name since launch_slurm_deltaai.sh sets both from the same RUN_TAG.

Usage:
  python watchdog_kill_dead_binary.py                     # poll every 15 min, forever
  python watchdog_kill_dead_binary.py --dry-run           # print decisions, don't scancel
  python watchdog_kill_dead_binary.py --once              # one-shot check and exit
  python watchdog_kill_dead_binary.py --min-improvement 0.04 --min-step 100 --interval 900

Env:
  WANDB_API_KEY — read from repo .env if present
  WANDB_ENTITY  — default 'stablegradients'
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECTS = (
    "Reasoning-GYM-GOS",
)
DEFAULT_MIN_IMPROVEMENT = 0.04   # binary runs: kill if (latest − baseline) < 4pp after MIN_STEP
DEFAULT_MIN_STEP = 100
DEFAULT_STALL_WINDOW = 80        # all runs: kill if improvement over last N steps < STALL_DELTA
DEFAULT_STALL_DELTA = 0.01       # 1 percentage point
DEFAULT_INTERVAL_SEC = 900       # 15 min
LOG_FILE = REPO_ROOT / "experiment10_reasoning_gym" / "logs" / "watchdog.log"


def source_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_active_slurm_job_names() -> set[str]:
    """Returns names of currently running or pending rgym_* SLURM jobs."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %T"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logging.warning("squeue failed: %s — assuming all jobs are active", e)
        return set()
    names: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 1 and parts[0].startswith("rgym_"):
            names.add(parts[0])
    return names


def scancel_by_name(name: str, dry_run: bool) -> None:
    if dry_run:
        logging.info("[DRY-RUN] would scancel --name=%s", name)
        return
    try:
        subprocess.run(
            ["scancel", "--name", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logging.info("scancel --name=%s  ✓", name)
    except subprocess.SubprocessError as e:
        logging.error("scancel --name=%s failed: %s", name, e)


def _score_history(run):
    """Return sorted [(step, score), ...] for critic/score/mean, or []."""
    try:
        rows = list(run.history(keys=["critic/score/mean", "_step"], samples=5000, pandas=False))
    except Exception as e:
        logging.warning("history() failed for %s: %s", run.name, e)
        return []
    pts = [
        (r.get("_step", 0), r.get("critic/score/mean"))
        for r in rows
        if r.get("critic/score/mean") is not None
    ]
    pts.sort(key=lambda t: t[0])
    return pts


def _baseline_and_latest(pts):
    if not pts:
        return (None, None, None)
    baseline = pts[0][1]
    latest_step, latest_score = pts[-1]
    return (latest_step, baseline, latest_score)


def _stall_window(pts, window_steps: int):
    """Return (latest_step, latest_score, past_step, past_score) or (None,)*4."""
    if len(pts) < 2:
        return (None, None, None, None)
    latest_step, latest_score = pts[-1]
    target = latest_step - window_steps
    past_step, past_score = min(pts, key=lambda t: abs(t[0] - target))
    if past_step == latest_step:
        return (None, None, None, None)
    return (latest_step, latest_score, past_step, past_score)


def check_once(
    api,
    projects: tuple[str, ...],
    entity: str,
    min_improvement: float,
    min_step: int,
    stall_window: int,
    stall_delta: float,
    dry_run: bool,
) -> int:
    """Two criteria:
      (A) binary runs only, after step >= min_step: (latest − baseline) < min_improvement → kill
      (B) all runs, if latest_step >= stall_window: (latest − past_{stall_window}) < stall_delta → kill
    """
    active = get_active_slurm_job_names()
    logging.info(
        "Active rgym SLURM jobs: %d  (sample: %s)",
        len(active),
        sorted(active)[:5],
    )
    kills = 0
    killed_names: set[str] = set()

    for project in projects:
        try:
            runs = list(api.runs(f"{entity}/{project}", filters={"state": "running"}))
        except Exception as e:
            logging.warning("wandb api %s/%s failed: %s", entity, project, e)
            continue

        for run in runs:
            name = run.name
            slurm_name = f"rgym_{name}"
            if slurm_name in killed_names:
                continue
            if slurm_name not in active:
                continue  # parent gone, skip
            pts = _score_history(run)
            if not pts:
                logging.info("[skip] %s  no history yet", name)
                continue

            latest_step, baseline, latest_score = _baseline_and_latest(pts)
            is_binary = "_binary_" in name

            # Criterion A: binary baseline check
            if is_binary and latest_step >= min_step:
                delta = latest_score - baseline
                if delta < min_improvement:
                    logging.warning(
                        "[kill-A] %s  step=%d  score=%.4f  baseline=%.4f  Δbaseline=%.4f < %.4f",
                        name, latest_step, latest_score, baseline, delta, min_improvement,
                    )
                    scancel_by_name(slurm_name, dry_run=dry_run)
                    killed_names.add(slurm_name); kills += 1
                    continue

            # Criterion B: last-N-step stall
            if latest_step >= stall_window:
                latest_step_b, latest_b, past_step, past_b = _stall_window(pts, stall_window)
                if latest_step_b is None:
                    pass
                else:
                    delta_recent = latest_b - past_b
                    if delta_recent < stall_delta:
                        logging.warning(
                            "[kill-B] %s  step=%d  score=%.4f  past(%d)=%.4f  Δ%d=%.4f < %.4f",
                            name, latest_step_b, latest_b, past_step, past_b,
                            stall_window, delta_recent, stall_delta,
                        )
                        scancel_by_name(slurm_name, dry_run=dry_run)
                        killed_names.add(slurm_name); kills += 1
                        continue

            # Still healthy
            if latest_step >= stall_window:
                delta_recent = latest_score - _stall_window(pts, stall_window)[3]
                logging.info(
                    "[ok] %s  step=%d  score=%.4f  Δ%d=%+.4f",
                    name, latest_step, latest_score, stall_window, delta_recent,
                )
            else:
                logging.info("[ok] %s  step=%d  score=%.4f  (below stall window)",
                             name, latest_step, latest_score)
    return kills


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-improvement", type=float, default=DEFAULT_MIN_IMPROVEMENT,
                        help=f"Criterion A (binary only): kill if (latest-baseline) < this (default {DEFAULT_MIN_IMPROVEMENT})")
    parser.add_argument("--min-step", type=int, default=DEFAULT_MIN_STEP,
                        help=f"Criterion A: only apply after step >= this (default {DEFAULT_MIN_STEP})")
    parser.add_argument("--stall-window", type=int, default=DEFAULT_STALL_WINDOW,
                        help=f"Criterion B (all runs): look-back window in steps (default {DEFAULT_STALL_WINDOW})")
    parser.add_argument("--stall-delta", type=float, default=DEFAULT_STALL_DELTA,
                        help=f"Criterion B: kill if improvement over window < this (default {DEFAULT_STALL_DELTA})")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC,
                        help=f"Poll interval in seconds (default {DEFAULT_INTERVAL_SEC})")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "stablegradients"))
    parser.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    parser.add_argument("--dry-run", action="store_true", help="Log decisions, don't scancel")
    parser.add_argument("--once", action="store_true", help="Run one check then exit")
    args = parser.parse_args()

    source_env_file(REPO_ROOT / ".env")
    if "WANDB_API_KEY" not in os.environ:
        print("ERROR: WANDB_API_KEY not set and no .env file found", file=sys.stderr)
        return 2

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # StreamHandler only — caller redirects stdout if they want a file. Avoids
    # duplicate log lines when running under `nohup ... >> watchdog.log`.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import wandb  # noqa: E402  (import after env set)
    api = wandb.Api()

    logging.info(
        "Watchdog armed: entity=%s projects=%s  A(bin): min_improvement=%.4f @ step>=%d  "
        "B(all): stall_window=%d delta<%.4f  interval=%ds  dry_run=%s",
        args.entity, args.projects, args.min_improvement, args.min_step,
        args.stall_window, args.stall_delta, args.interval, args.dry_run,
    )

    while True:
        try:
            n = check_once(
                api, tuple(args.projects), args.entity,
                args.min_improvement, args.min_step,
                args.stall_window, args.stall_delta,
                args.dry_run,
            )
            logging.info("cycle complete: %d %s", n, "kill(s)" if not args.dry_run else "would-kill(s)")
        except Exception as e:
            logging.exception("cycle crashed: %s", e)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
