"""Production gem5-speedup reward for PIE RL (#43) — correctness gate + gem5 timing.

Spec (user, 2026-06-12):
- The reward FUNCTION is an argument (no single fixed transform). All transforms
  assume reward 0 unless the rollout passes correctness on EVERY case (and 0 if it
  exceeds the per-case tick budget); ln is the NATURAL log. With s = src_time/rollout_time:
    ratio           r = s
    log1p_ratio     r = 1 + ln(1 + s)
    gated_log_ratio r = ln(s) if rollout_time < src_time else 0
- Phase 1 (native): each rollout runs ALL the dataset's filtered cases for the
  problem (`usable_case_ids` from pie-gem5-{bysrc,pairs}); any wrong output /
  crash / timeout => reward 0. Comparator and compiler are byte-identical to the
  screen that defined `usable_case_ids` (measurement.core._compile @ -O0 and
  _outputs_match: line-wise + 1e-3 float tol).
- Phase 2 (gem5): only gate-passers are timed, on at most K usable cases
  (k_eff = min(K, n_usable); K=-1 = all). Per-case budget: a rollout exceeding
  budget_mult (3x) the SRC's ticks on a case is stopped and gets reward 0
  (forkloop budget at 3 x max(src over K) stops the sim early; an exact
  per-case 3x post-check enforces the rule case-by-case).
- src_time / rollout_time = sums of gem5 ticks over the SAME K cases; src ticks
  come from the dataset (`gem5_src_per_tc_ticks`), rollout ticks are measured
  live with the IDENTICAL pinned toolchain + gem5 build that produced them.
- K-case sampling default "group": every rollout of the same (problem, src) at
  a step gets the same K cases (advantages compare like-for-like); rotates per
  call/step. Also "rollout" (independent) and "fixed" (stable across steps).
- Robustness: every child runs in its own process group (timeout kills the
  whole tree), prlimit caps address space / file size (print bombs) / nproc
  (fork bombs) / core dumps; per-rollout cumulative native budget; a global
  batch deadline zero-fills stragglers and SIGKILLs their process groups so an
  RL step can never stall. Data/infra faults zero the rollout immediately (no
  retry, per user decision); config errors raise loudly up front.
- Memory: expected outputs are loaded once per problem per batch and freed
  after; program stdout goes to size-capped files (never unbounded pipes);
  per-rollout tmpdirs are removed in finally; duplicate (problem, code)
  rollouts are graded once and fanned out.
- Parallelism: ThreadPoolExecutor with nearest-power-of-2(available cores)
  workers, each thread driving one pinned subprocess at a time (compile /
  native run / gem5 are all separate processes, so all cores are busy without
  fork()ing the trainer driver).

verl wiring:
    custom_reward_function.path=.../reward/gem5_reward.py
    custom_reward_function.name=compute_score
ground_truth JSON per sample must carry: problem_id, usable_case_ids,
gem5_src_per_tc_ticks (or src_per_tc_ticks), optionally src_id (group key).
Everything else (reward_kind, k_cases, sample_mode, the per-step `step` nonce,
etc.) is reward_kwargs / env, NOT per-sample ground_truth.

Self-test on a node with the real gem5 stack:
    python experiment11_code_optim/reward/gem5_reward.py --selftest --n 3
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait
from dataclasses import dataclass, field
from queue import Empty, Queue

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiment11_code_optim.measurement.core import (  # noqa: E402
    _compile, _outputs_match, setup_pch,
)
from experiment11_code_optim.measurement import gem5_backend as _g5  # noqa: E402

try:  # battle-tested completion->C++ parser (special tokens, </think>, anchors)
    from experiment11_code_optim.reward.pie_reward_fn import extract_code as _extract_code
except Exception:  # pragma: no cover - minimal fallback if pie_reward_fn is unavailable
    def _extract_code(text: str) -> str | None:
        m = re.findall(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", text or "", re.S)
        return m[-1].strip() if m else None

# public name: insitu_eval imports `compute_scores_batch, extract_code` from the
# active reward module (PIE_REWARD_IMPL=gem5 routes it here)
extract_code = _extract_code

# Actor-checkpoint carve-out (#43 §8.3). The keep-all hf_model snapshotter
# patches RayPPOTrainer._save_checkpoint, which executes INSIDE the Ray
# TaskRunner actor — the same process that imports this reward module (verl
# exec_module's the custom reward fn here, and the threaded reward executor runs
# here too). train.py's DRIVER-side install never reaches that actor, which is
# why actor_ckpts/ stayed EMPTY in every prior run. Install here as well:
# idempotent (_INSTALLED guard) and a no-op unless PIE_ACTOR_CARVEOUT_DIR is set.
try:
    from experiment11_code_optim.guards.actor_carveout import (  # noqa: E402
        install_actor_carveout_hook as _install_actor_carveout_hook,
    )
    _install_actor_carveout_hook()
except Exception as _carveout_e:  # never let carve-out wiring break the reward
    print(f"[carveout] WARN: install from gem5_reward failed: {_carveout_e}", file=sys.stderr)

_PRLIMIT = "/usr/bin/prlimit"
_TASKSET = "/usr/bin/taskset"

# ----------------------------------------------------------------------------- config

def _env_f(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _env_i(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Gem5RewardConfig:
    """All knobs. Defaults read the env at construction time (not import time)."""
    # reward selection
    reward_kind: str = field(default_factory=lambda: os.environ.get("PIE_GEM5_REWARD_KIND", "ratio"))
    k_cases: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_K", "5"))  # -1 = all usable
    sample_mode: str = field(default_factory=lambda: os.environ.get("PIE_GEM5_REWARD_SAMPLE", "group"))
    # WHICH usable cases get gem5-timed (correctness always runs on ALL usable):
    #   "sample" (default) -> random K via select_cases(sample_mode)
    #   "biggest"          -> the K largest-src-ticks usable cases (compute-heavy
    #                         cases that expose an algorithmic gap; deterministic).
    # Per-row ground_truth "gem5_case_ids" still overrides this when present.
    case_select: str = field(default_factory=lambda: os.environ.get("PIE_GEM5_CASE_SELECT", "sample"))
    seed: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_SEED", "0"))
    step: int | None = None          # RL step for group/rollout rotation; None -> per-call counter
    budget_mult: float = field(default_factory=lambda: _env_f("PIE_GEM5_BUDGET_MULT", "3.0"))
    # native gate
    native_timeout_s: float = field(default_factory=lambda: _env_f("PIE_NAT_TIMEOUT", "4"))
    native_rollout_budget_s: float = field(default_factory=lambda: _env_f("PIE_GEM5_REWARD_NATIVE_BUDGET", "120"))
    mem_limit_mb: int = field(default_factory=lambda: _env_i("PIE_PROGRAM_MEM_LIMIT_GB", "1") * 1024)
    fsize_limit_mb: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_FSIZE_MB", "64"))
    nproc_limit: int = field(default_factory=lambda: _env_i("PIE_GEM5_REWARD_NPROC", "2048"))
    use_pch: bool = field(default_factory=lambda: os.environ.get("PIE_GEM5_REWARD_PCH", "1") != "0")
    # (native phase always stops at the first failing case — there is no
    #  run-all-cases-anyway mode, so it is not a configurable knob)
    # parallelism / deadline
    workers: int | None = None       # None -> nearest_pow2(available cores); env PIE_GEM5_REWARD_WORKERS
    global_deadline_s: float = field(default_factory=lambda: _env_f("PIE_GEM5_REWARD_DEADLINE", "3600"))
    # data
    test_case_dir: str = field(default_factory=lambda: os.environ.get(
        "PIE_TEST_CASE_DIR",
        "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"))
    detail: bool = False             # include per-case native walls in results


# ----------------------------------------------------------------------- reward kinds

def _r_ratio(s: float) -> float:
    return s


def _r_log1p_ratio(s: float) -> float:
    return 1.0 + math.log1p(s)          # natural log; only ever applied on gate-pass


def _r_gated_log_ratio(s: float) -> float:
    return math.log(s) if s > 1.0 else 0.0   # natural log


REWARD_KINDS = {
    # ALL gated: reward is 0 unless the rollout passes EVERY usable test case
    # (and 0 if it exceeds the per-case tick budget); these transforms are
    # applied only on the full-pass path. s = src_time / rollout_time.
    "ratio": _r_ratio,                    # 11a: s
    "log1p_ratio": _r_log1p_ratio,        # 11b: 1 + ln(1 + s)
    "gated_log_ratio": _r_gated_log_ratio,  # 11c: 1{rollout<src} ln(s)
}


def nearest_pow2(n: int) -> int:
    """Nearest power of two (ties resolve DOWN: 96 -> 64, 72 -> 64, 288 -> 256)."""
    if n <= 1:
        return 1
    lo = 1 << (n.bit_length() - 1)
    hi = lo << 1
    return lo if (n - lo) <= (hi - n) else hi


def _resolve_workers(cfg: Gem5RewardConfig, ncores: int) -> int:
    env = os.environ.get("PIE_GEM5_REWARD_WORKERS")
    if cfg.workers:
        return max(1, cfg.workers)
    if env:
        return max(1, int(env))
    return nearest_pow2(ncores)


def _affinity_cores() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return list(range(os.cpu_count() or 1))


# ------------------------------------------------------------------------ case sampling

_CALL_COUNTER = itertools.count()


def _reset_call_counter() -> None:  # test hook
    global _CALL_COUNTER
    _CALL_COUNTER = itertools.count()


def select_cases(usable: list[int], k: int, mode: str, seed: int, nonce: int,
                 group_key: str, uid_idx: int) -> list[int]:
    """Pick the gem5-timed case ids. Pure + deterministic given the arguments.

    k=-1 or k>=n -> all cases (clamp k_eff = min(K, n); never pad/skip/zero a
    short problem). mode: group = same cases for every rollout sharing
    group_key this call; rollout = independent per rollout; fixed = stable
    across calls (no nonce). Returned sorted for stable input ordering.
    """
    n = len(usable)
    if k < 0 or k >= n:
        return sorted(usable)
    if mode == "group":
        key = f"{seed}|{nonce}|{group_key}"
    elif mode == "rollout":
        key = f"{seed}|{nonce}|{group_key}|{uid_idx}"
    elif mode == "fixed":
        key = f"{seed}|{group_key}"
    else:
        raise ValueError(f"unknown sample_mode {mode!r} (group|rollout|fixed)")
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    import random as _random
    return sorted(_random.Random(h).sample(sorted(usable), k))


# --------------------------------------------------------------------- process registry

class _ProcRegistry:
    """Live child process-group ids, so the global deadline can SIGKILL
    everything a straggler task has in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pg: dict[int, int] = {}

    def register(self, key: int, pgid: int) -> None:
        with self._lock:
            self._pg[key] = pgid

    def unregister(self, key: int) -> None:
        with self._lock:
            self._pg.pop(key, None)

    def kill_all(self) -> None:
        with self._lock:
            pgids = list(self._pg.values())
            self._pg.clear()
        for pg in pgids:
            try:
                os.killpg(pg, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _killpg_and_reap(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        pass


# -------------------------------------------------------------------------- native gate

def _run_case_native(exe: str, in_path: str, expected: str, cfg: Gem5RewardConfig,
                     cpu: int | None, reg: _ProcRegistry, reg_key: int,
                     ) -> tuple[bool, str, float]:
    """Run one test case. Returns (passed, why, wall_s).

    Verdict-identical to the screen's core._run_and_compare (same comparator,
    same AS cap, rc!=0/timeout => fail) with three robustness upgrades that
    cannot flip a legitimate verdict: stdout goes to a size-capped FILE (a
    print bomb gets SIGXFSZ instead of OOMing the trainer), the child runs in
    its own process group (timeout kills the whole tree, incl. fork bombs,
    which prlimit --nproc additionally caps), and core dumps are off.
    """
    argv = [_PRLIMIT,
            f"--as={cfg.mem_limit_mb << 20}",
            f"--fsize={cfg.fsize_limit_mb << 20}",
            f"--nproc={cfg.nproc_limit}",
            "--core=0", "--"]
    if cpu is not None:
        argv += [_TASKSET, "-c", str(cpu)]
    argv += [exe]
    out_path = exe + ".out"
    t0 = time.monotonic()
    try:
        with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
            p = subprocess.Popen(argv, stdin=fin, stdout=fout,
                                 stderr=subprocess.DEVNULL,
                                 cwd=os.path.dirname(exe), start_new_session=True)
            reg.register(reg_key, p.pid)
            try:
                rc = p.wait(timeout=cfg.native_timeout_s)
                # the leader exited; reap any descendants it left spinning
                # (e.g. a fork bomb whose parent returns 0) — the process
                # group outlives the leader while members are alive
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            except subprocess.TimeoutExpired:
                _killpg_and_reap(p)
                return False, "timeout", time.monotonic() - t0
            finally:
                reg.unregister(reg_key)
    except OSError as e:
        return False, f"os_error:{e}", time.monotonic() - t0
    wall = time.monotonic() - t0
    if rc != 0:
        return False, f"exit_{rc}", wall
    try:
        cap = (cfg.fsize_limit_mb << 20) + 1
        with open(out_path, "rb") as f:
            out = f.read(cap).decode("utf-8", "replace")
    except OSError as e:
        return False, f"read_error:{e}", wall
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return (_outputs_match(out, expected), "mismatch", wall)


# gem5 timing indirection (monkeypatch point for tests)
_gem5_time_one_fn = _g5.gem5_time_one


# ------------------------------------------------------------------------- one rollout

def _zero(phase: str, reason: str, **extra) -> dict:
    return {"score": 0.0, "acc": 0.0, "speedup": 0.0, "phase": phase,
            "reason": reason, **extra}


def _grade_one(rep: dict, cfg: Gem5RewardConfig, transform, prob: dict,
               core_q: Queue, reg: _ProcRegistry, scratch: str,
               pch: str | None, deadline_t: float) -> dict:
    """Grade ONE unique (problem, code) rollout: native gate on all usable
    cases, then gem5 on its selected K cases. Never raises."""
    cases_all: list[int] = rep["usable"]
    src_ticks: dict[int, int] = rep["src_ticks"]
    sel: list[int] = rep["sel_cases"]
    reg_key = threading.get_ident()

    try:
        cpu = core_q.get(timeout=max(1.0, deadline_t - time.monotonic()))
    except Empty:
        return _zero("deadline", "no core before deadline")
    tmpdir = tempfile.mkdtemp(prefix="g5rw_", dir=scratch)
    t_nat0 = time.monotonic()
    try:
        exe, cerr = _compile(rep["code"], tmpdir, pch_include_path=pch)
        if exe is None and pch is not None:
            # a forced -include of bits/stdc++.h can (rarely) clash with code
            # that redefines stdlib symbols; the screen compiled without a PCH,
            # so retry bare before calling it a compile failure
            exe, cerr = _compile(rep["code"], tmpdir, pch_include_path=None)
        if exe is None:
            return _zero("compile", f"native compile: {cerr}",
                         wall_native_s=time.monotonic() - t_nat0)

        # phase 1: ALL usable cases, cheapest (by src ticks) first for fast fails
        order = sorted(cases_all, key=lambda c: src_ticks.get(c, 1 << 62))
        walls: dict[int, float] = {}
        n_pass = 0
        for c in order:
            io = prob["cases"].get(c)
            if io is None:
                return _zero("infra", f"missing test files for case {c}",
                             n_native_pass=n_pass, n_native_total=len(order))
            ok, why, w = _run_case_native(exe, io[0], io[1], cfg, cpu, reg, reg_key)
            walls[c] = w
            if not ok:
                return _zero("native", f"case {c}: {why}",
                             n_native_pass=n_pass, n_native_total=len(order),
                             wall_native_s=time.monotonic() - t_nat0)
            n_pass += 1
            if (time.monotonic() - t_nat0 > cfg.native_rollout_budget_s
                    and n_pass < len(order)):
                return _zero("native_budget",
                             f"native phase over {cfg.native_rollout_budget_s}s "
                             f"after {n_pass}/{len(order)} cases",
                             n_native_pass=n_pass, n_native_total=len(order),
                             wall_native_s=time.monotonic() - t_nat0)
        wall_native = time.monotonic() - t_nat0

        # phase 2: gem5 on the selected cases (gate fully passed => acc=1)
        t_g50 = time.monotonic()
        srcs = [src_ticks.get(c) for c in sel]
        if not sel or any(s is None or s <= 0 for s in srcs):
            return _zero("infra", "missing/invalid src ticks for selected cases",
                         n_native_pass=n_pass, n_native_total=len(order)) | {"acc": 1.0}
        budget_abs = int(cfg.budget_mult * max(srcs))
        inputs = []
        for c in sel:
            try:
                with open(prob["cases"][c][0], "r", errors="replace") as f:
                    inputs.append(f.read())
            except OSError as e:
                return _zero("infra", f"read input {c}: {e}") | {"acc": 1.0}
        res = _gem5_time_one_fn(rep["code"], inputs, cpu=cpu,
                                max_tc_ticks=budget_abs)
        wall_g5 = time.monotonic() - t_g50
        base = {"acc": 1.0, "n_native_pass": n_pass, "n_native_total": len(order),
                "gem5_case_ids": sel, "wall_native_s": round(wall_native, 3),
                "wall_gem5_s": round(wall_g5, 3)}
        if cfg.detail:
            base["native_case_wall_s"] = {c: round(w, 4) for c, w in walls.items()}
        if res.get("error") or not res.get("per_tc"):
            return {**_zero("gem5", f"gem5: {res.get('error') or 'no per-tc ticks'}"),
                    **base}
        per_tc = res["per_tc"]
        if len(per_tc) != len(sel):
            return {**_zero("gem5", f"gem5 per-tc count {len(per_tc)} != {len(sel)}"),
                    **base}
        over = [c for c, s, t in zip(sel, srcs, per_tc)
                if t > cfg.budget_mult * s]
        if over:
            return {**_zero("gem5_over_budget",
                            f"rollout > {cfg.budget_mult}x src on case(s) {over}"),
                    **base}
        src_sum, cand_sum = sum(srcs), sum(per_tc)
        speedup = src_sum / cand_sum
        return {**base, "score": float(transform(speedup)), "speedup": float(speedup),
                "phase": "ok", "reason": None,
                "src_sum_ticks": int(src_sum), "cand_sum_ticks": int(cand_sum)}
    except Exception as e:  # never poison the batch
        return _zero("internal", f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        core_q.put(cpu)


# ------------------------------------------------------------------------ batch driver

_PCH_CACHE: dict[str, str | None] = {}
_PCH_LOCK = threading.Lock()


def _ensure_pch(scratch: str) -> str | None:
    with _PCH_LOCK:
        if scratch in _PCH_CACHE:
            return _PCH_CACHE[scratch]
        try:
            p = setup_pch(scratch)
        except Exception:
            p = None
        _PCH_CACHE[scratch] = p
        return p


def _load_problem(test_case_dir: str, pid: str, case_ids: set[int],
                  cache: dict[str, dict]) -> dict:
    """Per-problem case data: {case_id: (input_path, expected_str)}.
    Loaded once per batch; missing files map to None (-> infra zero)."""
    cur = cache.get(pid)
    if cur is None:
        cur = {"cases": {}}
        cache[pid] = cur
    d = os.path.join(test_case_dir, pid)
    for c in case_ids:
        if c in cur["cases"]:
            continue
        ip = os.path.join(d, f"input.{c}.txt")
        op = os.path.join(d, f"output.{c}.txt")
        try:
            with open(op, "r", errors="replace") as f:
                exp = f.read()
            cur["cases"][c] = (ip, exp) if os.path.exists(ip) else None
        except OSError:
            cur["cases"][c] = None
    return cur


def compute_rewards_batch(jobs: list[dict], cfg: Gem5RewardConfig | None = None,
                          ) -> list[dict]:
    """Grade a batch of rollouts. Returns one result dict per job, in order.

    Each job: {uid?, problem_id, code | solution_str, usable_case_ids,
               src_per_tc_ticks, src_id? | group_key?, gem5_case_ids?}.
    Config errors raise immediately (a typo must not silently zero a run);
    per-rollout data/infra/runtime errors NEVER raise — they zero that rollout
    with a structured phase/reason.
    """
    cfg = cfg or Gem5RewardConfig()
    if cfg.reward_kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward_kind {cfg.reward_kind!r}; "
                         f"choose from {sorted(REWARD_KINDS)}")
    if cfg.k_cases == 0 or cfg.k_cases < -1:
        raise ValueError("k_cases must be -1 (all) or >= 1")
    if cfg.budget_mult <= 0:
        raise ValueError("budget_mult must be > 0")
    if cfg.sample_mode not in ("group", "rollout", "fixed"):
        raise ValueError(f"unknown sample_mode {cfg.sample_mode!r}")
    if cfg.case_select not in ("sample", "biggest"):
        raise ValueError(f"unknown case_select {cfg.case_select!r} (sample|biggest)")
    transform = REWARD_KINDS[cfg.reward_kind]
    nonce = cfg.step if cfg.step is not None else next(_CALL_COUNTER)

    n = len(jobs)
    results: list[dict | None] = [None] * n
    if n == 0:
        return []

    # ---- normalize jobs; extract code; bad data -> immediate zero ----
    norm: list[dict | None] = [None] * n
    for i, j in enumerate(jobs):
        try:
            if j.get("_bad_gt"):
                results[i] = _zero("infra", f"bad ground_truth: {j['_bad_gt']}")
                continue
            code = j.get("code")
            if not code:
                code = _extract_code(j.get("solution_str") or "")
            if not code or not code.strip():
                results[i] = _zero("extract", "no C++ code in completion")
                continue
            uc = j["usable_case_ids"]
            if isinstance(uc, (str, bytes)):
                uc = json.loads(uc)
            usable = sorted({int(c) for c in uc})
            if not usable:
                results[i] = _zero("infra", "empty usable_case_ids")
                continue
            tk = j["src_per_tc_ticks"]
            if isinstance(tk, (str, bytes)):
                tk = json.loads(tk)
            ticks = {int(k): int(v) for k, v in dict(tk).items()}
            gk = j.get("group_key") or f"{j['problem_id']}|{j.get('src_id') or ''}"
            norm[i] = {"uid": j.get("uid", i), "problem_id": str(j["problem_id"]),
                       "code": code, "usable": usable, "src_ticks": ticks,
                       "group_key": gk, "sel_override": j.get("gem5_case_ids")}
        except Exception as e:
            results[i] = _zero("infra", f"bad job fields: {type(e).__name__}: {e}")

    live = [i for i in range(n) if norm[i] is not None]
    if not live:
        return [r | {"uid": jobs[i].get("uid", i)} for i, r in enumerate(results)]

    # ---- per-problem case data (loaded once) + per-job case selection ----
    prob_cache: dict[str, dict] = {}
    still = []
    for i in live:
        nj = norm[i]
        try:
            if nj["sel_override"]:                       # explicit per-row pin (highest priority)
                sel = sorted(int(c) for c in nj["sel_override"])
            elif cfg.case_select == "biggest":           # K largest-src-ticks usable cases
                k = cfg.k_cases if cfg.k_cases > 0 else len(nj["usable"])
                ranked = sorted(nj["usable"], key=lambda c: nj["src_ticks"].get(c, 0), reverse=True)
                sel = sorted(ranked[:k])
            else:                                        # random K (default)
                sel = select_cases(nj["usable"], cfg.k_cases, cfg.sample_mode,
                                   cfg.seed, nonce, nj["group_key"], i)
            nj["sel_cases"] = sel
            _load_problem(cfg.test_case_dir, nj["problem_id"],
                          set(nj["usable"]) | set(sel), prob_cache)
            still.append(i)
        except Exception as e:
            results[i] = _zero("internal", f"case selection/load: {type(e).__name__}: {e}")
            norm[i] = None
    live = still

    # ---- dedup identical rollouts. The key carries everything the reward
    # depends on: the row identity (group_key + usable + src ticks — the same
    # code against a DIFFERENT src has a different reward), the code, and the
    # selected cases (under "rollout" sampling two copies may differ).
    reps: dict[tuple, dict] = {}
    members: dict[tuple, list[int]] = {}
    for i in live:
        nj = norm[i]
        row_fp = hash((nj["group_key"], tuple(nj["usable"]),
                       tuple(sorted(nj["src_ticks"].items()))))
        key = (nj["problem_id"], row_fp,
               hashlib.sha1(nj["code"].encode("utf-8", "replace")).hexdigest(),
               tuple(nj["sel_cases"]))
        reps.setdefault(key, nj)
        members.setdefault(key, []).append(i)

    # ---- thread pool over unique rollouts ----
    cores = _affinity_cores()
    workers = min(_resolve_workers(cfg, len(cores)), max(1, len(reps)))
    core_q: Queue = Queue()
    for c in cores[::-1]:           # tail cores first (away from Ray/vLLM head cores)
        core_q.put(c)
    reg = _ProcRegistry()
    scratch = (os.environ.get("PIE_COMPILE_SCRATCH")
               or os.environ.get("PIE_SCRATCH_DIR")
               or os.environ.get("TMPDIR") or tempfile.gettempdir())
    pch = _ensure_pch(scratch) if cfg.use_pch else None
    deadline_t = time.monotonic() + cfg.global_deadline_s

    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="g5rw")
    futs = {}
    try:
        for key, rep in reps.items():
            prob = prob_cache[rep["problem_id"]]
            futs[ex.submit(_grade_one, rep, cfg, transform, prob,
                           core_q, reg, scratch, pch, deadline_t)] = key
        done, not_done = cf_wait(futs, timeout=cfg.global_deadline_s)
        key_res: dict[tuple, dict] = {}
        for f in done:
            try:
                key_res[futs[f]] = f.result()
            except Exception as e:  # _grade_one never raises, but be paranoid
                key_res[futs[f]] = _zero("internal", f"{type(e).__name__}: {e}")
        if not_done:
            reg.kill_all()          # unblock straggler subprocesses immediately
            for f in not_done:
                f.cancel()
                key_res[futs[f]] = _zero(
                    "deadline", f"global deadline {cfg.global_deadline_s}s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    for key, idxs in members.items():
        r = key_res.get(key) or _zero("internal", "result missing")
        for i in idxs:
            results[i] = dict(r)

    out = []
    for i in range(n):
        r = results[i] or _zero("internal", "no result")
        r["uid"] = (norm[i] or {}).get("uid", jobs[i].get("uid", i))
        r["reward_kind"] = cfg.reward_kind
        out.append(r)
    return out


# ------------------------------------------------------------------------ verl adapters

_CFG_KEYS = ("reward_kind", "k_cases", "sample_mode", "case_select", "seed", "step", "budget_mult",
             "native_timeout_s", "native_rollout_budget_s", "mem_limit_mb",
             "fsize_limit_mb", "nproc_limit", "use_pch", "workers",
             "global_deadline_s", "test_case_dir", "detail")


def _cfg_from_kwargs(kw: dict) -> Gem5RewardConfig:
    return Gem5RewardConfig(**{k: kw[k] for k in _CFG_KEYS if k in kw and kw[k] is not None})


def _job_from_ground_truth(i: int, solution_str: str, ground_truth) -> dict:
    job = {"uid": i, "solution_str": solution_str or ""}
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, (str, bytes)) else dict(ground_truth)
        ticks = gt.get("gem5_src_per_tc_ticks", gt.get("src_per_tc_ticks"))
        if isinstance(ticks, (str, bytes)):
            ticks = json.loads(ticks)
        usable = gt.get("usable_case_ids")
        if isinstance(usable, (str, bytes)):
            usable = json.loads(usable)
        # Optional: pin the gem5-TIMED cases (correctness still runs on ALL
        # usable cases in _grade_one; only the speedup is measured on these).
        # Used by the algorithmic-headroom overfit: time only the large case
        # that exposes the asymptotic gap. Accepts gem5_case_ids|timed_case_ids.
        gem5_cases = gt.get("gem5_case_ids", gt.get("timed_case_ids"))
        if isinstance(gem5_cases, (str, bytes)):
            gem5_cases = json.loads(gem5_cases)
        job.update(problem_id=gt["problem_id"], usable_case_ids=usable,
                   src_per_tc_ticks=ticks, src_id=gt.get("src_id"),
                   gem5_case_ids=gem5_cases)
    except Exception as e:
        job["_bad_gt"] = f"{type(e).__name__}: {e}"
    return job


# --------------------------------------------------------------- distributed reward
# The reward normally runs as ONE Ray task on ONE node (verl's @ray.remote(num_cpus=1)
# compute_reward_async), so on a multi-node run the other nodes' cores + RAM sit idle.
# With PIE_GEM5_REWARD_DISTRIBUTED=1, compute_scores_batch shards the batch across the
# live Ray nodes (one child task per node, NodeAffinity-pinned), grades each shard with
# the SAME local path, and gathers IN ORDER. The whole gem5 stack + test cases live on
# the shared FS, so any node can grade any rollout. Default OFF (single-node behavior).
# Shards are GROUP-ALIGNED (split at multiples of G) so each prompt's G rollouts stay on
# one node -> within-group dedup preserved and (for case_select=sample) the per-group
# case choice stays self-consistent. Falls back to local on any error.


def _gkey(item: dict):
    """The (problem_id|src_id) group key from a reward_input's ground_truth, or None."""
    try:
        gt = item.get("ground_truth")
        gt = json.loads(gt) if isinstance(gt, (str, bytes)) else dict(gt)
        return f"{gt.get('problem_id')}|{gt.get('src_id')}"
    except Exception:
        return None


def _group_aligned_shards(reward_inputs: list, n_shards: int) -> list[list]:
    """Split into <= n_shards contiguous chunks that NEVER split a prompt's group,
    using the ACTUAL group_key from ground_truth (robust to any group size G, and to
    the rollouts being group-contiguous as verl lays them out). Order-preserving.
    Falls back to a fixed-G split if group keys can't be parsed."""
    if n_shards <= 1 or len(reward_inputs) <= 1:
        return [reward_inputs]
    keys = [_gkey(it) for it in reward_inputs]
    if any(k is None for k in keys):     # malformed -> safe fixed-size fallback
        return _split_group_aligned(reward_inputs, n_shards, _env_i("PIE_GEM5_GROUP_SIZE", "16"))
    bounds = [0]                          # start index of each contiguous group
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            bounds.append(i)
    bounds.append(len(reward_inputs))
    n_groups = len(bounds) - 1
    per = max(1, (n_groups + n_shards - 1) // n_shards)
    shards = []
    for g0 in range(0, n_groups, per):
        s = bounds[g0]; e = bounds[min(g0 + per, n_groups)]
        shards.append(reward_inputs[s:e])
    return shards


def _split_group_aligned(items: list, n_shards: int, group_size: int) -> list[list]:
    """Fixed-G fallback splitter: contiguous chunks whose boundaries land on multiples
    of group_size. Order-preserving. (Primary path is _group_aligned_shards.)"""
    if n_shards <= 1 or len(items) <= group_size:
        return [items]
    n_groups = (len(items) + group_size - 1) // group_size
    groups_per_shard = (n_groups + n_shards - 1) // n_shards
    chunk = max(1, groups_per_shard) * group_size
    return [items[i:i + chunk] for i in range(0, len(items), chunk)]


def _grade_shard_fn(shard, kwargs):  # runs (possibly) on a remote node
    return compute_scores_batch(shard, _local_only=True, **kwargs)


def _distribute_scores_batch(reward_inputs: list[dict], kwargs: dict):
    """Fan the batch out across live Ray nodes; return index-aligned results, or
    None to signal 'fall back to local'."""
    try:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        if not ray.is_initialized():
            return None
        nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU")]
        if len(nodes) < 2:
            return None
        shards = _group_aligned_shards(reward_inputs, len(nodes))
        # Resolve the reward config on THIS (head) node and pass it explicitly to
        # every shard, so a worker node with a different/missing PIE_GEM5_* env still
        # grades with identical settings. workers/step stay None -> each node resolves
        # its own worker count from its env (128/node is what we want).
        cfg = _cfg_from_kwargs(kwargs)
        child_kwargs = dict(kwargs)
        child_kwargs.update({k: getattr(cfg, k) for k in _CFG_KEYS
                             if getattr(cfg, k, None) is not None})
        remote_fn = ray.remote(num_cpus=1)(_grade_shard_fn)
        futs = []
        for shard, node in zip(shards, nodes):
            if not shard:
                continue
            strat = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=True)
            futs.append(remote_fn.options(scheduling_strategy=strat).remote(shard, child_kwargs))
        out: list = []
        for f in futs:                       # gather in shard order -> index-aligned
            out.extend(ray.get(f))
        if len(out) != len(reward_inputs):    # shape sanity; never return a mismatched batch
            print(f"[dist-reward] shard count {len(out)} != {len(reward_inputs)}; local fallback",
                  file=sys.stderr)
            return None
        return out
    except Exception as e:
        print(f"[dist-reward] fell back to local: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def compute_scores_batch(reward_inputs: list[dict], **kwargs) -> list[dict]:
    """verl batch entry. reward_inputs: [{solution_str, ground_truth}, ...].
    Returns [{score, acc, speedup, reason}, ...] (extra keys are pivoted into
    non_tensor_batch by verl's reward manager)."""
    if (not kwargs.pop("_local_only", False)
            and os.environ.get("PIE_GEM5_REWARD_DISTRIBUTED") == "1"
            and len(reward_inputs) > _env_i("PIE_GEM5_GROUP_SIZE", "16")):
        dist = _distribute_scores_batch(reward_inputs, dict(kwargs))
        if dist is not None:
            return dist
    cfg = _cfg_from_kwargs(kwargs)
    jobs = [_job_from_ground_truth(i, it.get("solution_str", ""), it.get("ground_truth"))
            for i, it in enumerate(reward_inputs)]
    res = compute_rewards_batch(jobs, cfg)
    return [{"score": r["score"], "acc": r["acc"], "speedup": r["speedup"],
             "reason": r.get("reason") or ""} for r in res]


def compute_score(solution_str: str, ground_truth, data_source=None,
                  extra_info=None, **kwargs) -> dict:
    """verl per-sample entry (thin wrapper over the batch path)."""
    return compute_scores_batch(
        [{"solution_str": solution_str, "ground_truth": ground_truth}], **kwargs)[0]


# ---------------------------------------------------------------------------- selftest

def _selftest(args) -> int:  # pragma: no cover - needs the real gem5 stack
    """Golden check on real dataset rows: src code as the rollout must score
    ratio ~= 1.0; the oracle tgt must score >> 1. Run on a compute node."""
    import pandas as pd
    if not _g5.gem5_available():
        print("gem5 stack not available on this node", file=sys.stderr)
        return 2
    ds = os.environ.get("PIE_RL_DATASET", "/work/nvme/bgtv/sramasubramanian/pie_rl_dataset")
    df = pd.read_parquet(f"{ds}/bysrc_{args.split}.parquet")
    df = df[df.n_usable >= max(args.k, 2)].head(args.n)
    jobs, expect = [], []
    for _, row in df.iterrows():
        gt = {"problem_id": row.problem_id, "src_id": row.src_id,
              "usable_case_ids": json.loads(row.usable_case_ids),
              "gem5_src_per_tc_ticks": row.gem5_src_per_tc_ticks}
        for tag, code in (("src", row.src_code), ("tgt", row.oracle_tgt_code)):
            jobs.append(_job_from_ground_truth(len(jobs), f"```cpp\n{code}\n```", json.dumps(gt)))
            expect.append((row.problem_id, tag,
                           1.0 if tag == "src" else float(row.our_speedup_usable)))
    cfg = Gem5RewardConfig(reward_kind="ratio", k_cases=args.k, sample_mode="fixed")
    t0 = time.time()
    res = compute_rewards_batch(jobs, cfg)
    ok = True
    for (pid, tag, exp), r in zip(expect, res):
        line = (f"{pid} {tag:3s} score={r['score']:.3f} acc={r['acc']:.0f} "
                f"phase={r['phase']} reason={r['reason']} (dataset ref ~{exp:.3f})")
        if tag == "src" and not (r["acc"] == 1.0 and 0.85 <= r["score"] <= 1.15):
            line += "  <-- FAIL (src-as-rollout should be ~1.0)"
            ok = False
        print(line)
    print(f"wall {time.time() - t0:.1f}s   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest(a))
    ap.print_help()
