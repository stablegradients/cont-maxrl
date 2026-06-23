"""gem5 timing backend for the PIE reward (#43).

Times candidate programs with the validated x86 gem5 + skylake config pipeline:
  1. cross-compile aarch64 -> x86_64 with the byte-identical g++-9 toolchain (#42)
  2. probe the pre-input-read tick (startup) via run_se_probe + GEM5_PREINPUT_FORK
  3. fork-amortized sequential forkloop over the K test inputs (run_se_forkloop);
     per-tc tick = absolute run-to-completion tick (startup included), matching
     the dataset's agg_runtime semantics
  4. gem5_ticks = sum of per-tc ticks over the SAME deterministic K-subset the
     correctness gate and the precomputed src/tgt references use

Correctness is NOT checked here - callers gate with the native runner first and
only send gate-passing unique programs (gem5 stdout is not compared).

All toolchain paths are env-overridable (PIE_GEM5_*); defaults point at the
validated build on Delta. Failure modes return ticks=None with an error string;
callers decide the reward consequence (score 0).
"""
from __future__ import annotations

import atexit
import glob
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait as cf_wait

_RF = os.environ.get(
    "PIE_GEM5_ROOTFS",
    "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs",
)
_XROOT = os.environ.get(
    "PIE_GEM5_XCROSS_ROOT", "/work/nvme/bgtv/sramasubramanian/x86cross/root"
)
_GEM5 = os.environ.get(
    "PIE_GEM5_BIN",
    "/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.fast",
)
_ENVD = os.environ.get(
    "PIE_GEM5_PYENV", "/work/nvme/bgtv/sramasubramanian/envs/gem5build"
)
_CFG = f"{_RF}/home/gem5-skylake-config/gem5-configs"
_PROBE = f"{_CFG}/run_se_probe.py"
_SEQ = f"{_CFG}/run_se_forkloop.py"
_GXX = f"{_XROOT}/usr/bin/x86_64-linux-gnu-g++-9"
_LD = f"{_RF}/lib/x86_64-linux-gnu/ld-2.31.so"

_COMPILE_TIMEOUT = float(os.environ.get("PIE_GEM5_COMPILE_TIMEOUT", "120"))
_PROBE_TIMEOUT = float(os.environ.get("PIE_GEM5_PROBE_TIMEOUT", "300"))
_FORKLOOP_TIMEOUT = float(os.environ.get("PIE_GEM5_FORKLOOP_TIMEOUT", "900"))
# Wall-clock cap on a whole gem5_time_batch call (the RL-step analog of
# grade_dedup_batch's global_timeout / deferred fix F5): stragglers past the
# cap are reported as {"ticks": None, "error": "gem5 batch timeout"}.
_BATCH_TIMEOUT = float(os.environ.get("PIE_GEM5_BATCH_TIMEOUT", "1200"))
_FORK_MARGIN = 3000000  # ticks before READ_TICK to place the fork (validated)

_GENV = {
    **os.environ,
    "PATH": f"{_XROOT}/usr/bin:" + os.environ.get("PATH", ""),
    "LD_LIBRARY_PATH": f"{_XROOT}/usr/lib/aarch64-linux-gnu:"
                       f"{_XROOT}/lib/aarch64-linux-gnu:{_XROOT}/usr/lib",
}
_LFLAGS = [
    "--sysroot=" + _RF,
    f"-L{_RF}/usr/lib/gcc/x86_64-linux-gnu/9",
    f"-L{_RF}/usr/lib/x86_64-linux-gnu",
    f"-L{_RF}/lib/x86_64-linux-gnu",
]
_NATENV = {
    **os.environ,
    "LD_LIBRARY_PATH": f"{_ENVD}/lib",
    "PYTHONPATH": f"{_CFG}:{_CFG}/system",
}

_PRAGMA_RE = re.compile(r"#pragma GCC target.*\n")


def gem5_available() -> bool:
    """True if the cross-compiler + gem5 + config paths all exist."""
    return all(os.path.exists(p) for p in (_GXX, _GEM5, _PROBE, _SEQ, _LD))


def _scratch_base() -> str:
    return os.environ.get("PIE_GEM5_SCRATCH") or (
        "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    )


def _scratch_root() -> str:
    d = os.path.join(_scratch_base(), f"pie_gem5_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    return d


def _sweep_stale_roots() -> None:
    """Remove pie_gem5_<pid> roots left by dead processes (crash/SIGKILL)."""
    for d in glob.glob(os.path.join(_scratch_base(), "pie_gem5_*")):
        try:
            pid = int(d.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)            # alive -> leave its scratch alone
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
        except PermissionError:
            pass                        # someone else's live process


_sweep_stale_roots()
atexit.register(lambda: shutil.rmtree(
    os.path.join(_scratch_base(), f"pie_gem5_{os.getpid()}"), ignore_errors=True))


def _run_pg(argv: list[str], env: dict, timeout: float) -> bytes | None:
    """Run argv in its own process group; on timeout SIGKILL the whole group
    (gem5's forkloop m5.fork()s real OS children that a plain subprocess.run
    kill would orphan mid-simulation). Returns stdout, or None on timeout."""
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.communicate(timeout=10)
        except Exception:
            pass
        return None


def _expand_cpuset(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _core_pool_cores() -> list[int]:
    """Cores for gem5 sims: PIE_GEM5_CPUSET > PIE_REWARD_CPUSET > affinity.
    Guards against being called from a single-core-pinned thread (which would
    silently serialize the whole batch)."""
    affinity = sorted(os.sched_getaffinity(0))
    spec = os.environ.get("PIE_GEM5_CPUSET") or os.environ.get("PIE_REWARD_CPUSET")
    if spec:
        try:
            cores = _expand_cpuset(spec)
            return cores or affinity
        except ValueError:
            pass
    if len(affinity) <= 1 and (os.cpu_count() or 1) > 1:
        return list(range(os.cpu_count()))
    return affinity


def _cross_compile(code: str, workdir: str, cpu: int | None) -> tuple[str | None, str | None]:
    """Compile to an x86 binary. Returns (binary_path, error)."""
    src = os.path.join(workdir, "cand.cpp")
    binp = os.path.join(workdir, "cand")
    with open(src, "w") as f:
        f.write(_PRAGMA_RE.sub("", code))
    argv = ["timeout", "-s", "KILL", str(max(1, int(_COMPILE_TIMEOUT) - 5))]
    if cpu is not None:
        argv += ["taskset", "-c", str(cpu)]
    argv += [_GXX] + _LFLAGS + [
        "-O3", "-std=c++17", "-Wl,--dynamic-linker=" + _LD, src, "-o", binp,
    ]
    try:
        r = subprocess.run(argv, env=_GENV, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=_COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "gem5 cross-compile timeout"
    if r.returncode != 0 or not os.path.exists(binp):
        return None, "gem5 cross-compile failed: " + r.stderr.decode("utf-8", "ignore")[-300:]
    return binp, None


def _probe_startup(binp: str, inp_path: str, workdir: str, cpu: int | None) -> tuple[int | None, str | None]:
    """READ_TICK of the first stdin read (input-invariant). (tick, error)."""
    out = os.path.join(workdir, "probe_m5out")
    os.makedirs(out, exist_ok=True)
    argv = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [
        _GEM5, "-d", out, "--stats-file=s.txt", _PROBE, "Verbatim", binp,
    ]
    env = {**_NATENV, "GEM5_STDIN": inp_path, "GEM5_PREINPUT_FORK": "1"}
    try:
        so = _run_pg(argv, env, _PROBE_TIMEOUT)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    if so is None:
        return None, "gem5 startup probe timeout"
    m = re.search(r"READ_TICK (\d+) cause (.+)", so.decode("utf-8", "ignore"))
    if not m:
        return None, "gem5 startup probe parse failure"
    if m.group(2).strip() != "preinput_read":
        # program never read stdin before exiting: the probe already simulated
        # the FULL run, and this tick is the exit tick (validated against the
        # reward-level sweep). The caller reuses it directly — no second sim.
        return int(m.group(1)), "no_stdin"
    return int(m.group(1)), None


def _forkloop_ticks(binp: str, inp_paths: list[str], fork_tick: int,
                    workdir: str, cpu: int | None,
                    max_tc_ticks: int | None = None) -> tuple[list[int] | None, str | None]:
    """Per-tc absolute completion ticks via the sequential forkloop.

    max_tc_ticks: optional ABSOLUTE per-tc tick budget (e.g. 3x the src's
    mean per-tc ticks). A child exceeding it reports cause=budget_exceeded
    and the program is censored as too-slow instead of simulating to
    completion or the wall timeout.
    """
    fix = os.path.join(workdir, "fixed_stdin")
    tcf = os.path.join(workdir, "tc_list")
    shutil.copyfile(inp_paths[0], fix)
    with open(tcf, "w") as f:
        f.write("\n".join(inp_paths))
    out = os.path.join(workdir, "fl_m5out")
    os.makedirs(out, exist_ok=True)
    argv = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [
        _GEM5, "-d", out, "--stats-file=s.txt", _SEQ, "Verbatim", binp,
    ]
    env = {**_NATENV, "GEM5_FIXED_STDIN": fix, "GEM5_TC_LIST": tcf,
           "GEM5_FORK_TICK": str(fork_tick)}
    if max_tc_ticks:
        # forkloop budget is RELATIVE to the fork point; keep a sane minimum
        env["GEM5_TC_MAX_TICKS"] = str(max(int(max_tc_ticks) - fork_tick, 100_000_000))
    try:
        raw = _run_pg(argv, env, _FORKLOOP_TIMEOUT)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    if raw is None:
        return None, "gem5 forkloop timeout"
    so = raw.decode("utf-8", "ignore")
    ms = re.search(r"STARTUP cause=(.+) tick=(\d+)", so)
    if not ms or "limit reached" not in ms.group(1) or int(ms.group(2)) != fork_tick:
        return None, "gem5 forkloop bad startup phase"
    pairs = re.findall(r"TC \d+ tick (\d+) cause (.+)", so)
    over = [c for _, c in pairs if c.strip() == "budget_exceeded"]
    if over:
        return None, "gem5 tc budget exceeded (rollout slower than the budget factor x src)"
    bad = [c for _, c in pairs if not c.strip().startswith("exiting")]
    if bad:
        return None, f"gem5 forkloop abnormal tc exit ({len(bad)}/{len(pairs)})"
    if len(pairs) != len(inp_paths):
        return None, f"gem5 forkloop partial ({len(pairs)}/{len(inp_paths)} tcs)"
    return [int(t) for t, _ in pairs], None


def gem5_time_one(code: str, inputs: list[str], *, cpu: int | None = None,
                  scratch: str | None = None,
                  max_tc_ticks: int | None = None) -> dict:
    """Time one program on a list of test-case input STRINGS.

    Returns {"ticks": int | None, "per_tc": list[int] | None, "error": str | None}.
    ticks = sum of absolute per-tc completion ticks (startup included), the
    same quantity as the dataset's agg_runtime.
    max_tc_ticks: optional absolute per-tc tick budget (censors slow programs
    early; see _forkloop_ticks).
    """
    if not inputs:
        return {"ticks": None, "per_tc": None, "error": "no test inputs"}
    root = scratch or _scratch_root()
    workdir = tempfile.mkdtemp(dir=root, prefix="g5_")
    try:
        binp, err = _cross_compile(code, workdir, cpu)
        if binp is None:
            return {"ticks": None, "per_tc": None, "error": err}
        inp_paths = []
        for i, s in enumerate(inputs):
            p = os.path.join(workdir, f"input.{i}.txt")
            with open(p, "w") as f:
                f.write(s)
            inp_paths.append(p)
        startup, err = _probe_startup(binp, inp_paths[0], workdir, cpu)
        if startup is None:
            return {"ticks": None, "per_tc": None, "error": err}
        if err == "no_stdin":
            # input-invariant program: the probe's tick IS the full-run exit
            # tick (the hook never fired), and every tc costs the same
            return {"ticks": startup * len(inputs),
                    "per_tc": [startup] * len(inputs), "error": None}
        if startup <= _FORK_MARGIN + 1000:
            return {"ticks": None, "per_tc": None,
                    "error": "startup tick below fork margin"}
        per_tc, err = _forkloop_ticks(binp, inp_paths, startup - _FORK_MARGIN,
                                      workdir, cpu, max_tc_ticks=max_tc_ticks)
        if per_tc is None:
            return {"ticks": None, "per_tc": None, "error": err}
        return {"ticks": sum(per_tc), "per_tc": per_tc, "error": None}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def gem5_time_batch(tasks: dict[str, dict], *, num_workers: int | None = None,
                    batch_timeout: float | None = None) -> dict[str, dict]:
    """Time a batch of unique programs in parallel.

    tasks: {key: {"code": str, "inputs": [input_str, ...]}}
    Returns {key: result-of-gem5_time_one}; every input key is present.

    One core per in-flight program (compile + 2 sequential gem5 processes are
    all single-threaded). Cores come from PIE_GEM5_CPUSET / PIE_REWARD_CPUSET
    (tail-first, away from where Ray/vLLM workers land) or the affinity mask.
    Worker default is conservative (half the cores, max 64) — each gem5 is
    ~0.5-1.5 GB RSS and this runs on the training node.

    batch_timeout (default PIE_GEM5_BATCH_TIMEOUT=1200s) is a wall cap on the
    whole call: unfinished programs return {"ticks": None, "error": "gem5
    batch timeout"} instead of stalling the RL step (deferred-fix-F5 analog).
    Their in-flight gem5 subprocesses still die via their own per-phase
    process-group timeouts.
    """
    if not tasks:
        return {}
    cores = _core_pool_cores()
    env_workers = os.environ.get("PIE_GEM5_WORKERS")
    default_n = max(1, min(len(cores) // 2 or 1, 64))
    n = max(1, min(num_workers or (int(env_workers) if env_workers else default_n),
                   len(cores), len(tasks)))
    pool: queue.Queue = queue.Queue()
    for c in cores[-n:]:                 # tail cores: first cores host Ray/vLLM
        pool.put(c)
    root = _scratch_root()
    if batch_timeout is None:
        batch_timeout = _BATCH_TIMEOUT

    def _one(item: tuple[str, dict]) -> tuple[str, dict]:
        key, t = item
        cpu = pool.get()
        try:
            return key, gem5_time_one(t["code"], t["inputs"], cpu=cpu, scratch=root,
                                      max_tc_ticks=t.get("max_tc_ticks"))
        except Exception as e:  # never poison the batch
            return key, {"ticks": None, "per_tc": None,
                         "error": f"gem5 backend error: {type(e).__name__}: {e}"}
        finally:
            pool.put(cpu)

    out: dict[str, dict] = {}
    ex = ThreadPoolExecutor(max_workers=n)
    try:
        futs = {ex.submit(_one, item): item[0] for item in tasks.items()}
        done, not_done = cf_wait(futs, timeout=batch_timeout)
        for f in done:
            key, res = f.result()
            out[key] = res
        for f in not_done:
            f.cancel()
            out[futs[f]] = {"ticks": None, "per_tc": None,
                            "error": "gem5 batch timeout"}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return out
