# PROBE_REPORT_partial.md — §14 step 3 (item 0) outcomes

**Date:** 2026-05-19
**Compute node:** `gh152.hsn.cm.delta.internal.ncsa.edu`

This file captures spec §13 items 0–2 only. Items 3 (determinism), 4 (throughput), 5 (precompute extrapolation), 6 (step-budget) are deferred to **§14 step 5** where the calibration script runs against the to-be-built C++ measurement core.

## Item 0 — Runtime boundary (architecture decision)

**Outcome: CONDA. §4.0 primary architecture applies (in-process `ThreadPoolExecutor`, no `/local` handoff, no Appendix-A pool lifecycle).**

Evidence:
- `env | grep -iE '^(SIF|SINGULARITY|APPTAINER|CONTAINER)'` → empty
- Active interpreter: `/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python` (Python 3.10.19)
- The DeepCoder live scripts at `experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_{maxrl,grpo}_g16.sh:23-25` use `module load python/miniforge3_pytorch/2.10.0` + `conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl` (no SIF wrapping)

## Item 1 — `perf_event_paranoid`

**Outcome: GREEN (`perf` as primary backend).**

- `/proc/sys/kernel/perf_event_paranoid` = `2` — user-process hardware counters allowed; only kernel/all-CPU profiling restricted. PIE uses user-process counters → permitted.
- `perf stat -e instructions /bin/true` returned `281,447 instructions:u` cleanly.
- Backend selection at measurement-core init: `perf` is primary; PAPI 7.2.0.2 (`papi/7.2.0.2`) remains the documented fallback per spec §1.

## Item 2 — `g++` version

**Outcome: GREEN (spec-mandated path resolves to 14.2.0).**

- `/opt/cray/pe/gcc-native/14/bin/g++` → symlink → `/usr/bin/g++-14`
- `g++ --version` → `g++ (SUSE Linux) 14.2.0`
- Spec §1 mandates this absolute path with `-O3 -std=c++17`; measurement core uses it verbatim.

## Hardware (recorded for §13 item 4 + measurement-core NUMA pinning)

- Architecture: aarch64, ARM Neoverse-V2 (GH200 grace-hopper)
- 288 cores total: 4 sockets × 72 cores × 1 thread
- 4 NUMA nodes with CPUs: node 0 (cores 0–71), node 1 (72–143), node 2 (144–215), node 3 (216–287). Nodes 4–35 are CPU-less memory regions (HBM/expanded RAM)
- Per spec §1: policy/vLLM GPU0 → NUMA1; **reward compute on NUMA0 (cores 0–71)**. The measurement core pins counted child processes to NUMA-0 cores via `numactl -N 0 -C <core>` and uses a per-core lock on the counted run.

## `/local` scratch

- `/local` exists, 3.5 TB NVMe, 3.5 TB free on `/dev/nvme0n1`. Available as the spec-mandated compile scratch for the measurement core's per-record tempdirs.

## Tools verified

- `numactl`, `taskset`, `setarch -R` (ASLR disable) — verified available
- `perf` at `/usr/bin/perf`
- `g++` at `/opt/cray/pe/gcc-native/14/bin/g++`

## What this gates

- §4.0 PIE architecture: **primary** (in-process). Appendix-A (`/local` filesystem handoff) is **eliminated**.
- The measurement core uses `subprocess.run` + module-level `ThreadPoolExecutor`, mirroring the structure proven in `experiment4_llm/src/reward_function.py` (line 268, real concurrency; ignore the stale "multiprocessing.Pool" docstring at line 288).
- Compile + gate are parallel; counted (perf/PAPI) runs serialize per physical core via a per-core lock.

## Deferred to §14 step 5

- Item 3: determinism check (run fixed C++ program ×20, mean/std/max-min; require jitter < inter-ref gaps)
- Item 4: throughput (one G=16 batch through chosen path, 64 workers, counted serialized per core)
- Item 5: precompute extrapolation (20-problem calibration)
- Item 6: step-budget vs DeepCoder ~7 min step; project total wall-clock vs computed `total_training_steps=3050`
- Final GO / GO-WITH-MITIGATION / NO-GO decision in `PROBE_REPORT.md` (this file is the partial precursor)
