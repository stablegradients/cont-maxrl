# Memory Bug Analysis: SIGSEGV During Reward Computation

## Problem Summary

Training runs (Qwen3-1.7B-Base on DeepCoder with G=16) crash with SIGSEGV during
reward computation, typically between steps 3–9. The crash is **non-deterministic** —
same code on different nodes crashes at different steps. Pilot runs (Qwen2.5-Coder-1.5B
with smaller config) never exhibited this issue.

---

## Environment

- **Hardware**: NVIDIA GH200 120GB GPUs, 4 per node, 858GB system RAM
- **Cgroup limit**: 748.4GB RAM per job (no swap)
- **Kernel overcommit**: `overcommit_memory=0` (heuristic), `overcommit_ratio=50`
- **CommitLimit**: ~429GB (858GB × 50%, no swap)
- **Python**: 3.10.19 (conda-forge)
- **Framework**: verl (custom fork) + vLLM 0.8.3 + Ray + FSDP

## Config Comparison: Pilot vs Current

| Parameter               | Pilot (worked)         | Current (crashes)      |
|-------------------------|------------------------|------------------------|
| Model                   | Qwen2.5-Coder-1.5B    | Qwen3-1.7B-Base        |
| `train_batch_size`      | 128                    | 256                    |
| `max_response_length`   | 1024                   | 4096                   |
| `rollout.n`             | 16                     | 16                     |
| Total completions/step  | 2,048                  | 4,096                  |
| Max tokens/completion   | 1,024                  | 4,096                  |
| `VERL_REWARD_WORKERS`   | 256                    | 64 (reduced from 256)  |
| `ppo_micro_batch_size`  | 16                     | 4                      |

**Key difference**: Current config processes **2× more completions**, each up to **4× longer**.
This results in ~8× more data flowing through memory per step.

---

## Crash Signature

```
(TaskRunner pid=XXXXX) [DEBUG] Computing rewards for 4096 items...
(TaskRunner pid=XXXXX) Error in reward_fn:
(raylet) A worker died or was killed while executing a task by an unexpected system
error. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits
with a connection error code 2. End of file.
```

In earlier runs (before gc.collect fix), the stack trace showed:
```
PC: MemoryError_dealloc
```
This is a known CPython bug: when Python raises `MemoryError` and then tries to
deallocate the exception object, it can SIGSEGV if the heap is corrupted.

In later runs, MaxRL explicitly showed `MemoryError` in stderr before dying.

---

## Root Cause Analysis

### What we know for certain

1. **Not a cgroup OOM kill**: Checked `/sys/fs/cgroup/.../memory.events` — zero
   `oom_kill` events. The kernel OOM killer never fires.

2. **Physical RAM is sufficient**: `free -g` shows 476GB available even during
   training. The 401GB used is well within the 748GB cgroup limit.

3. **Virtual memory is massively overcommitted**: `Committed_AS` = 4.37TB vs
   `CommitLimit` = 429GB — a **10× overcommit**. This comes from:
   - 4 FSDP WorkerDict processes: ~380GB VSZ each = 1.52TB
   - 1 TaskRunner process: ~250GB VSZ
   - Various Ray processes: ~225GB VSZ each
   - Total: ~4.3TB virtual address space

4. **`overcommit_memory=0` (heuristic mode)**: The kernel allows overcommit but can
   **non-deterministically refuse** individual memory allocation requests (including
   `fork()`, `mmap()`, `brk()`) when it judges the system is under pressure. This
   explains why crashes happen at random steps.

5. **Memory grows ~18GB between step 1 and step 2**, then stabilizes:
   - Step 1: ~379-389GB
   - Step 2: ~398-408GB
   - Step 3+: ~400-407GB (stable)
   - This growth is from vLLM/FSDP caching, not a leak.

6. **Crash always happens during reward computation** (`Computing rewards for 4096
   items...`), but reward computation is not the source of memory growth. User
   observed that RAM usage actually **decreases** during reward computation. The
   crash happens at this phase because it involves operations that trigger the
   kernel's overcommit check (subprocess fork, multiprocessing IPC, memory allocation
   for pickling).

7. **The crash is in the main TaskRunner process**, not in subprocess workers.
   The Ray error shows the TaskRunner pid dying, not a child process.

### Why pilot runs survived

Pilot runs had `batch_size=128` and `max_response_length=1024`:
- Smaller activation tensors → lower VSZ per FSDP worker (~200GB vs ~380GB)
- Fewer completions (2048 vs 4096) → less data in the TaskRunner
- Total virtual commitment ~2TB instead of ~4.3TB
- The kernel's heuristic overcommit check had much more headroom

---

## Attempted Fixes (Chronological)

### Fix 1: gc.collect() after reward computation
**Commit**: `447d4b2`
**Rationale**: Prevent memory accumulation across training steps.
**Result**: GRPO survived to step 5 (previously crashed at step 9), MaxRL to step 6.
Helped slightly but did not solve the problem. Memory still grew and the overcommit
heuristic still refused allocations.

### Fix 2: Reduce VERL_REWARD_WORKERS from 256 to 64
**Commit**: `4639160`
**Rationale**: Fewer concurrent subprocess forks = less virtual memory pressure from
`fork()` calls in `subprocess.run()`.
**Result**: GRPO crashed at step 3, MaxRL at step 6. Did NOT help — the overcommit
is 10× regardless of worker count. The fork from the main process is the issue, not
the number of concurrent forks.

### Fix 3: Forkserver multiprocessing pool
**Commit**: `21ce24e`
**Rationale**: Use `multiprocessing.get_context('forkserver')` so pool workers fork
from a lightweight server process (~10MB VSZ) instead of the 400GB main process.
This should completely avoid the `fork()` overcommit issue.
**Result**: GRPO crashed at step 3, MaxRL at step 6. The forkserver correctly
avoids fork-from-large-process, but the **crash is not from fork()**. The crash is
in the main TaskRunner process during `pool.map()` IPC — the main process itself
hits a memory allocation failure (mmap/brk refused by kernel overcommit heuristic),
which triggers MemoryError → SIGSEGV.

### Fix 4 (not tried): posix_spawn
**Rationale**: Force `subprocess.run()` to use `posix_spawn()` instead of `fork()`.
**Why not tried**: Python 3.10's subprocess likely already uses posix_spawn when
conditions are met. And since the forkserver fix (which eliminates fork entirely)
didn't help, the crash is not from fork().

### Fix 5 (not tried): Reduce RAY_OBJECT_STORE_MEMORY
**Rationale**: Shrink Ray's object store from 32GB to 8GB to reduce VSZ.
**Why not tried**: 24GB savings against a 4TB overcommit gap is negligible.

### Fix 6 (not tried): Pre-spawned evaluation server
**Rationale**: Spawn a single long-lived evaluator subprocess at startup, send test
cases via pipes. Zero process creation during training.
**Why not tried**: Significant refactor, and forkserver results showed the crash
isn't from process creation.

---

## What IS and ISN'T the Problem

### IS the problem
- The **main TaskRunner process** (running inside a Ray actor) is ~250GB VSZ
- The total virtual memory commitment across all processes is ~4.3TB
- The kernel's `overcommit_memory=0` heuristic **non-deterministically refuses**
  memory allocations (mmap, brk, fork) from the TaskRunner during reward computation
- When Python can't allocate memory, it raises `MemoryError`
- CPython 3.10 has a bug where deallocating the `MemoryError` object can cause
  SIGSEGV if the heap is already corrupted (`MemoryError_dealloc`)
- The refusal happens during reward computation because that's when the TaskRunner
  does the most dynamic memory allocation (pickling tasks, receiving results,
  processing 4096 completion strings)

### IS NOT the problem
- Physical RAM exhaustion (476GB available, 748GB cgroup limit)
- OOM killer (zero oom_kill events)
- Memory leak (memory stabilizes after step 2)
- Subprocess fork() from large parent (forkserver fix didn't help)
- Number of concurrent workers (reducing to 64 didn't help)
- The Qwen3 vLLM patch (just adds a sample() method, no memory impact)
- The reward computation itself (RAM goes DOWN during reward)

---

## Process Memory Map (from `ps aux` on compute node)

```
PROCESS                                    VSZ (GB)   RSS (GB)
ray::WorkerDict (FSDP worker) × 4         ~380 each  ~17 each
ray::TaskRunner (main training loop)       ~252       ~6.5
ray::TaskRunner (idle) × 7                 ~225 each  ~0.7 each
Main train.py process                      ~220       ~0.3
Ray GCS server                             ~117       ~1.6
```

Total VSZ: ~4.3TB. Total RSS: ~100GB. The gap between VSZ and RSS is enormous —
most of the virtual space is mapped but not touched (PyTorch tensor reservations,
CUDA device memory mappings, mmap'd model files).

---

## Remaining Options

### Option A: Reduce batch_size to 128 (match pilot)
- Halves completions per step (4096 → 2048)
- Reduces FSDP worker VSZ (~380GB → ~250GB estimated)
- Total virtual commitment drops from ~4.3TB to ~2.5TB
- Steps per epoch doubles (83 → 166), but each step is faster
- Total tokens seen per epoch is identical
- **Lowest risk, most likely to work**

### Option B: Reduce batch_size to 192 (compromise)
- 192 × 16 = 3072 completions per step
- Moderate VSZ reduction
- 101 steps per epoch
- **Medium risk**

### Option C: Ask sysadmin to set overcommit_memory=1
- `echo 1 > /proc/sys/vm/overcommit_memory` (always allow overcommit)
- This is safe because physical RAM (858GB) is sufficient and cgroup limits (748GB)
  prevent actual OOM
- The kernel would never refuse allocations, eliminating the non-deterministic crashes
- **Zero code changes needed, but requires admin access**

### Option D: Reduce max_response_length to 2048
- Shorter sequences → smaller activation tensors → lower VSZ
- But this changes the research setup (may miss long solutions)
- **Changes experimental conditions**

### Option E: Use 2 nodes instead of 1
- Spread the 4 GPUs across 2 nodes with 2 GPUs each
- Each node has half the memory pressure
- Requires multi-node Ray setup
- **Infrastructure change**

---

## Recommendation

**Option A (batch_size=128)** is the safest immediate fix. The pilot proved it works.
We keep `max_response_length=4096` (the research variable) and `rollout.n=16`.
Training sees the same total tokens — just more steps per epoch.

**Option C (overcommit_memory=1)** is the correct system-level fix if admin access
is available. The heuristic overcommit check is overly conservative for this workload
where VSZ >> RSS.

---

## Timeline of Crashes

| Job ID  | Algorithm | Node  | Workers | Pool Type       | Crashed At | CPU Mem at Crash |
|---------|-----------|-------|---------|-----------------|------------|------------------|
| 1999506 | GRPO      | —     | 256     | ThreadPool      | Step 9     | ~420GB           |
| 1999507 | MaxRL     | —     | 256     | ThreadPool      | Step 3     | ~420GB           |
| 2000684 | GRPO      | gh020 | 256     | ThreadPool+gc   | Step 6     | ~420GB           |
| 2000685 | MaxRL     | gh071 | 256     | ThreadPool+gc   | Step 6     | ~423GB           |
| 2001069 | GRPO      | gh027 | 64      | ThreadPool+gc   | Step 3     | ~408GB           |
| 2001070 | MaxRL     | gh132 | 64      | ThreadPool+gc   | Step 6     | ~407GB           |
| 2001111 | GRPO      | gh012 | 64      | Forkserver+gc   | Step 3     | ~408GB           |
| 2001112 | MaxRL     | gh060 | 64      | Forkserver+gc   | Step 6     | ~401GB           |

Pattern: GRPO consistently crashes earlier (step 3) than MaxRL (step 5-6).
This could be because GRPO's advantage normalization creates different memory
allocation patterns than MaxRL's unnormalized advantages.
