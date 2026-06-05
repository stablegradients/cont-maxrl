# Startup-cost amortization (#41) — prototype findings (2026-06-05)

**Goal:** simulate the input-independent dynamic-linker+libc startup (99.2% of total ticks) ONCE per
candidate, reuse per test case → ~27–56× at K≈100. Config: `run_se_ckpt.py` (file-backed stdin + ckpt/restore).

## What's PROVEN (the building blocks work)
- **File-backed stdin is bit-exact.** `Process(input=<file>)` gives sim_ticks **1,029,044,588** — *identical*
  to inherited-stdin (the dataset-matching value). So the config change required for ckpt/restore does NOT
  change the metric. ✅
- **Checkpoint create + restore mechanics function** (write to `/dev/shm`, m5.cpt + physmem stores, restore
  loads). gem5 v20.1 exposes `m5.checkpoint(dir)` (drains internally) and `m5.fork(...)`.

## The BLOCKERS (why naive checkpoint-restore isn't the answer)
1. **Input doesn't replay across an O3 restore.** Checkpoint taken mid-dynamic-linker (tick 5.06e8), restore
   → program ran only ~9.3e6 more ticks and exited at 5.16e8 (vs full 1.03e9) — it got no/garbage input and
   quit early. The file-backed stdin fd state isn't correctly re-established across the restore (needs
   `FDArray::restoreFileOffsets` handling + checkpoint at a CLEAN point, i.e. main()-entry after the linker
   finishes, not mid-relocation).
2. **`m5.checkpoint()` perturbs ticks +0.72%** — `memWriteback(root)` flushes caches at the checkpoint, so the
   continuation re-warms cold (within the 2% gate, but not bit-exact).
3. **Disk serialize is slow** — 33 MB physmem/checkpoint; on lustre it timed out (2 min), /dev/shm fixes the
   write but the cold continuation is still slow when the post-checkpoint region is large.

## Conclusion → fork-from-live is the right path (confirms the workflow)
`m5.fork()` (COW child per test case from the post-startup parent) **avoids all three blockers**: no
serialize (no 33 MB write), no `memWriteback` (warm caches inherited → near-bit-exact, only ~1e-4 drain
bubble), and the child continues the live process. The remaining hard part is **per-child input**: fork
before the program reads stdin, then point each child at a different test-case file. Needs:
- checkpoint/fork at **main()-entry** (clean state) — trigger via a PC breakpoint on the `main` ELF symbol, or
  exit-on-first-`read(0)` syscall hook, not a fixed tick.
- per-child stdin swap (re-point fd 0 to the tc file in each child before it reads).
- validate exactness on an **input-DEPENDENT** program (p02552 is startup-only, so it can't show the per-tc
  amortization; pick a program whose tc ticks vary).

## UPDATE: fork-from-live VALIDATED (works, <2%)
`run_se_fork.py` (`GEM5_FORK_TICK=n` → run to n, `m5.fork()`, child continues to exit). On p02552:
- **full (no fork) = 1,029,044,588**; **fork@5e8 → child = 1,036,351,030 (+0.71%)**, child output **'1' (correct)**,
  clean exit, 9s wall. UNLIKE checkpoint-restore (which corrupted the run / no output), **fork works** — the
  COW child runs to completion correctly. The +0.71% is the `drain()` cache-flush at the fork point (same as
  checkpoint's memWriteback) — **within the 2% gate**.
- `m5.fork()` = drain → `os.fork()` → child gets new outdir; returns child-pid (parent) / 0 (child); requires
  `m5.disableAllListeners()` first. No serialize, no disk, warm memory (COW).

REMAINING to productionize (all now well-defined):
1. **Pre-input fork trigger** — fork right after startup, before the program reads stdin (so children can be
   fed different inputs). Robust trigger = exit-on-first-`read(0)` syscall (small gem5 hook) or the per-binary
   startup tick (deterministic). Forking LATE (small post-fork region) should also SHRINK the +0.71% (the cold
   region is just the tiny algorithm, not 5e8 of startup).
2. **Per-child input** — fixed input path with swapped CONTENT per child (gem5 reads fd0 lazily from the host
   file at read time, validated), forking children SEQUENTIALLY (parent waits via os.waitpid, re-forks from the
   same pre-input state). No FD-repointing needed.
3. **Speedup-exactness validation** on INPUT-DEPENDENT programs — confirm the +0.71% drain perturbation stays
   <2% on the algorithm region and (partially) cancels in src/tgt ratios.

Configs: `run_se_fork.py`, `run_se_ckpt.py` (both backed up in repo `configs/gem5-skylake-config/`).

## Practical recommendation (now)
- **Ship gem5.fast (done, 1.25×, exact) + K=5 subsampling → ~3 min/step.** That's already a usable in-loop
  reward and unblocks Step 5 (RL integration).
- Treat full fork-from-live (the 27–56× all-tc win) as a **scoped follow-up** — foundations proven here, but
  it needs the main()-entry trigger + per-child-stdin + exactness validation on input-dependent programs
  (est. several hours of focused gem5 FD/fork work).
