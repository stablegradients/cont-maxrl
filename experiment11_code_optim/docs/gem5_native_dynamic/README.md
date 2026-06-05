# Native-x86 gem5: PIE's exact gem5 timing, fast, no qemu

**Goal.** Compute PIE's *exact* gem5 timing metric (the one used in the ICLR-2024 PIE paper and dataset) on
our aarch64 DeltaAI node, fast enough for in-loop RL, replacing the broken retired-instruction proxy.

**Outcome (2026-06-05).** Native-x86 gem5 (an aarch64 binary that simulates x86, **no qemu**) now runs PIE's
**dynamic** binaries and reproduces the dataset gem5 ticks. Single-eval: native-dynamic p02552
`sim_ticks = 1,029,044,588` vs PIE/qemu `1,026,675,364` = **0.23%** (glibc-IFUNC init offset; cancels in
speedup ratios). 5-problem validation (merged test cases): **4/4 within 2%**, K_speedup 0.9875–0.9984.
Speed: **~11 s/eval** (dynamic startup dominates the *simulated* work) vs qemu's 27 s, and **256-way parallel**.

---

## The metric (verified against PIE's source — github.com/LearningOpt/pie)

- `agg_runtime` = **SUM over the test cases of** `sim_seconds_precise` (= `sim_ticks / sim_freq`,
  `sim_freq = 1e12`) — `gem5/simulator.py:266`.
- `speedup = src_agg_runtime / tgt_agg_runtime` (`gem5_eval.py:196`); verified `speedup == src_agg/tgt_agg`
  with **0% mismatch on 5000 dataset rows**.
- Compile flags: **`g++-9 -O3 -std=c++17`, NO `-static`** (PIE builds **dynamic** binaries; `benchmarking.py:53`).
- gem5 config: the **Verbatim** Intel-Skylake config (`alexshypula/gem5-skylake:api` image), gem5 **v20.1.0.2**.
- Test set: the **merged** (public+generated) test cases — PIE's paper set, median ~104 tc/problem.
  *Caveat:* on the small CodeNet inputs the gem5 time is largely **binary-startup** (see
  `memory/project_gem5_reward_startup_artifact.md`); this faithfully reproduces PIE's metric, artifact and all.

## The pipeline

1. **Compile dynamic** with PIE's g++-9 under qemu+proot, baking a host-resolvable interpreter:
   `g++ -O3 -std=c++17 -Wl,--dynamic-linker=<rootfs>/lib/x86_64-linux-gnu/ld-2.31.so` (gem5 opens PT_INTERP
   raw at ELF-parse, before redirect_paths, so it must be a host path, not the dangling `/lib64/ld-linux`).
2. **Run native gem5** with `run_se_redirect.py` (sets `system.redirect_paths` mapping `/lib64`, `/lib/*`,
   `/usr/lib/*` into the rootfs so the simulated loader resolves libc/libstdc++ during simulation):
   ```
   LD_LIBRARY_PATH=<gem5build>/lib PYTHONPATH=<CFG>:<CFG>/system \
     gem5.opt -d OUT --stats-file=s.txt <CFG>/run_se_redirect.py Verbatim <dyn_binary> < input.N.txt
   ```
3. **Aggregate:** `agg = Σ_tc sim_ticks/1e12`; `speedup = src_agg/tgt_agg`. 2-min/test timeout → `inf`
   (worst reward), never silently dropped.

## The three gem5 source patches (why native+dynamic was blocked, and the fix)

The native build was previously **static-only**. Three issues blocked dynamic binaries on the **64 KB-page**
Grace host; all fixed in `/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/src` (rebuilt with the py3.8 gem5build env):

| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `sim/vma.cc` (`MappedFileBuffer`) | `mmap(...,offset)` needs a **host-page-aligned** offset; x86 ELF segments are 4 KB-aligned → `EINVAL` on the 64 KB host | replace `mmap` with `calloc`+`pread` (buffer is read-only, copied by `fillMemPages` → semantically identical, no alignment requirement) |
| 2 | `sim/syscall_emul.hh` (mmapImpl debug-symbol block) | loads each lib the loader mmaps via `createObjectFile`, which **fatally panics** if the path doesn't resolve on the host | guard with `access(R_OK)` (debug symbol table only — zero timing effect) |
| 3 | `base/loader/elf_object.cc` (PT_INTERP) | a runtime-loaded lib's `PT_INTERP=/lib64/ld-linux` can't resolve on host → panic | guard `createObjectFile(interp_path)` with `access(R_OK)` |

Plus two non-source fixes: bake the host interp path (above), and make the dangling
`rootfs/lib64/ld-linux-x86-64.so.2` symlink **relative** (`ln -sf ../lib/x86_64-linux-gnu/ld-2.31.so`).

## Key paths

| | |
|---|---|
| Patched native gem5 (no qemu) | `/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.opt` |
| gem5 source (patched) | `/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/src/{sim/vma.cc,sim/syscall_emul.hh,base/loader/elf_object.cc}` |
| build env (py3.8+scons<4) | `/work/nvme/bgtv/sramasubramanian/envs/gem5build` |
| rootfs (Skylake cfg, g++-9, x86 libs) | `/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs` |
| Skylake redirect config | `<rootfs>/home/gem5-skylake-config/gem5-configs/run_se_redirect.py` |
| merged test cases | `/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases` |
| dataset (test=978, with tc2time) | `experiment11_code_optim/data_pie/{test,val,train}.jsonl` |
| qemu+proot (compile only) | `/work/nvme/bgtv/sramasubramanian/qemu_x86/{qu10,proot}` |
| validate harness (Step 3) | `experiment11_code_optim/scripts/reward_analysis/native_dynamic_validate.py` |
| scale harness (Step 4) | `experiment11_code_optim/scripts/reward_analysis/native_dynamic_scale.py` |

## To rebuild gem5 after editing a source patch
```
export PATH=/work/nvme/bgtv/sramasubramanian/envs/gem5build/bin:/opt/cray/pe/gcc-native/12/bin:$PATH
export LD_LIBRARY_PATH=/work/nvme/bgtv/sramasubramanian/envs/gem5build/lib:$LD_LIBRARY_PATH
cd /work/nvme/bgtv/sramasubramanian/gem5_v20_x86
scons build/X86/gem5.opt -j64 CC=/opt/cray/pe/gcc-native/12/bin/gcc CXX=/opt/cray/pe/gcc-native/12/bin/g++ \
  --without-tcmalloc CCFLAGS_EXTRA='-Wno-error' PYTHON_CONFIG=python3.8-config
```

## Scale result (Step 4) — full 978-pair PIE test set, 256 cores (2026-06-05)

| metric | value |
|---|---|
| pairs / unique binaries | 978 / 1454 — **1452 compiled** (2 transient cc1plus fails, 145 retries succeeded) |
| **absolute** K = our_agg/dataset_agg | median **1.0026**, **within 2% = 99.95%** (1950 program-sides) |
| **speedup** K = our_sp/dataset_sp | median **0.9977**, **within 2% = 99.9%** (974 pairs); 99% in [0.987, 1.003] |
| **determinism** (serial == 256-way parallel ticks) | **60/60, 0 mismatches** — sim_ticks parallelism-invariant |
| output-correct | **99.4%** (1 timeout in 129k evals, ~0.6% SE-emulation gaps) |
| throughput | **23.2 evals/s** on 256 cores |
| per-step (1024 rollouts) | K=5: **3.7 min** · K=10: 7.4 min · all-100-tc: 73.7 min |

Single speedup outlier: **p02802** (K_sp=0.88 — a genuine per-program SE-emulation divergence, not a build/harness bug; no timeout, no output error). Output-mismatches don't bias the match (clean pairs K_sp 0.9978 vs with-mismatch 0.9929). Plots: `scale.png`. Raw: `gem5_scale_out/scale_summary.json`.

**Verdict:** native-dynamic gem5 reproduces PIE's exact dataset metric across the full test set to <2% on 99.9% of programs, with perfect determinism at 256-way parallelism — production-ready as the RL reward (pending the in-loop integration, Step 5).

## Speed roadmap (verified — see `speedup_options.md`)
- **gem5.fast + tcmalloc** (#40): ~1.25–1.7×, exact ticks. Gate: `.opt`-vs-`.fast` tick byte-diff on p02552.
- **Startup-cost amortization / fork-from-live** (#41): 99.2% of total ticks is input-independent startup → ~27–56× at K≈100; <1% cold-cache error on our ~1e9-floor dynamic binaries (within the 2% gate). Needs file-backed-stdin config copy + calibration.
- **Native x86 g++-9 cross-compiler** (#42): 10–20× faster candidate compile (must stay g++-9).
- All-opts per-step (all-100-tc × 1024) → **~1–6 min/step**.
