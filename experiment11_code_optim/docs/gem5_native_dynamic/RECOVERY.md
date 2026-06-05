# RECOVERY manifest — native-x86 gem5 exact PIE reward (2026-06-05)

Everything needed to recover / rebuild / re-run the exact-gem5 PIE reward. **Durable** = survives node/session
end (committed to git in this repo, or on the persistent `/work/nvme` Lustre). **Ephemeral** = could be wiped
(the `build-temp` rootfs, `/tmp`, `/dev/shm`) — rebuild steps given.

## A. DURABLE — in this git repo (`experiment11_code_optim/`)
| Artifact | Path |
|---|---|
| Recipe + scale results (read first) | `docs/gem5_native_dynamic/README.md` |
| gem5 source patches (git diff) | `docs/gem5_native_dynamic/native_dynamic.patch` |
| Speed-options analysis | `docs/gem5_native_dynamic/speedup_options.md` |
| Startup-amortization (#41) findings | `docs/gem5_native_dynamic/startup_amortization_findings.md` |
| Scale plot + raw summary | `docs/gem5_native_dynamic/scale.png`, `scale_summary.json` |
| 5-problem validation | `docs/gem5_native_dynamic/validate5.json` |
| **Skylake configs (copied out of the rootfs!)** | `configs/gem5-skylake-config/{run-se.py,run_se_redirect.py,run_se_ckpt.py,system/*.py}` |
| Validate harness (Step 3) | `scripts/reward_analysis/native_dynamic_validate.py` |
| Scale harness (Step 4) | `scripts/reward_analysis/native_dynamic_scale.py` |
| Plot harness | `scripts/reward_analysis/plot_scale.py` |

## B. DURABLE — on `/work/nvme` (persistent Lustre, NOT in git: too large)
| Artifact | Path | Recreate if lost |
|---|---|---|
| **Patched native gem5 (.opt, 1.2 GB)** | `/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.opt` | rebuild (§D) |
| **gem5.fast (20 MB, exact, 1.25×)** | `/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.fast` | rebuild (§D) |
| gem5 SOURCE (patched) | `/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/src` | apply `native_dynamic.patch` |
| build env (py3.8+scons<4) | `/work/nvme/bgtv/sramasubramanian/envs/gem5build` | §D |
| merged test cases (97 MB, PIE paper set) | `/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases` | gdown id 1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr |
| scale outputs (ticks.json, summary) | `/work/nvme/bgtv/sramasubramanian/gem5_scale_out/` | re-run scale harness |
| qemu10 + proot (compile only) | `/work/nvme/bgtv/sramasubramanian/qemu_x86/{qu10,proot}` | Debian arm64 debs (see [[project_gem5_build]]) |
| dataset (test=978 with tc2time) | `experiment11_code_optim/data_pie/{test,val,train}.jsonl` | (in repo path, large) |

## C. EPHEMERAL — the rootfs (`/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs`)
Contains PIE's x86 gem5 v20.1, **g++-9**, x86 libs, the Skylake configs, and `generated_test_cases`. It's a
`build-temp` dir → **could be cleaned**. The CONFIGS are now backed up in `configs/gem5-skylake-config/` (§A).
To rebuild the rootfs: `apptainer build --sandbox <dest> docker://alexshypula/gem5-skylake:api` then apply the
fixes in memory `project_gem5_build` (4KB-aligned libmpc swap; relative `lib64/ld-linux-x86-64.so.2` symlink:
`ln -sf ../lib/x86_64-linux-gnu/ld-2.31.so <rootfs>/lib64/ld-linux-x86-64.so.2`). Re-copy the backed-up configs
into `<rootfs>/home/gem5-skylake-config/gem5-configs/`.

## D. REBUILD gem5 (patched, from source)
```
export PATH=/work/nvme/bgtv/sramasubramanian/envs/gem5build/bin:/opt/cray/pe/gcc-native/12/bin:$PATH
export LD_LIBRARY_PATH=/work/nvme/bgtv/sramasubramanian/envs/gem5build/lib:$LD_LIBRARY_PATH
cd /work/nvme/bgtv/sramasubramanian/gem5_v20_x86
# (if source lost) re-apply: git apply <repo>/docs/gem5_native_dynamic/native_dynamic.patch
scons build/X86/gem5.fast -j64 CC=/opt/cray/pe/gcc-native/12/bin/gcc CXX=/opt/cray/pe/gcc-native/12/bin/g++ \
  --without-tcmalloc CCFLAGS_EXTRA='-Wno-error' PYTHON_CONFIG=python3.8-config
```
The 3 patches: `src/sim/vma.cc` (mmap→calloc+pread), `src/sim/syscall_emul.hh` + `src/base/loader/elf_object.cc`
(access()-guard debug object loads). Build env recipe: memory `project_gem5_build`.

## E. RUN the reward (one eval)
```
RF=<rootfs>; G=/work/nvme/bgtv/sramasubramanian/gem5_v20_x86/build/X86/gem5.fast
CFG=$RF/home/gem5-skylake-config/gem5-configs   # or configs/gem5-skylake-config (repo copy)
# compile candidate DYNAMIC with rootfs interp baked in (g++-9 under qemu+proot):
#   g++ src.cpp -o bin -O3 -std=c++17 -Wl,--dynamic-linker=$RF/lib/x86_64-linux-gnu/ld-2.31.so
LD_LIBRARY_PATH=/work/nvme/bgtv/sramasubramanian/envs/gem5build/lib PYTHONPATH=$CFG:$CFG/system \
  $G -d OUT --stats-file=s.txt $CFG/run_se_redirect.py Verbatim <bin> < <input.txt>
# reward = sum over test cases of sim_ticks/1e12 ; speedup = src_agg/tgt_agg
```

## F. Memory (durable, cross-session)
`~/.claude/projects/.../memory/`: `project_gem5_build.md` (build + native-dynamic + scale + speed roadmap),
`project_gem5_reward_startup_artifact.md` (the startup-artifact analysis). Index in `MEMORY.md`.
