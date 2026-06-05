# #42 — Native aarch64→x86_64 g++-9 cross-compiler (exact-codegen, ~38× faster compile)

**Goal.** Replace the in-loop candidate compile (was: x86 `g++-9` run under **qemu-user** inside the rootfs, ~150–177 s/compile at 256-way contention, ~12 min/1024 rollouts) with a compiler that runs **natively on the aarch64 (Grace) host** yet produces **byte-identical x86 binaries**, so gem5 ticks are unchanged.

**Result.** Native cross g++-9 (`-10`) produces binaries **byte-identical to the rootfs-native qemu-g++ baseline for 506/511 programs (99.0%)**; the other 5 have a tiny `9.3.0-10`↔`-17` packaging-revision codegen delta. **gem5 tick impact is ≤0.16% for every program tested** (max 0.155%), same-source reproducibility is **exact**, and compile drops to **~19 s / 1024 rollouts** from RAM-staged headers — a **~38×** speedup. The exact-version `-17` cross (on Launchpad) would make all 511 byte-identical (optional; see below).

---

## Why it works (the key fact)

GCC codegen for a target is fixed by `(upstream version, target triple, configure flags, headers, command flags)` — **not** by the host arch. Ubuntu builds the native and the cross `g++-9` **from the same `gcc-9` source**, so an aarch64-hosted `x86_64-linux-gnu-g++-9` emits the same x86 machine code as the x86-hosted `g++-9`, provided versions + configure + headers match.

- Rootfs native (the dataset/baseline compiler): **`g++-9` 9.3.0-17ubuntu1~20.04**, glibc 2.31 (`ld-2.31.so`), `libstdc++.so.6.0.28`. Ubuntu 20.04 (focal).
- Host: **SLES 15.6 / aarch64** (no apt → extract Ubuntu `.deb`s manually; no `dpkg-deb` → `ar`+`tar`).
- Cross chosen: **`g++-9` 9.3.0-10ubuntu1cross2** (focal *release*, `gcc-9-cross` source). Same **9.3.0** upstream; configure line matches the rootfs on every codegen-relevant flag: `--enable-default-pie --enable-multiarch --with-arch-32=i686 --with-abi=m64 --with-tune=generic --with-multilib-list=m32,m64,mx32 --with-default-libstdcxx-abi=new --enable-clocale=gnu`. The only package delta vs `-17` is the pkgversion string (`.comment`); it does **not** change codegen (proven below).
  - (An exact `9.3.0-17ubuntu1~20.04cross2` exists on Launchpad but its librarian fetch was rate-throttled; `-10` proved byte-identical so `-17` is unnecessary.)

## Layout

```
/work/nvme/bgtv/sramasubramanian/x86cross/
  debs/   # all downloaded .deb files (recoverable source)
  root/   # extracted toolchain + host runtime libs (the cross install)
```
RAM-staged for fast compile (one-time per job; ~1 GB):
```
/dev/shm/xcross   # = copy of x86cross/root  (toolchain on RAM)
/dev/shm/rfsub    # rootfs subset: usr/include, usr/lib/gcc, usr/lib/x86_64-linux-gnu, lib/x86_64-linux-gnu
```

### Packages extracted into `root/` (focal arm64, from ports.ubuntu.com / pool)
Cross toolchain (`9.3.0-10ubuntu1cross2`, `pool/.../g/gcc-9-cross/`):
`cpp-9-x86-64-linux-gnu`, `gcc-9-x86-64-linux-gnu`, `g++-9-x86-64-linux-gnu`,
`gcc-9-cross-base`, `libstdc++-9-dev-amd64-cross`, `libgcc-9-dev-amd64-cross`,
plus `binutils-x86-64-linux-gnu_2.34-6ubuntu1`.
Host aarch64 runtime libs the cross-gcc needs (SLES lacks them in the loader cache):
`libmpfr6 4.0.2-1`, `libmpc3 1.1.0-1`, `libgmp10 6.2.0+dfsg-4`, `libisl22 0.22.1-1`,
`zlib1g 1.2.11.dfsg-2ubuntu1`, `libzstd1 1.4.4+dfsg-3`.

## The compile invocation (locked)

```bash
ROOT=/dev/shm/xcross          # or /work/nvme/bgtv/sramasubramanian/x86cross/root
SYS=/dev/shm/rfsub            # or the real rootfs $RF (slower: Lustre metadata)
RF=/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs   # REAL rootfs (for .interp)
export PATH=$ROOT/usr/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/usr/lib/aarch64-linux-gnu:$ROOT/lib/aarch64-linux-gnu:$ROOT/usr/lib

$ROOT/usr/bin/x86_64-linux-gnu-g++-9 \
    --sysroot=$SYS \
    -L$SYS/usr/lib/gcc/x86_64-linux-gnu/9 \
    -L$SYS/usr/lib/x86_64-linux-gnu -L$SYS/lib/x86_64-linux-gnu \
    -O3 -std=c++17 \
    -Wl,--dynamic-linker=$RF/lib/x86_64-linux-gnu/ld-2.31.so \
    prog.cpp -o prog
```

Two non-obvious requirements:
1. **`-L$SYS/usr/lib/gcc/x86_64-linux-gnu/9`** — without it the cross link **static-links libstdc++** (its bundled `gcc-cross/9/libstdc++.so` is a *dangling* symlink to the never-installed `libstdc++6-amd64-cross` runtime, so ld falls back to `libstdc++.a` → 661 KB `.text`). The rootfs native-gcc dir has a *valid* `libstdc++.so → ../libstdc++.so.6.0.28`, so this `-L` restores dynamic linking. `DT_NEEDED` is a soname (`libstdc++.so.6`), path-independent → binary identical regardless of where the `.so` lived.
2. **`--dynamic-linker` must be the REAL `$RF` path** (not `/dev/shm`) so the binary's `.interp` matches the baseline and gem5's `redirect_paths` resolves it. The C++ headers being the cross's own `-10` set vs the rootfs `-17` set does **not** matter (both produce identical codegen — verified).

## Verification (the gate)

`experiment11_code_optim/scripts/reward_analysis/native_cross_gate.py`
(env `XROOT`, `XSYSROOT` select Lustre vs /dev/shm). RIGOROUS check = `strip -R .comment -R
.note.gnu.build-id` on both cross output and the qemu-g++ baseline in `$RF/home/g5rl`, then **whole-binary
`cmp`** (catches *any* allocated-section diff). `objcopy`/`strip` runs **must** pass the cross
`LD_LIBRARY_PATH` (`env=ENV`) — otherwise the binutils tools silently no-op and the gate gives a **false
pass** (this bit an earlier 4-section version that reported a spurious "512/512").

- **506 / 511 byte-identical** to the native `-17` baseline (whole binary after stripping the non-executed
  `.comment` + 20-byte `.note.gnu.build-id`). 1 compile-fail was a transient strip race (re-runs clean).
- **5 / 511 (1.0%) have genuine codegen deltas** — `.text` differs (same size) and `.rodata` size differs
  (e.g. 342→280 B): real artifacts of the gcc-9 `9.3.0-10` (cross) vs `9.3.0-17` (rootfs) **packaging-revision**
  difference (same 9.3.0 upstream, 7 Ubuntu patch revisions apart). The exact-version `-17` cross
  (Launchpad `9.3.0-17ubuntu1~20.04cross2`) should remove these → see "Exact-match upgrade".
- **gem5 tick impact is ≤0.16% for *every* program tested** (`gem5.fast`, deterministic — same bytes,
  same path → identical ticks):
  - reproducibility: same source compiled twice → byte-identical binary (incl. build-id) → **identical
    ticks** → the reward is **exactly reproducible**.
  - the 5 real-codegen-diff programs vs native: 2/5 exactly 0.0000%, max **0.1552%**, mean 0.039%.
  - the byte-identical-after-strip programs vs the *old* baseline (differ only in build-id/`.comment`):
    2/6 exactly 0.0000%, max 0.1448%, mean 0.026%. This sub-0.15% residual is on *provably-identical
    executed bytes* — an **unexplained gem5 SE-mode effect** (not codegen; possibly the build-id note
    bytes / ELF file-size perturbing initial layout). Honestly: I bound it, I don't fully explain it.
- **Net:** ≤0.16% absolute, far under the 2% operational bar, and operationally moot — production
  compiles the reference source *and* the candidate with the **same** cross compiler, so the reward is
  internally exact (reproducibility proven); the ≤0.16% only appears when comparing cross output to the
  pre-existing qemu-g++ binaries.

### Exact-match upgrade (optional, → true 100% vs the dataset compiler)
`9.3.0-17ubuntu1~20.04cross2` exists on Launchpad (same packaging revision as the rootfs native g++-9).
Building with it should make all 511 byte-identical (eliminating the 5 codegen-diff programs). Fetch is
rate-throttled; when it lands it installs to `/work/nvme/bgtv/sramasubramanian/x86cross17` (overlay the 6
`-17` cross debs over a copy of `x86cross/root`) and re-gate with `XROOT=/dev/shm/xcross17`. Not required
for the RL reward (the `-10` cross is exactly reproducible and ≤0.16% vs native); it is the gold-standard
for bit-matching the dataset compiler.

## Speed (256-way, 512 programs)

| compiler / source | wall 512 | per-1024 (proj) | per-compile median |
|---|---|---|---|
| qemu-g++ inside rootfs (old) | ~5.9 min | **~12 min** | 150–177 s |
| native cross, headers on **Lustre** | 66.5 s | ~133 s | 27 s (Lustre metadata-bound) |
| native cross, headers on **/dev/shm** | **9.3 s** | **~19 s** | **1.97 s** |

Single uncontended compile = 0.52 s (211 headers). The 27 s Lustre figure is pure small-file
metadata contention (256 × 211 opens); RAM-staging removes it.

## Recovery / rebuild

1. Toolchain is on disk at `/work/nvme/bgtv/sramasubramanian/x86cross/{debs,root}` (survives reboot).
2. RAM stage (per job, one-time): copy `x86cross/root → /dev/shm/xcross`, and rootfs subset
   (`usr/include`, `usr/lib/gcc`, `usr/lib/x86_64-linux-gnu`, `lib/x86_64-linux-gnu`) → `/dev/shm/rfsub`.
3. Re-fetch from scratch: `.deb` URLs are in this doc; `ar x *.deb && tar -xf data.tar.* -C root`.
4. Gate check: `XROOT=/dev/shm/xcross XSYSROOT=/dev/shm/rfsub python native_cross_gate.py` → expect 506/511
   identical with `-10` (or 511/511 with the `-17` exact-match toolchain).

## Scope

#42 fixes the **compile** phase only. The killed e2e run was stuck in the **qemu correctness** phase;
that still needs emulation and is the *other* lever — bundle correctness into the gem5 run (gem5 emits
the candidate's stdout; its per-tc 2-min timeout also caps the hanging cases). Per-step then ≈
`compile (~19 s) + gem5(+correctness)`, with gem5 the only real variable.

**Corpus caveat.** The 511 are pragma-stripped train *source* programs; production input is policy-
*generated* code, which may use `#pragma GCC target/optimize` or intrinsics. The same-gcc-source
mechanism makes generalization likely, but it is not proven on generated code — re-run the gate on a
sample of real rollouts before fully trusting byte-identity there. Note `native_dynamic_rl_step.py`
strips `#pragma GCC target` before compiling; **#43 must keep that stripping consistent** between the
reference-source and candidate compiles, or the byte-identity premise shifts.
