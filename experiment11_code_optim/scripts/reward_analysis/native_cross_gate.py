"""#42 gate: compile every baseline .cpp with the NATIVE aarch64->x86_64 cross g++-9 and
prove the output is byte-identical (allocated sections) to the rootfs-native qemu-g++ binary.
Identical .text+.rodata(+.data,.data.rel.ro) => identical gem5 ticks by construction (gem5 proven
deterministic 60/60). Also times native compile (the #42 speedup). Run: python native_cross_gate.py [N]
"""
import os, sys, time, subprocess, queue, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("XROOT", "/work/nvme/bgtv/sramasubramanian/x86cross/root")   # toolchain (Lustre or /dev/shm)
RF   = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"               # baseline + REAL dynamic-linker
SYS  = os.environ.get("XSYSROOT", RF)   # headers/libs read here at compile (can be /dev/shm staged copy)
GXX  = f"{ROOT}/usr/bin/x86_64-linux-gnu-g++-9"
OBJ  = f"{ROOT}/usr/bin/x86_64-linux-gnu-objcopy"
STRIP= f"{ROOT}/usr/bin/x86_64-linux-gnu-strip"
LD   = f"{RF}/lib/x86_64-linux-gnu/ld-2.31.so"   # REAL rootfs path -> .interp matches baseline
G5   = f"{RF}/home/g5rl"
OUT  = "/dev/shm/xgate"; os.makedirs(OUT, exist_ok=True)
ENV  = {**os.environ,
        "PATH": f"{ROOT}/usr/bin:" + os.environ.get("PATH",""),
        "LD_LIBRARY_PATH": f"{ROOT}/usr/lib/aarch64-linux-gnu:{ROOT}/lib/aarch64-linux-gnu:{ROOT}/usr/lib"}
LFLAGS = ["--sysroot="+SYS, f"-L{SYS}/usr/lib/gcc/x86_64-linux-gnu/9",
          f"-L{SYS}/usr/lib/x86_64-linux-gnu", f"-L{SYS}/lib/x86_64-linux-gnu"]
# RIGOROUS check = strip .comment + .note.gnu.build-id from both, then cmp WHOLE binary (catches ANY
# allocated-section diff). SECS is only used to LOCALIZE where a mismatch lives.
SECS = [".text", ".rodata", ".rodata.cst8", ".rodata.cst16", ".data", ".data.rel.ro", ".init_array",
        ".eh_frame", ".eh_frame_hdr", ".gcc_except_table", ".gnu.version_r", ".dynsym", ".dynstr",
        ".rela.dyn", ".rela.plt", ".plt", ".plt.got", ".got", ".got.plt", ".tdata", ".tbss"]

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
CORES = sorted(os.sched_getaffinity(0)); W = len(CORES)
POOL = queue.Queue()
for c in CORES: POOL.put(c)

def dump(binp, sec, dst):
    if os.path.exists(dst): os.remove(dst)   # avoid stale file masking a tool failure
    subprocess.run([OBJ, "-O", "binary", "--only-section="+sec, binp, dst],
                   env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.getsize(dst) if os.path.exists(dst) else -1

def one(sid):
    cpp = f"{G5}/{sid}.cpp"; base = f"{G5}/{sid}"; out = f"{OUT}/{sid}"
    c = POOL.get()
    try:
        cmd = ["taskset","-c",str(c), GXX] + LFLAGS + ["-O3","-std=c++17",
               "-Wl,--dynamic-linker="+LD, cpp, "-o", out]
        t0 = time.time()
        p = subprocess.run(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        ct = time.time() - t0
        if p.returncode != 0 or not os.path.exists(out):
            return {"sid": sid, "ok": False, "ct": ct, "err": p.stderr.decode("utf-8","ignore")[:120]}
        # RIGOROUS: strip non-executed sections from both, cmp whole binary
        cb = f"{OUT}/{sid}.cb"; cx = f"{OUT}/{sid}.cx"
        shutil.copyfile(base, cb); shutil.copyfile(out, cx)
        srok = True
        for f in (cb, cx):
            r = subprocess.run([STRIP, "-R", ".comment", "-R", ".note.gnu.build-id", f],
                               env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if r.returncode != 0: srok = False
        if not srok:                                       # strip tool failed -> cannot trust result
            return {"sid": sid, "ok": False, "ct": ct, "err": "strip failed"}
        identical = subprocess.run(["cmp","-s",cb,cx]).returncode == 0
        diffs = []
        if not identical:                                  # localize which allocated section differs
            for s in SECS:
                a = f"{OUT}/{sid}.a"; b = f"{OUT}/{sid}.b"
                sa = dump(out, s, a); sb = dump(base, s, b)
                if sa == -1 and sb == -1: continue
                if sa != sb or subprocess.run(["cmp","-s",a,b]).returncode != 0:
                    diffs.append(f"{s}({sa}v{sb})")
                for f in (a, b):
                    try: os.remove(f)
                    except OSError: pass
        for f in (cb, cx, out):
            try: os.remove(f)
            except OSError: pass
        return {"sid": sid, "ok": True, "ct": ct, "match": identical, "diffs": diffs}
    except Exception as e:
        return {"sid": sid, "ok": False, "ct": 0, "err": str(e)[:120]}
    finally:
        POOL.put(c)

sids = sorted(f[:-4] for f in os.listdir(G5) if f.endswith(".cpp") and os.path.exists(f"{G5}/{f[:-4]}"))[:N]
print(f"[xgate] {len(sids)} programs, {W} cores", flush=True)
t0 = time.time(); res = []
with ThreadPoolExecutor(max_workers=W) as ex:
    futs = [ex.submit(one, s) for s in sids]
    for i, f in enumerate(as_completed(futs)):
        res.append(f.result())
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(sids)} ({time.time()-t0:.0f}s)", flush=True)
wall = time.time() - t0

okc = [r for r in res if r["ok"]]
failc = [r for r in res if not r["ok"]]
matched = [r for r in okc if r.get("match")]
mism = [r for r in okc if not r.get("match")]
cts = sorted(r["ct"] for r in okc)
print(f"\n=== #42 NATIVE CROSS-COMPILE GATE ===")
print(f"  programs:            {len(sids)}")
print(f"  compiled ok:         {len(okc)}   compile-fail: {len(failc)}")
print(f"  byte-identical (strip .comment+build-id, WHOLE binary cmp): {len(matched)}/{len(okc)}  ({100*len(matched)/max(len(okc),1):.1f}%)")
print(f"  binary-mismatch:     {len(mism)}")
print(f"  wall (all, {W}-way):  {wall:.1f}s   => per-1024 projected: {wall/max(len(sids),1)*1024:.0f}s")
if cts:
    print(f"  per-compile native:  median={cts[len(cts)//2]:.2f}s  p95={cts[int(len(cts)*0.95)]:.2f}s  max={cts[-1]:.2f}s")
if mism:
    print("  --- mismatches (sid: sections) ---")
    for r in mism[:30]: print(f"    {r['sid']}: {r['diffs']}")
if failc:
    print("  --- compile fails ---")
    for r in failc[:15]: print(f"    {r['sid']}: {r.get('err','')}")
import json
json.dump({"n":len(sids),"ok":len(okc),"identical":len(matched),"mismatch":len(mism),"fail":len(failc),
           "wall_s":round(wall,1),"per1024_s":round(wall/max(len(sids),1)*1024),
           "median_ct":round(cts[len(cts)//2],3) if cts else None,
           "mismatch_sids":[r["sid"] for r in mism],"fail_sids":[r["sid"] for r in failc]},
          open("/work/nvme/bgtv/sramasubramanian/gem5_scale_out/xgate_summary.json","w"), indent=2)
print("  wrote xgate_summary.json")
