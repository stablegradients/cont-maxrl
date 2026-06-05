"""Full-dataset gem5-ARM sweep: for every problem with a (src,tgt) pair + tests,
measure our gem5-ARM time (mean simSeconds/test over up to K tests) for src and
tgt, verify correctness, and record alongside the dataset's reported gem5 times.
RESUMABLE (per-pid result files) + shardable (--shard i/N)."""
import json, os, re, hashlib, subprocess, sys, time, random
from concurrent.futures import ThreadPoolExecutor
import argparse

G="/work/nvme/bgtv/sramasubramanian/gem5"; GEM5=f"{G}/build/ARM/gem5.opt"; SE=f"{G}/configs/deprecated/example/se.py"
GPP="/opt/cray/pe/gcc-native/14/bin/g++"; TESTS="/work/nvme/bgmo/sramasubramanian/pie_data/tests"
RAW="/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"

ap=argparse.ArgumentParser()
ap.add_argument("--out_dir", default="/work/nvme/bgtv/sramasubramanian/gem5_sweep")
ap.add_argument("--shard", default="0/1")          # i/N
ap.add_argument("--k_tests", type=int, default=10) # cap tests/problem -> mean per test
ap.add_argument("--workers", type=int, default=48)
ap.add_argument("--per_test_timeout", type=int, default=90)
a=ap.parse_args()
i,N=map(int,a.shard.split("/")); os.makedirs(a.out_dir,exist_ok=True)
SCR=f"/tmp/g5sweep_{i}"; os.makedirs(SCR,exist_ok=True)

# build pid -> best-gem5 (speedup, src, tgt, src_agg, tgt_agg); only problems with tests on disk
pairs={}
for line in open(RAW):
    r=json.loads(line); pid=r["problem_id"]
    if not os.path.exists(f"{TESTS}/{pid}/input.0.txt"): continue
    sp=float(r["speedup"])
    if pid not in pairs or sp>pairs[pid][0]:
        pairs[pid]=(sp,r["src_code"],r["tgt_code"],float(r["src_agg_runtime"]),float(r["tgt_agg_runtime"]))
allp=sorted(pairs); mine=[p for j,p in enumerate(allp) if j%N==i]
todo=[p for p in mine if not os.path.exists(f"{a.out_dir}/{p}.json")]
print(f"shard {i}/{N}: {len(mine)} problems, {len(todo)} remaining (resumable)", flush=True)

def cc(code,path):
    code=re.sub(r'#pragma GCC target.*\n','',code); open(path+".cpp","w").write(code)
    return subprocess.run([GPP,"-O3","-std=c++17","-static",path+".cpp","-o",path],capture_output=True,timeout=90).returncode==0

def gem5_mpt(prog,pid,ntest):
    ts=list(range(ntest)); random.Random(int(hashlib.md5(pid.encode()).hexdigest()[:8],16)).shuffle(ts)
    ts=ts[:a.k_tests]; tot=0.0; n=0; ok=True
    for j in ts:
        out=f"{SCR}/{os.path.basename(prog)}_{j}"; of=f"{out}.po"
        try:
            subprocess.run([GEM5,"-d",out,SE,"--cmd",prog,"--input",f"{TESTS}/{pid}/input.{j}.txt","--output",of,
                            "--cpu-type=O3CPU","--caches","--l2cache"],capture_output=True,timeout=a.per_test_timeout)
        except Exception: continue
        try:
            if open(of).read().strip()!=open(f"{TESTS}/{pid}/output.{j}.txt").read().strip(): ok=False
            for l in open(f"{out}/stats.txt"):
                if l.startswith("simSeconds"): tot+=float(l.split()[1]); n+=1; break
        except Exception: pass
        finally:
            subprocess.run(["rm","-rf",out],capture_output=True)
    return (tot/n if n else None), ok, n

def work(pid):
    sp,src,tgt,sa,ta=pairs[pid]
    nt=0
    while os.path.exists(f"{TESTS}/{pid}/input.{nt}.txt"): nt+=1
    ps=f"{SCR}/{pid}_s"; pt=f"{SCR}/{pid}_t"
    try:
        if not cc(src,ps) or not cc(tgt,pt):
            res={"pid":pid,"status":"compile_fail"};
        else:
            os_,oks,ns=gem5_mpt(ps,pid,nt); ot,okt,ntt=gem5_mpt(pt,pid,nt)
            if not os_ or not ot:
                res={"pid":pid,"status":"gem5_fail"}
            else:
                res={"pid":pid,"status":"ok","n_tests":nt,"data_sp":sp,"data_src":sa,"data_tgt":ta,
                     "our_src_mpt":os_,"our_tgt_mpt":ot,"our_sp":os_/ot,"correct":bool(oks and okt)}
    except Exception as e:
        res={"pid":pid,"status":f"err:{e}"}
    json.dump(res,open(f"{a.out_dir}/{pid}.json","w")); return res["status"]

t0=time.time(); done=0
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    for st in ex.map(work, todo):
        done+=1
        if done%20==0: print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
print(f"shard {i} done: {done} problems in {time.time()-t0:.0f}s", flush=True)
