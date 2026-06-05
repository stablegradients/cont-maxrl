"""Check: is our gem5-ARM time a ~CONSTANT multiple of the dataset's reported
gem5-x86 time, for src AND tgt across problems? If yes -> our build faithfully
reproduces the dataset gem5 (rescaled), and speedup ratios are preserved."""
import json, os, re, hashlib, subprocess, time
from concurrent.futures import ThreadPoolExecutor
G="/work/nvme/bgtv/sramasubramanian/gem5"; GEM5=f"{G}/build/ARM/gem5.opt"; SE=f"{G}/configs/deprecated/example/se.py"
GPP="/opt/cray/pe/gcc-native/14/bin/g++"; TESTS="/work/nvme/bgmo/sramasubramanian/pie_data/tests"
MAN=json.load(open(f"{TESTS}/MANIFEST.json"))["per_problem"]
SCR="/tmp/g5val"; os.makedirs(SCR, exist_ok=True)

# sample public problems with few tests (so gem5 over ALL tests is affordable)
pairs={}  # pid -> best-gem5 (speedup, src, tgt, src_agg, tgt_agg)
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r=json.loads(line); pid=r["problem_id"]
    m=MAN.get(pid,{})
    if m.get("source")!="public" or not (2<=m.get("n_tests",0)<=8): continue
    sp=float(r["speedup"])
    if pid not in pairs or sp>pairs[pid][0]:
        pairs[pid]=(sp, r["src_code"], r["tgt_code"], float(r["src_agg_runtime"]), float(r["tgt_agg_runtime"]))
import random
pids=random.Random(0).sample(list(pairs), 16)

def compile_static(code, path):
    code=re.sub(r'#pragma GCC target.*\n','',code)
    open(path+".cpp","w").write(code)
    r=subprocess.run([GPP,"-O2","-std=c++17","-static",path+".cpp","-o",path],capture_output=True,timeout=90)
    return r.returncode==0

def gem5_time(prog, pid):
    """sum simSeconds over ALL the problem's tests; verify correctness."""
    ins=[]; i=0
    while os.path.exists(f"{TESTS}/{pid}/input.{i}.txt"): ins.append(i); i+=1
    tot=0.0; ok=True
    for j in ins:
        out=f"{SCR}/{os.path.basename(prog)}_{j}"
        ofile=f"{out}.progout"
        try:
            subprocess.run([GEM5,"-d",out,SE,"--cmd",prog,"--input",f"{TESTS}/{pid}/input.{j}.txt",
                            "--output",ofile,"--cpu-type=O3CPU","--caches","--l2cache"],
                           capture_output=True,timeout=120)
        except Exception:
            return None,False
        got=open(ofile).read().strip() if os.path.exists(ofile) else ""
        exp=open(f"{TESTS}/{pid}/output.{j}.txt").read().strip()
        if got!=exp: ok=False
        for l in open(f"{out}/stats.txt"):
            if l.startswith("simSeconds"): tot+=float(l.split()[1]); break
    return tot, ok

def work(pid):
    sp,src,tgt,sa,ta=pairs[pid]
    ps=f"{SCR}/{pid}_s"; pt=f"{SCR}/{pid}_t"
    if not compile_static(src,ps) or not compile_static(tgt,pt): return None
    os_,oks=gem5_time(ps,pid); ot,okt=gem5_time(pt,pid)
    if os_ is None or ot is None or os_<=0 or ot<=0: return None
    return dict(pid=pid, data_sp=sp, data_src=sa, data_tgt=ta,
                our_src=os_, our_tgt=ot, our_sp=os_/ot,
                ratio_src=os_/sa, ratio_tgt=ot/ta, correct=(oks and okt))

print(f"running gem5-ARM on {len(pids)} public problems (src+tgt, all tests) ...", flush=True)
with ThreadPoolExecutor(max_workers=16) as ex:
    res=[r for r in ex.map(work, pids) if r]

print(f"\n{'pid':>8} {'data_src':>9} {'our_src':>10} {'ratioS':>8} | {'data_tgt':>9} {'our_tgt':>10} {'ratioT':>8} | {'data_sp':>8} {'our_sp':>7} {'ok':>3}")
import statistics as st
rs=[];rt=[];dsp=[];osp=[]
for r in sorted(res,key=lambda r:r['pid']):
    print(f"{r['pid']:>8} {r['data_src']:9.4f} {r['our_src']*1e6:10.2f} {r['ratio_src']*1e6:8.1f} | "
          f"{r['data_tgt']:9.4f} {r['our_tgt']*1e6:10.2f} {r['ratio_tgt']*1e6:8.1f} | "
          f"{r['data_sp']:8.2f} {r['our_sp']:7.2f} {str(r['correct']):>3}")
    rs.append(r['ratio_src']);rt.append(r['ratio_tgt']);dsp.append(r['data_sp']);osp.append(r['our_sp'])
allr=rs+rt
print(f"\n our_ARM/data_x86 ratio (x1e6, src+tgt pooled, n={len(allr)}): "
      f"median={st.median(allr)*1e6:.1f}  CoV(std/mean)={st.pstdev(allr)/st.mean(allr):.2f}")
print(f"   -> if CoV is SMALL, our gem5 is ~constant x dataset gem5 (faithful, rescaled)")
import numpy as np
print(f" speedup agreement: corr(our_sp, data_sp) log-log = {np.corrcoef(np.log(osp),np.log(dsp))[0,1]:+.2f}; "
      f"median |our_sp/data_sp| = {st.median([o/d for o,d in zip(osp,dsp)]):.2f}")
