"""For the gem5 sweep's failed/incorrect problems: does the failure (timeout vs
wrong-output) hit SRC (slow) or TGT (fast) code? src-side timeouts would bias
the measured speedup DOWNWARD."""
import json, os, re, glob, subprocess, random
G="/work/nvme/bgtv/sramasubramanian/gem5"; GEM5=f"{G}/build/ARM/gem5.opt"; SE=f"{G}/configs/deprecated/example/se.py"
GPP="/opt/cray/pe/gcc-native/14/bin/g++"; TESTS="/work/nvme/bgmo/sramasubramanian/pie_data/tests"
SWEEP="/work/nvme/bgtv/sramasubramanian/gem5_sweep"; SCR="/tmp/g5diag"; os.makedirs(SCR,exist_ok=True)

# failed problems: gem5_fail, or ok-but-correct=False
bad=[]
for f in glob.glob(f"{SWEEP}/*.json"):
    r=json.load(open(f))
    if r.get("status")=="gem5_fail" or (r.get("status")=="ok" and not r.get("correct")):
        bad.append(r["pid"])
print(f"failed/incorrect problems total: {len(bad)}")
sample=random.Random(0).sample(bad, min(25,len(bad)))
# src/tgt from raw
pair={}
for line in open("/work/nvme/bgmo/sramasubramanian/pie_data/raw/train.jsonl"):
    r=json.loads(line); pid=r["problem_id"]
    if pid in sample:
        sp=float(r["speedup"])
        if pid not in pair or sp>pair[pid][0]: pair[pid]=(sp,r["src_code"],r["tgt_code"])

def cc(code,p):
    code=re.sub(r'#pragma GCC target.*\n','',code); open(p+".cpp","w").write(code)
    return subprocess.run([GPP,"-O3","-std=c++17","-static",p+".cpp","-o",p],capture_output=True,timeout=90).returncode==0

def classify(prog, pid):
    """run gem5 on first few tests; return overall outcome for this program."""
    n=0; outcomes=[]
    while os.path.exists(f"{TESTS}/{pid}/input.{n}.txt") and n<5: n+=1
    for j in range(n):
        out=f"{SCR}/{os.path.basename(prog)}_{j}"; of=f"{out}.po"
        try:
            subprocess.run([GEM5,"-d",out,SE,"--cmd",prog,"--input",f"{TESTS}/{pid}/input.{j}.txt","--output",of,
                            "--cpu-type=O3CPU","--caches","--l2cache"],capture_output=True,timeout=90)
        except subprocess.TimeoutExpired:
            outcomes.append("TIMEOUT"); continue
        try:
            got=open(of).read().strip(); exp=open(f"{TESTS}/{pid}/output.{j}.txt").read().strip()
            outcomes.append("ok" if got==exp else "WRONG")
        except Exception: outcomes.append("ERR")
        finally: subprocess.run(["rm","-rf",out],capture_output=True)
    if "TIMEOUT" in outcomes: return "timeout"
    if "WRONG" in outcomes: return "wrong_output"
    if "ERR" in outcomes or not outcomes: return "error"
    return "ok"

from collections import Counter
res=Counter()
print(f"\n{'pid':>8} {'inp_bytes':>9} {'SRC':>12} {'TGT':>12}")
for pid in sample:
    if pid not in pair: continue
    sp,src,tgt=pair[pid]
    ps,pt=f"{SCR}/{pid}_s",f"{SCR}/{pid}_t"
    cs,ct=cc(src,ps),cc(tgt,pt)
    so=classify(ps,pid) if cs else "compile_fail"
    to=classify(pt,pid) if ct else "compile_fail"
    ib=os.path.getsize(f"{TESTS}/{pid}/input.0.txt")
    print(f"{pid:>8} {ib:>9} {so:>12} {to:>12}")
    res[("src",so)]+=1; res[("tgt",to)]+=1
print("\n=== tally (which side, which failure) ===")
for side in ["src","tgt"]:
    c={k[1]:v for k,v in res.items() if k[0]==side}
    print(f"  {side}: {dict(c)}")
print(f"\nINTERPRETATION: if 'timeout' is much higher for SRC than TGT -> the slow code times out -> speedup UNDERestimated (our 0.70 bias).")
