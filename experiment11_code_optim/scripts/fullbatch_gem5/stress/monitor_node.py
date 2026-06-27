"""Robust 1 Hz node resource monitor (CPU% via /proc/stat, RAM via /proc/meminfo,
GPU via nvidia-smi). Tags each sample with the running config from the job's
CURRENT_CONFIG marker. Stops when STOP_SAMPLER appears in the run dir (job end)
or after max iters. Launched via `srun --overlap --jobid=<job>` onto the live node
because the in-job bash sampler's awk ternary is rejected by this system's awk.

  python monitor_node.py <run_dir>
"""
import os, sys, time, subprocess

run_dir = sys.argv[1]
out = os.path.join(run_dir, "resources_cpu.csv")
marker = os.path.join(run_dir, "CURRENT_CONFIG")
stop = os.path.join(run_dir, "STOP_SAMPLER")


def cpu_times():
    p = list(map(int, open("/proc/stat").readline().split()[1:]))
    return p[3] + p[4], sum(p)            # idle+iowait, total


def ram_mib():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, v = line.partition(":")
        d[k] = int(v.split()[0])
    return (d["MemTotal"] - d["MemAvailable"]) // 1024, d["MemTotal"] // 1024


pi, pt = cpu_times()
f = open(out, "w")
f.write("ts,phase,cpu_pct,ram_used_mib,ram_total_mib,gpu,mem_used_mib,mem_total_mib,gpu_util\n")
for _ in range(40000):
    if os.path.exists(stop):
        break
    time.sleep(1)
    i, t = cpu_times()
    cpu = 100.0 * (1 - (i - pi) / (t - pt)) if t > pt else 0.0
    pi, pt = i, t
    ru, rt = ram_mib()
    phase = open(marker).read().strip() if os.path.exists(marker) else "-"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        smi = ""
    ts = int(time.time())
    for line in smi.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 4:
            f.write(f"{ts},{phase},{cpu:.1f},{ru},{rt},{parts[0]},{parts[1]},{parts[2]},{parts[3]}\n")
    f.flush()
f.close()
