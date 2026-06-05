# Checkpoint/restore + fork-capable Verbatim SE config for startup-cost amortization (#41).
# Modes via env vars (so the PIE driver stays subprocess-based):
#   GEM5_STDIN=<file>       : file-backed stdin (REQUIRED for ckpt/restore to replay input)
#   GEM5_CKPT_DIR=<dir> + GEM5_CKPT_TICK=<n> : run to tick n, write checkpoint, then continue to exit
#   GEM5_RESTORE_DIR=<dir>  : restore from checkpoint, continue to exit (caches cold on restore)
#   (none)                  : normal full run
import m5, os
from m5.objects import *
import argparse
from system.se import MySystem
from system.core import *

valid = {c.__name__[:-3]: c for c in [VerbatimCPU, TunedCPU, UnconstrainedCPU]}
ap = argparse.ArgumentParser()
ap.add_argument('config', choices=valid.keys())
ap.add_argument('binary', type=str)
args = ap.parse_args()

class TestSystem(MySystem):
    _CPUModel = valid[args.config]

system = TestSystem()
system.setTestBinary(args.binary)

# file-backed stdin (so checkpoint can save+replay the input fd offset)
_stdin = os.environ.get("GEM5_STDIN")
if _stdin:
    system.cpu.workload[0].input = _stdin

_RF = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
system.redirect_paths = [
    RedirectPath(app_path="/lib64",                    host_paths=[_RF + "/lib64"]),
    RedirectPath(app_path="/lib/x86_64-linux-gnu",     host_paths=[_RF + "/lib/x86_64-linux-gnu"]),
    RedirectPath(app_path="/usr/lib/x86_64-linux-gnu", host_paths=[_RF + "/usr/lib/x86_64-linux-gnu"]),
    RedirectPath(app_path="/usr/lib",                  host_paths=[_RF + "/usr/lib"]),
    RedirectPath(app_path="/lib",                      host_paths=[_RF + "/lib"]),
]

root = Root(full_system=False, system=system)

_restore = os.environ.get("GEM5_RESTORE_DIR")
_ckpt_dir = os.environ.get("GEM5_CKPT_DIR")
_ckpt_tick = os.environ.get("GEM5_CKPT_TICK")

if _restore:
    m5.instantiate(_restore)
else:
    m5.instantiate()

if _ckpt_dir and _ckpt_tick and not _restore:
    # run to the checkpoint tick, snapshot, then continue to exit (so we also get the full ticks)
    ev = m5.simulate(int(_ckpt_tick))
    if ev.getCause() == "simulate() limit reached":
        m5.checkpoint(_ckpt_dir)
        print("GEM5_CKPT_DONE tick=%d" % m5.curTick())
    ev = m5.simulate()
else:
    ev = m5.simulate()

print("EXIT_CAUSE %s tick=%d" % (ev.getCause(), m5.curTick()))
