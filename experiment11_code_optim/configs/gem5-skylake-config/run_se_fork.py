# Fork-from-live Verbatim SE config (#41). Simulate the shared startup once, then m5.fork() a COW child
# that continues to exit (warm caches inherited -> near-bit-exact, no serialize/restore corruption).
#   GEM5_STDIN=<file>     : file-backed stdin
#   GEM5_FORK_TICK=<n>    : run to tick n, fork, child continues to exit (prints CHILD_EXIT tick)
#   (none)               : normal full run (prints FULL tick)
import m5, os, sys
from m5.objects import *
import argparse
from system.se import MySystem
from system.core import *

valid = {c.__name__[:-3]: c for c in [VerbatimCPU, TunedCPU, UnconstrainedCPU]}
ap = argparse.ArgumentParser(); ap.add_argument('config', choices=valid.keys()); ap.add_argument('binary')
args = ap.parse_args()
class TestSystem(MySystem):
    _CPUModel = valid[args.config]
system = TestSystem(); system.setTestBinary(args.binary)
_stdin = os.environ.get("GEM5_STDIN")
if _stdin:
    system.cpu.workload[0].input = _stdin
_RF = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
system.redirect_paths = [RedirectPath(app_path=p, host_paths=[_RF + p])
                         for p in ["/lib64", "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/lib"]]
root = Root(full_system=False, system=system)
m5.instantiate()

_ft = os.environ.get("GEM5_FORK_TICK")
if _ft:
    m5.disableAllListeners()
    ev = m5.simulate(int(_ft))
    sys.stdout.write("FORKPOINT cause=%s tick=%d\n" % (ev.getCause(), m5.curTick())); sys.stdout.flush()
    pid = m5.fork()
    if pid == 0:                      # child: continue to exit
        ev = m5.simulate()
        sys.stdout.write("CHILD_EXIT cause=%s tick=%d\n" % (ev.getCause(), m5.curTick())); sys.stdout.flush()
        os._exit(0)
    else:                            # parent: wait, done
        os.waitpid(pid, 0)
        sys.stdout.write("PARENT_DONE\n"); sys.stdout.flush()
        os._exit(0)
else:
    ev = m5.simulate()
    sys.stdout.write("FULL cause=%s tick=%d\n" % (ev.getCause(), m5.curTick())); sys.stdout.flush()
