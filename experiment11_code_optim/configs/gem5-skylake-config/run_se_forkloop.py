# Startup amortization via FORK-AT-TICK (avoids the retry-read bug): parent simulates the startup to
# GEM5_FORK_TICK (just BEFORE the program reads stdin), then forks a COW child per test case. Each child
# continues a few ticks to the read and does a FRESH read (which works) of the current fixed-path content.
#   GEM5_FIXED_STDIN=<path>  GEM5_TC_LIST=<file>  GEM5_FORK_TICK=<n>
import m5, os, sys, shutil
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
FIXED = os.environ["GEM5_FIXED_STDIN"]
system.cpu.workload[0].input = FIXED
_RF = "/work/nvme/bgyd/sramasubramanian/build-temp-332796050/rootfs"
system.redirect_paths = [RedirectPath(app_path=p, host_paths=[_RF + p])
                         for p in ["/lib64", "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib", "/lib"]]
root = Root(full_system=False, system=system)
m5.disableAllListeners()
m5.instantiate()

tcs = [l.strip() for l in open(os.environ["GEM5_TC_LIST"]) if l.strip()]
FT = int(os.environ["GEM5_FORK_TICK"])

def reset_fixed_offset():
    # gem5's host fd backing fd0 is shared across forks (one open file description);
    # a child's read advances it, so reset it to 0 before each fork so the next child
    # reads its input from the start. (Find the fd(s) pointing at FIXED in this process.)
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath("/proc/self/fd/" + fd) == os.path.realpath(FIXED):
                os.lseek(int(fd), 0, os.SEEK_SET)
        except OSError:
            pass
ev = m5.simulate(FT)                      # run startup to just before the read
sys.stdout.write("STARTUP cause=%s tick=%d\n" % (ev.getCause(), m5.curTick())); sys.stdout.flush()

for i, tc in enumerate(tcs):
    shutil.copyfile(tc, FIXED)
    reset_fixed_offset()                  # rewind the shared fd0 host offset to 0
    pid = m5.fork()
    if pid == 0:                          # child: continue to exit (fresh read of FIXED)
        ev = m5.simulate()
        sys.stdout.write("TC %d tick %d cause %s\n" % (i, m5.curTick(), ev.getCause())); sys.stdout.flush()
        os._exit(0)
    else:
        os.waitpid(pid, 0)
sys.stdout.write("FORKLOOP_DONE ntc=%d\n" % len(tcs)); sys.stdout.flush()
os._exit(0)
