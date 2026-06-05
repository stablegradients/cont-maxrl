# Startup-cost amortization driver (#41): simulate the input-independent startup ONCE, then fork a COW
# child per test case from the post-startup (pre-input) pause point. Requires gem5 built with the
# read(0) hook (GEM5_PREINPUT_FORK). Env:
#   GEM5_FIXED_STDIN=<path>  : the file fd0 is bound to; we overwrite its CONTENT per test case
#   GEM5_TC_LIST=<file>      : newline-separated list of test-case input file paths
#   GEM5_PREINPUT_FORK=1     : enable the read(0) pause hook
# Prints "TC <i> tick <n> cause <c>" per child and "STARTUP_TICK <n>".
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

# run startup; the read(0) hook pauses us right before the program reads stdin
ev = m5.simulate()
sys.stdout.write("STARTUP cause=%s tick=%d\n" % (ev.getCause(), m5.curTick())); sys.stdout.flush()
if ev.getCause() != "preinput_read":
    sys.stdout.write("WARN: expected preinput_read, got %s\n" % ev.getCause()); sys.stdout.flush()

if os.environ.get("GEM5_NOFORK"):       # DEBUG: pause, set tc0, continue in-process (no fork)
    shutil.copyfile(tcs[0], FIXED)
    ev = m5.simulate()
    sys.stdout.write("NOFORK tick %d cause %s\n" % (m5.curTick(), ev.getCause())); sys.stdout.flush()
    os._exit(0)

for i, tc in enumerate(tcs):
    shutil.copyfile(tc, FIXED)          # set fd0 content = this test case
    pid = m5.fork()
    if pid == 0:                         # child: resume (re-executes read with this input) to exit
        ev = m5.simulate()
        sys.stdout.write("TC %d tick %d cause %s\n" % (i, m5.curTick(), ev.getCause())); sys.stdout.flush()
        os._exit(0)
    else:                                # parent: wait, then fork next from the same pause point
        os.waitpid(pid, 0)

sys.stdout.write("AMORTIZE_DONE ntc=%d\n" % len(tcs)); sys.stdout.flush()
os._exit(0)
