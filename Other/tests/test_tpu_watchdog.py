"""Local test of the TPU kernel's trainer watchdog (scripts/run_tpu_train.py, run_trainer): the exact function text from
the kernel template is executed against two fake trainers - one prints and then hangs (must be killed after WATCHDOG_S
of silence and reported as a failure), one prints and finishes (its output must pass through and the run succeed).
On Windows, os.killpg / signal.SIGKILL are stood in by a taskkill of the process tree (the kernel runs on Linux).
Usage: python tests/test_tpu_watchdog.py"""
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
src = (SCRIPTS / 'run_tpu_train.py').read_text(encoding='utf-8')
kernel = src[src.index("KERNEL = r'''") + len("KERNEL = r'''"):]
kernel = kernel[:kernel.index("'''")]
func = kernel[kernel.index('def run_trainer('):kernel.index('def train_with_resume(')]
func = func.replace('__SMOKE_LIMIT_S__', '60')

if not hasattr(os, 'killpg'):
    os.killpg = lambda pid, sig: subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], capture_output=True)
if not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = 9

tmp = Path(tempfile.mkdtemp(prefix='wd_'))
(tmp / 'scripts').mkdir()
ns = {'os': os, 'sys': sys, 'subprocess': subprocess, 'signal': signal, 'threading': threading, 'time': time,
      'Path': Path, 'ROOT': tmp, 'config_path': tmp / 'config.json', 'env': dict(os.environ), 'started': time.time(),
      'WATCHDOG_S': 4}
exec(compile(func, 'run_trainer', 'exec'), ns)
run_trainer = ns['run_trainer']

# (1) prints, then hangs -> killed after ~4 s of silence, reported as a failure
(tmp / 'scripts/train_xla.py').write_text("import time, sys\nprint('step 0', flush=True)\ntime.sleep(600)\n")
out = tmp / 'out1'; out.mkdir()
t = time.time()
ok = run_trainer(tmp, out, time.time() + 3600)
dt = time.time() - t
assert not ok and dt < 60, (ok, dt)
print(f'(1) hung trainer killed after {dt:.0f}s, run reported failed OK')

# (2) prints for 3 s, writes TRAINING_COMPLETE, exits 0 -> success, never killed
(tmp / 'scripts/train_xla.py').write_text(
    "import time, sys\nfrom pathlib import Path\nout = Path(sys.argv[sys.argv.index('--output') + 1])\n"
    "for i in range(3):\n    print('step', i, flush=True)\n    time.sleep(1)\n(out / 'TRAINING_COMPLETE').write_text('ok')\n")
out = tmp / 'out2'; out.mkdir()
ok = run_trainer(tmp, out, time.time() + 3600)
assert ok, 'normal trainer should succeed'
print('(2) normal trainer passes OK')
print('WATCHDOG TESTS PASS')
