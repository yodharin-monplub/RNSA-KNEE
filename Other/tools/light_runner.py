"""Submit LIGHT-gated notebooks whose dummy runs were pushed (and ideally verified) beforehand: wait until the given
local time (the daily submission limit resets at 07:00 +07), then run tick_light.py for each model in order until it is
SUBMITTED (or fails verification). Each model's submission message is read from artifacts/<model>_notebook/message.txt.
Usage: python light_runner.py 2026-10-05T07:01 model135 model136 ..."""
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
target = dt.datetime.fromisoformat(sys.argv[1])
NAMES = sys.argv[2:]
for name in NAMES:
    assert (OTHER / f'artifacts/{name}_notebook/message.txt').exists(), f'{name}: no message.txt'
    assert not (OTHER / f'artifacts/{name}_notebook/never_submit.txt').exists(), f'{name}: never_submit.txt'
while dt.datetime.now() < target:
    time.sleep(30)
print(dt.datetime.now().strftime('%H:%M'), 'submitting', NAMES, flush=True)
pending, done = list(NAMES), {}
end = time.time() + 6 * 3600
while pending and time.time() < end:
    for name in list(pending):
        message = (OTHER / f'artifacts/{name}_notebook/message.txt').read_text(encoding='utf-8').strip()
        try:
            r = subprocess.run([PY, str(SCRATCH / 'tick_light.py'), name, message], cwd=OTHER, capture_output=True,
                               text=True, timeout=900)
            last = ((r.stdout + r.stderr).strip().splitlines() or [''])[-1]
        except subprocess.TimeoutExpired:
            last = 'timeout'
        if last.startswith(('STATE SUBMITTED', 'STATE SCORED', 'STATE VERIFY_FAILED', 'STATE FAILED',
                            'STATE SUBMISSION_FAILED')):
            done[name] = last
            pending.remove(name)
            print(dt.datetime.now().strftime('%H:%M'), name, '|', last[:220], flush=True)
        elif 'error' in last.lower() or 'exception' in last.lower():
            print(dt.datetime.now().strftime('%H:%M'), name, 'retry after:', last[:200], flush=True)
    if pending:
        time.sleep(120)
print('EVENT light runner done', done, '| still pending', pending, flush=True)
