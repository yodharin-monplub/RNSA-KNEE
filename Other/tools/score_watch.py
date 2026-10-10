"""Poll tick_light.py for submitted notebooks until each has a public score (or failed); print EVENT lines and exit.
Usage: score_watch.py HOURS MODEL [MODEL ...]"""
import subprocess
import sys
import time
from pathlib import Path

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
hours, names = float(sys.argv[1]), sys.argv[2:]
end = time.time() + hours * 3600
pending = list(names)
while pending and time.time() < end:
    for name in list(pending):
        msg = (OTHER / f'artifacts/{name}_notebook/message.txt').read_text(encoding='utf-8').strip()
        try:
            r = subprocess.run([PY, str(OTHER / 'tools/tick_light.py'), name, msg], cwd=OTHER, capture_output=True,
                               text=True, timeout=600)
            last = ((r.stdout + r.stderr).strip().splitlines() or [''])[-1]
        except subprocess.TimeoutExpired:
            last = 'timeout'
        if last.startswith(('STATE SCORED', 'STATE SUBMISSION_FAILED')):
            print(time.strftime('%m-%d %H:%M'), 'EVENT', name, '|', last[:200], flush=True)
            pending.remove(name)
    if pending:
        time.sleep(600)
print(time.strftime('%m-%d %H:%M'), 'done; still pending:', pending, flush=True)
