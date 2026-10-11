"""Every 20 minutes, look at the background jobs' output files for new EVENT / STOPPED / Traceback lines and at the main
account's GPU quota; exit (which wakes the session) as soon as something new appears, or after HOURS.
Usage: python tools/heartbeat.py HOURS OUTPUT_FILE [OUTPUT_FILE ...]"""
import re
import subprocess
import sys
import time
from pathlib import Path

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
hours, files = float(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
PATTERN = re.compile(r'EVENT|STOPPED|Traceback|Error')


def hits(path):
    try:
        return [l for l in path.read_text(encoding='utf-8', errors='replace').splitlines() if PATTERN.search(l)]
    except OSError:
        return []


seen = {f: len(hits(f)) for f in files}
end = time.time() + hours * 3600
while time.time() < end:
    time.sleep(1200)
    new = []
    for f in files:
        h = hits(f)
        new += [f'{f.stem}: {l[:220]}' for l in h[seen[f]:]]
        seen[f] = len(h)
    try:
        q = subprocess.run([PY, str(OTHER / 'tools/quota_both.py')], cwd=OTHER, capture_output=True, text=True,
                           timeout=180).stdout
        m = re.search(r'^yodharinmonplub GPU .*?"timeUsed": "([\d.]+)', q, re.M)
        used = float(m.group(1).split('.')[0]) if m else None
    except Exception:
        used = None
    print(time.strftime('%m-%d %H:%M'), 'tick | main GPU used', used, flush=True)
    if new or (used is not None and used > 10 ** 9):
        print('\n'.join(new) or f'main GPU used {used} s of 21600', flush=True)
        break
