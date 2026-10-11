"""Submit notebooks whose dummy run was already verified (tick_light.py --no-submit wrote verify.txt 'OK ...') after the
07:00 (+07) daily submission reset, then watch their scores. No new GPU session is started (the verified version is
submitted). DMs a summary after submitting.
Usage: python tools/sunday_submit.py START_ISO MODEL [MODEL ...]   (e.g. 2026-10-11T07:01 model157 model158 model159)"""
import datetime as dt
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
target = dt.datetime.fromisoformat(sys.argv[1])
NAMES = sys.argv[2:]


def now():
    return dt.datetime.now().strftime('%m-%d %H:%M')


def run(args, timeout):
    try:
        r = subprocess.run([PY] + args, cwd=OTHER, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 'timeout'


for name in NAMES:
    d = OTHER / f'artifacts/{name}_notebook'
    assert (d / 'message.txt').exists(), name
    assert (d / 'verify.txt').read_text().startswith('OK '), f'{name} not verified'
    assert not (d / 'never_submit.txt').exists(), name
while dt.datetime.now() < target:
    time.sleep(30)
print(now(), 'start: submitting', NAMES, flush=True)
results = {}
for name in NAMES:
    if (OTHER / f'artifacts/{name}_notebook/never_submit.txt').exists():   # vetoed after arming
        results[name] = 'skipped (never_submit.txt)'
        continue
    msg = (OTHER / f'artifacts/{name}_notebook/message.txt').read_text(encoding='utf-8').strip()
    for attempt in range(3):
        last = (run([str(OTHER / 'tools/tick_light.py'), name, msg], 900).splitlines() or [''])[-1]
        if last.startswith('STATE SUBMITTED'):
            break
        time.sleep(120)
    results[name] = last
    print(now(), 'EVENT', name, '|', last[:260], flush=True)
summary = '\n'.join(f'{k}: {v[:120]}' for k, v in results.items())
with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write('Submitted after the 07:00 reset (dummy runs verified yesterday, no new GPU session):\n' + summary +
            '\nScores in ~2-3 h.')
subprocess.run([PY, str(OTHER / 'scripts/discord_notify.py'), 'RSNA: Sunday submissions sent', f.name, '--dm'],
               cwd=OTHER, capture_output=True, text=True, timeout=120)
print(now(), 'EVENT submissions done', flush=True)
submitted = [k for k, v in results.items() if v.startswith('STATE SUBMITTED')]
if submitted:
    subprocess.run([PY, '-u', str(OTHER / 'tools/score_watch.py'), '8'] + submitted, cwd=OTHER)
