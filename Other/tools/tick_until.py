"""Tick already-pushed dummy runs with tick_light.py (verify, then submit once) until each is submitted or failed, DM a
summary, then watch the scores with score_watch.py.
Usage: python tools/tick_until.py MODEL [MODEL ...]"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
NAMES = sys.argv[1:]
TERMINAL = ('STATE SUBMITTED', 'STATE SCORED', 'STATE VERIFY_FAILED', 'STATE FAILED', 'STATE SUBMISSION_FAILED')


def now():
    return time.strftime('%m-%d %H:%M')


results, pending = {}, list(NAMES)
end = time.time() + 3 * 3600
while pending and time.time() < end:
    for name in list(pending):
        msg = (OTHER / f'artifacts/{name}_notebook/message.txt').read_text(encoding='utf-8').strip()
        try:
            r = subprocess.run([PY, str(OTHER / 'tools/tick_light.py'), name, msg], cwd=OTHER, capture_output=True,
                               text=True, timeout=900)
            last = ((r.stdout + r.stderr).strip().splitlines() or [''])[-1]
        except subprocess.TimeoutExpired:
            last = 'timeout'
        if last.startswith(TERMINAL):
            results[name] = last
            pending.remove(name)
            print(now(), 'EVENT', name, '|', last[:300], flush=True)
    if pending:
        time.sleep(90)
summary = '\n'.join(f'{k}: {v[:150]}' for k, v in results.items()) + (f'\nnot finished: {pending}' if pending else '')
with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write('Dummy runs checked and submitted:\n' + summary + '\nScores in ~2-5 h.')
subprocess.run([PY, str(OTHER / 'scripts/discord_notify.py'), 'RSNA: submissions sent', f.name, '--dm'],
               cwd=OTHER, capture_output=True, text=True, timeout=120)
print(now(), 'EVENT submissions done', flush=True)
submitted = [k for k, v in results.items() if v.startswith('STATE SUBMITTED')]
if submitted:
    subprocess.run([PY, '-u', str(OTHER / 'tools/score_watch.py'), '10'] + submitted, cwd=OTHER)
