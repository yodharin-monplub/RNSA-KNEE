"""Saturday 2026-10-10: after the 07:00 (+07) Kaggle GPU quota refresh / daily submission reset, push the dummy runs of
the prepared notebooks on the main account (light builds + fleet-only), verify each with tick_light.py and submit once;
then push the two Speedy profiling notebooks on the teammate account's GPU. Prints EVENT lines, DMs a summary.
Usage: python saturday_runner3.py [START_ISO] (default 2026-10-10T07:01)"""
import datetime as dt
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
NAMES = ['model150', 'model151', 'model153', 'model152', 'model143']   # priority order (2026-10-09: Versia_5 base + our leg; model143 = student alone)
target = dt.datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else '2026-10-10T07:01')


def now():
    return dt.datetime.now().strftime('%m-%d %H:%M')


def run(args, timeout):
    try:
        r = subprocess.run([PY] + args, cwd=OTHER, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 'timeout'


for name in NAMES:
    assert (OTHER / f'artifacts/{name}_notebook/message.txt').exists(), name
while dt.datetime.now() < target:
    time.sleep(30)
print(now(), 'start: pushing', NAMES, flush=True)
for name in NAMES:
    out = run([str(SCRATCH / 'push_any.py'), 'yodharinmonplub', f'artifacts/{name}_notebook', 'NvidiaTeslaT4', '180'], 4 * 3600)
    print(now(), name, 'push:', (out.splitlines() or ['?'])[-1][:200], flush=True)
results, pending, repushes = {}, list(NAMES), {n: 0 for n in NAMES}
end = time.time() + 6 * 3600
while pending and time.time() < end:
    for name in list(pending):
        msg = (OTHER / f'artifacts/{name}_notebook/message.txt').read_text(encoding='utf-8').strip()
        last = (run([str(SCRATCH / 'tick_light.py'), name, msg], 900).splitlines() or [''])[-1]
        if last.startswith('STATE FAILED') and repushes[name] < 2:
            # 2026-10-08: Kaggle's ERRORED_MOUNTING_DATASET bug fails runs before any code runs -> push the run again
            repushes[name] += 1
            out = run([str(SCRATCH / 'push_any.py'), 'yodharinmonplub', f'artifacts/{name}_notebook', 'NvidiaTeslaT4', '60'], 2 * 3600)
            print(now(), name, f'dummy run FAILED -> re-pushed ({repushes[name]})', (out.splitlines() or ['?'])[-1][:160], flush=True)
            continue
        if last.startswith(('STATE SUBMITTED', 'STATE SCORED', 'STATE VERIFY_FAILED', 'STATE FAILED', 'STATE SUBMISSION_FAILED')):
            results[name] = last
            pending.remove(name)
            print(now(), 'EVENT', name, '|', last[:260], flush=True)
    if pending:
        time.sleep(90)
# Speedy profiling on the teammate's GPU (free of the main account's quota)
for variant in ('bf16', 'fp16'):
    out = run(['scripts/make_speedy_profile.py', '--variant', variant, '--account', 'krittanutsomtuas', '--push'], 900)
    print(now(), 'profile', variant, '|', (out.splitlines() or ['?'])[-1][:200], flush=True)
summary = '\n'.join(f'{k}: {v[:160]}' for k, v in results.items()) + (f'\nnot finished: {pending}' if pending else '')
with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write('Saturday submissions (scores in ~1-8 h):\n' + summary +
            '\nAlso started: Speedy profiling (bf16 vs fp16) on the teammate GPU.')
subprocess.run([PY, str(OTHER / 'scripts/discord_notify.py'), 'RSNA: Saturday submissions sent', f.name, '--dm'],
               cwd=OTHER, capture_output=True, text=True, timeout=120)
print(now(), 'EVENT saturday runner done |', summary.replace('\n', ' || '), flush=True)
