"""Prepare a batch of LIGHT-gated Speedy + fleet-leg submissions: for each spec build the notebook with
run_speedy_fleet_v2.py --dummy light, write Model/<name>/readme.txt and artifacts/<name>_notebook/message.txt, push the
dummy run on the main account (retrying while GPU slots are busy), then verify every run with tick_light.py
--no-submit. Submitting is left to light_runner.py (after the 07:00 daily reset).
Spec file: JSON list of {"name", "title" (<= 50 chars), "weight", "leg" (v3|v4), "fleet" (python literal string),
"message", "readme"}.
Usage: python prep_light_batch.py SPEC.json"""
import json
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
MODEL = OTHER.parent / 'Model'
PY = str(OTHER / '.venv/Scripts/python.exe')
specs = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for s in specs:
    assert len(s['title']) <= 50 and s['leg'] in ('v3', 'v4') and 0 < float(s['weight']) < 1, s
    assert not (MODEL / s['name']).exists() or not any((MODEL / s['name']).iterdir()), f"{s['name']} exists"


def run(args, timeout=1800):
    r = subprocess.run([PY] + args, cwd=OTHER, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


for s in specs:
    code, out = run(['scripts/run_speedy_fleet_v2.py', '--name', s['name'], '--title', s['title'], '--weight',
                     str(s['weight']), '--leg', s['leg'], '--dummy', 'light', '--fleet', s['fleet']])
    assert code == 0 and 'built ' in out, out[-1500:]
    build = (f"python scripts/run_speedy_fleet_v2.py --name {s['name']} --title \"{s['title']}\" --weight {s['weight']} "
             f"--leg {s['leg']} --dummy light --fleet \"{s['fleet']}\"")
    (MODEL / s['name'] / 'readme.txt').write_text(s['readme'].rstrip() + '\nBuild (from Other/): ' + build +
                                                  '\nDummy run: light (scripts/light_gate.py).\nScore: pending.\n',
                                                  encoding='utf-8')
    (OTHER / f"artifacts/{s['name']}_notebook/message.txt").write_text(s['message'], encoding='utf-8')
    print(time.strftime('%H:%M'), 'built', s['name'], flush=True)
for s in specs:
    code, out = run([str(SCRATCH / 'push_any.py'), 'yodharinmonplub', f"artifacts/{s['name']}_notebook", 'NvidiaTeslaT4',
                     '120'], timeout=3 * 3600)
    print(time.strftime('%H:%M'), s['name'], 'push:', (out.strip().splitlines() or ['?'])[-1][:160], flush=True)
pending, states = [s['name'] for s in specs], {}
end = time.time() + 4 * 3600
while pending and time.time() < end:
    for name in list(pending):
        code, out = run([str(SCRATCH / 'tick_light.py'), name, 'not submitted here', '--no-submit'], timeout=900)
        last = (out.strip().splitlines() or [''])[-1]
        if last.startswith(('STATE VERIFIED', 'STATE VERIFY_FAILED', 'STATE FAILED')):
            states[name] = last
            pending.remove(name)
            print(time.strftime('%H:%M'), name, '|', last[:260], flush=True)
    if pending:
        time.sleep(90)
print('EVENT light batch prepared', states, '| not finished', pending, flush=True)
