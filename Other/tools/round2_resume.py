"""Resume the round-2 chain at step 3 (2026-10-11): round2_chain.py stopped after model147's five folds were collected
because Other/data/llm_relabel was still a junction into the pre-move OneDrive folder (fixed). Builds the round-2 targets
from the collected folds, pushes both kits, then the round-2 students model148 and model149 (teammate account).
Usage: python tools/round2_resume.py [MAX_HOURS]"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
SCRATCH = Path(__file__).resolve().parent
DEADLINE = time.time() + 3600 * float(sys.argv[1] if len(sys.argv) > 1 else 120)
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org',
        'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org'}
ENV = dotenv_values(OTHER / '.env')
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi


def api_for(account):
    os.environ['KAGGLE_API_TOKEN'] = ENV[KEYS[account]]
    a = KaggleApi(); a.authenticate()
    return a


def now():
    return time.strftime('%m-%d %H:%M')


def dm(subject, body):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(body)
    subprocess.run([PY, str(OTHER / 'scripts/discord_notify.py'), subject, f.name, '--dm'], cwd=OTHER,
                   capture_output=True, text=True, timeout=120)


def run(args, timeout=3600):
    r = subprocess.run([PY] + args, cwd=OTHER, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


def stop(msg):
    print(now(), 'EVENT round-2 chain STOPPED |', msg, flush=True)
    dm('RSNA round 2: chain stopped', msg)
    sys.exit(1)


def push(account, folder):
    if (folder / 'run_response.txt').exists():   # pushed earlier (e.g. folds 3-4 on 2026-10-07 with that week's spare TPU)
        print(now(), 'already pushed, not pushing again:', folder.name, flush=True)
        return
    api = api_for(account)
    while time.time() < DEADLINE:
        try:
            text = str(api.kernels_push(str(folder), acc='TpuV5E8'))
        except Exception as exc:
            text = f'exception {type(exc).__name__}: {str(exc)[:200]}'
        if re.findall(r'"invalid\w+Sources": \[([^\]]+)\]', text):
            stop(f'Kaggle rejected sources for {folder.name}: {text[:500]}')
        if '"versionNumber"' in text and '"versionNumber": null' not in text and '"error": null' in text:
            (folder / 'run_response.txt').write_text(text)
            print(now(), 'EVENT pushed', folder.name, flush=True)
            return
        print(now(), 'push not accepted yet', folder.name, text[:160], flush=True)
        time.sleep(300)
    stop(f'timeout pushing {folder.name}')


def wait_end(account, ref):
    api = api_for(account)
    last = None
    while time.time() < DEADLINE:
        try:
            st = str(api.kernels_status(ref).status).split('.')[-1]
        except Exception as exc:
            st = 'poll-error ' + type(exc).__name__
        if st != last:
            print(now(), ref, st, flush=True); last = st
        if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCEL_REQUESTED'):
            return st
        time.sleep(300)
    stop(f'timeout waiting for {ref}')


st_a, st_b = 'COMPLETE', 'COMPLETE'
folds147 = sorted(str(p.parent.relative_to(OTHER)).replace(chr(92), '/') for p in
                  OTHER.glob('artifacts/remote_model147-*/model147/fold*/TRAINING_COMPLETE'))
folds110 = sorted(str(p.parent.relative_to(OTHER)).replace(chr(92), '/') for p in
                  OTHER.glob('artifacts/remote_model110-*/model110/fold*/TRAINING_COMPLETE'))
if len(folds147) != 5 or len(folds110) != 5:
    stop(f'model147 sessions ended ({st_a}, {st_b}) with {len(folds147)} of 5 folds complete: {folds147}')

# 3) round-2 targets + kits + round-2 students
code, out = run(['scripts/make_pseudo_targets.py', '--labels', 'data/llm_relabel/targets_g38f_low.csv', '--alpha', '0.35',
                 '--runs'] + folds147 + folds110 + ['--out', 'data/llm_relabel/targets_v2_oof147_110_35.csv'], 900)
if code != 0:
    stop(f'round-2 targets failed:\n{out[-1200:]}')
report = ' | '.join(l.strip() for l in out.splitlines() if 'coverage' in l or 'mix alpha' in l or 'alone' in l)
print(now(), 'round-2 targets |', report, flush=True)
for account in ('krittanutsomtuas', 'yodharinmonplub'):
    for attempt in range(6):
        code, out = run(['scripts/build_trainer_kit.py', '--account', account, '--push', 'round-2 targets (model147 + model110 OOF)'], 1800)
        if '"status": "Ok"' in out:
            break
        time.sleep(120)
    else:
        stop(f'kit push failed for {account}:\n{out[-800:]}')
time.sleep(600)
for name, account in (('model148', 'krittanutsomtuas'), ('model149', 'krittanutsomtuas')):
    code, out = run(['scripts/run_tpu_train.py', '--name', name, '--plan', 'full', '--cache', 's96', '--account', account], 600)
    m = re.search(r'built (.+?_notebook) \|', out)
    if not m:
        stop(f'{name} build failed:\n{out[-800:]}')
    push(account, Path(m.group(1)))
dm('RSNA round 2: students pushed', f'5-fold student done ({len(folds147)} folds). Round-2 targets: {report}\n'
   'Pushed model148 (main TPU) and model149 (teammate TPU), ~2.3 h each after the queue.')
print(now(), 'EVENT round-2 chain done', flush=True)
