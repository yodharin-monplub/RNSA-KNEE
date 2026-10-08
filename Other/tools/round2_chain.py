"""Round-2 pseudo-label chain (armed 2026-10-07; acts from Sat 2026-10-10).
1) wait for model143's public score (model113 = round-1 student ALONE, submitted by saturday_runner3.py); go on only if
   it beats model104 alone (0.942) - otherwise DM and stop;
2) push the pre-built 5-fold student kernels (model147 folds 0,1,2 on the teammate account, folds 3,4 on the main
   account; TPU quotas refresh Sat 07:00), retrying while slots are busy;
3) wait for both, collect, build round-2 targets (alpha 0.35 x labels + 0.65 x mean OOF of model147 and model110 folds),
   push both kits, then push the round-2 students model148 (main) and model149 (teammate) in parallel.
DMs every step. Usage: python round2_chain.py [MAX_HOURS]"""
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


# 1) model143's public score
message = (OTHER / 'artifacts/model143_notebook/message.txt').read_text(encoding='utf-8').strip()
main_api = api_for('yodharinmonplub')
score = None
while time.time() < DEADLINE and score is None:
    try:
        for s in main_api.competition_submissions('rsna-knee-abnormality-detection') or []:
            if (getattr(s, 'description', '') or '') == message:
                pub = getattr(s, 'public_score', None)
                err = (getattr(s, 'error_description', '') or '').strip()
                if err:
                    stop(f'model143 submission failed: {err[:300]}')
                if pub not in (None, ''):
                    score = float(pub)
                break
    except Exception as exc:
        print(now(), 'poll error', type(exc).__name__, flush=True)
    if score is None:
        time.sleep(600)
if score is None:
    stop('no model143 score before the deadline')
print(now(), 'model143 (student alone) public', score, flush=True)
if score < 0.943:
    stop(f'model143 = model113 alone scored {score} (model104 alone 0.942): the student does not beat the plain model '
         f'on the LB -> round 2 skipped. Next week\'s TPU goes to other ideas (will DM the plan).')
dm('RSNA: the student helps on the LB -> round 2 starts',
   f'model143 (the pseudo-label student alone) scored {score} vs 0.942 for the plain model104 alone. Pushing round 2 now: '
   f'5-fold student model147 (folds 0-2 teammate TPU, 3-4 main TPU), then two round-2 students.')

# 2) the 5-fold student. A session can fail before our code runs (2026-10-07/08: Kaggle's "ERRORED_MOUNTING_DATASET"
#    bug on notebook-output inputs - empty log, status ERROR, no TPU time charged): re-push such a session up to twice.
SESSIONS = [('krittanutsomtuas', 'model147-f012', OTHER / 'artifacts/tpu_model147_f012_notebook', [0, 1, 2]),
            ('yodharinmonplub', 'model147-f34', OTHER / 'artifacts/tpu_model147_f34_notebook', [3, 4])]


def folds_done(k_list):
    return all(any(OTHER.glob(f'artifacts/remote_model147-*/model147/fold{k}/TRAINING_COMPLETE')) for k in k_list)


for account, suffix, folder, k_list in SESSIONS:
    push(account, folder)
statuses = {}
for account, suffix, folder, k_list in SESSIONS:
    for attempt in range(3):
        statuses[suffix] = wait_end(account, f'{account}/rsna-knee-tpu-train-{suffix}')
        for _ in range(6):
            code, out = run(['scripts/collect_tpu_run.py', suffix, '--account', account], 3 * 3600)
            if code == 0 and 'downloaded 0 files' not in out:
                break
            time.sleep(240)
        if folds_done(k_list):
            break
        print(now(), f'{suffix} ended {statuses[suffix]} without folds {k_list}; re-pushing (attempt {attempt + 1})', flush=True)
        dm('RSNA round 2: TPU session failed, retrying', f'{suffix} ended {statuses[suffix]} without its folds; re-pushed.')
        (folder / 'run_response.txt').unlink(missing_ok=True)
        push(account, folder)
        time.sleep(600)   # let the status move past the previous version's ERROR before polling again
st_a, st_b = statuses.get('model147-f012'), statuses.get('model147-f34')
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
for name, account in (('model148', 'yodharinmonplub'), ('model149', 'krittanutsomtuas')):
    code, out = run(['scripts/run_tpu_train.py', '--name', name, '--plan', 'full', '--cache', 's96', '--account', account], 600)
    m = re.search(r'built (.+?_notebook) \|', out)
    if not m:
        stop(f'{name} build failed:\n{out[-800:]}')
    push(account, Path(m.group(1)))
dm('RSNA round 2: students pushed', f'5-fold student done ({len(folds147)} folds). Round-2 targets: {report}\n'
   'Pushed model148 (main TPU) and model149 (teammate TPU), ~2.3 h each after the queue.')
print(now(), 'EVENT round-2 chain done', flush=True)
