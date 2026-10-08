"""Wait for a TPU session (teammate or main account) to end, collect its outputs, DM a summary, print an EVENT and exit.
Usage: python wait_collect.py SUFFIX [MAX_HOURS] [ACCOUNT]   (ACCOUNT: krittanutsomtuas (default) | yodharinmonplub)"""
import os, subprocess, sys, tempfile, time
from pathlib import Path
from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
PY = str(OTHER / '.venv/Scripts/python.exe')
SUFFIX = sys.argv[1]
DEADLINE = time.time() + 3600 * float(sys.argv[2] if len(sys.argv) > 2 else 30)
ACCOUNT = sys.argv[3] if len(sys.argv) > 3 else 'krittanutsomtuas'
KEYS = {'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org',
        'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org'}
os.environ['KAGGLE_API_TOKEN'] = dotenv_values(OTHER / '.env')[KEYS[ACCOUNT]]
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
ref = f'{ACCOUNT}/rsna-knee-tpu-train-{SUFFIX}'
last = None
while time.time() < DEADLINE:
    try:
        st = str(api.kernels_status(ref).status).split('.')[-1]
    except Exception as exc:
        st = 'poll-error ' + type(exc).__name__
    if st != last:
        print(time.strftime('%m-%d %H:%M'), 'status', st, flush=True); last = st
    if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCEL_REQUESTED'):
        break
    time.sleep(300)
else:
    print('EVENT TIMEOUT', flush=True); sys.exit(1)
summary = ''
for attempt in range(8):
    r = subprocess.run([PY, 'scripts/collect_tpu_run.py', SUFFIX, '--account', ACCOUNT], cwd=OTHER,
                       capture_output=True, text=True, timeout=3 * 3600)
    summary = (r.stdout + r.stderr)[-3000:]
    if r.returncode == 0 and 'downloaded 0 files' not in summary:
        break
    time.sleep(240)
with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(f'{ref} ended ({last}). Collected:\n{summary[-1800:]}')
subprocess.run([PY, str(OTHER / 'scripts/discord_notify.py'), f'RSNA TPU: {SUFFIX} ended ({last})', f.name, '--dm'],
               cwd=OTHER, capture_output=True, text=True, timeout=120)
print(time.strftime('%m-%d %H:%M'), 'EVENT collected', SUFFIX, last, '|', summary.replace('\n', ' || ')[-1500:], flush=True)
