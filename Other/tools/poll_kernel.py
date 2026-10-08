"""Wait for one kernel run to end, download its outputs and print the log lines matching a pattern.
Usage: python poll_kernel.py ACCOUNT OWNER/SLUG OUT_DIR [REGEX] [MAX_HOURS]"""
import json, os, re, sys, time
from pathlib import Path
from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', 'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org',
        'actuarialtutor': 'KAGGLE_API_KEY_krittanutsomtung@actuarialtutor.org', 'aikayui': 'KAGGLE_API_KEY_aikayui@actuarialtutor.org'}
account, ref, out = sys.argv[1], sys.argv[2], OTHER / sys.argv[3]
pattern = re.compile(sys.argv[4] if len(sys.argv) > 4 else 'SUMMARY|Traceback|Error')
deadline = time.time() + 3600 * float(sys.argv[5] if len(sys.argv) > 5 else 6)
os.environ['KAGGLE_API_TOKEN'] = dotenv_values(OTHER / '.env')[KEYS[account]]
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
last = None
while time.time() < deadline:
    try:
        st = str(api.kernels_status(ref).status).split('.')[-1]
    except Exception as exc:
        st = 'poll-error ' + type(exc).__name__
    if st != last:
        print(time.strftime('%H:%M:%S'), st, flush=True); last = st
    if st in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
        break
    time.sleep(60)
out.mkdir(parents=True, exist_ok=True)
for attempt in range(5):
    try:
        api.kernels_output(ref, path=str(out), force=True, quiet=True)
        break
    except Exception as exc:
        print('output error', type(exc).__name__, flush=True); time.sleep(60)
for f in out.glob('*.log'):
    raw = f.read_text(encoding='utf-8', errors='replace')
    try:
        text = ''.join(e.get('data', '') for e in json.loads(raw))
    except Exception:
        text = raw
    for line in text.splitlines():
        if pattern.search(line):
            print(line[:3000], flush=True)
print('done', last, flush=True)
