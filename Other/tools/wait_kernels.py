"""Wait until every given kernel has ended (status only), then download just their .log files and print the lines
matching a pattern. Never downloads other outputs (e.g. multi-GB caches).
Usage: python wait_kernels.py ACCOUNT MAX_HOURS REGEX REF [REF ...]"""
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org',
        'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org'}
account, hours, pattern, refs = sys.argv[1], float(sys.argv[2]), re.compile(sys.argv[3]), sys.argv[4:]
os.environ['KAGGLE_API_TOKEN'] = dotenv_values(OTHER / '.env')[KEYS[account]]
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
deadline = time.time() + 3600 * hours
status = {}
while time.time() < deadline:
    for ref in refs:
        if status.get(ref) in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'):
            continue
        try:
            st = str(api.kernels_status(ref).status).split('.')[-1]
        except Exception as exc:
            st = 'poll-error ' + type(exc).__name__
        if st != status.get(ref):
            print(time.strftime('%H:%M'), ref, st, flush=True)
        status[ref] = st
    if all(status.get(r) in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED') for r in refs):
        break
    time.sleep(120)
for ref in refs:
    out = Path(tempfile.mkdtemp(prefix='klog_'))
    try:
        api.kernels_output(ref, path=str(out), file_pattern=r'\.log$', force=True, quiet=True)
        text = ''
        for p in out.rglob('*.log'):
            try:
                text += ''.join(e.get('data', '') for e in json.loads(p.read_text(encoding='utf-8')))
            except Exception:
                text += p.read_text(encoding='utf-8', errors='replace')
        hits = [l for l in text.splitlines() if pattern.search(l)]
        print('==', ref, status.get(ref), '|', len(text.splitlines()), 'log lines')
        for l in hits[-8:]:
            print('   ', l[:300])
    except Exception as exc:
        print('==', ref, status.get(ref), '| log download failed', type(exc).__name__, str(exc)[:150])
print('EVENT kernels ended', {r.split('/')[-1]: s for r, s in status.items()}, flush=True)
