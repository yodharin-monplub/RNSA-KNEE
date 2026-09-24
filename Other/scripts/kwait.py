"""Poll Kaggle kernels until all are terminal (or timeout), then print final status."""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

TERMINAL = ('COMPLETE', 'ERROR', 'CANCEL')
api = KaggleApi(); api.authenticate()
refs = [s if '/' in s else f'yodharinmonplub/{s}' for s in sys.argv[1:]]
deadline = time.time() + float(os.environ.get('KWAIT_SECONDS', 3 * 3600))
while True:
    states = {}
    for ref in refs:
        try:
            r = api.kernels_status(ref)
            states[ref] = (str(getattr(r, 'status', r)), getattr(r, 'failure_message', '') or '')
        except Exception as e:
            states[ref] = ('QUERY_ERROR', str(e)[:200])
    if all(any(t in s for t in TERMINAL) for s, _ in states.values()) or time.time() > deadline:
        break
    time.sleep(120)
for ref, (s, msg) in states.items():
    print(time.strftime('%H:%M:%S'), ref, s, msg[:400])
