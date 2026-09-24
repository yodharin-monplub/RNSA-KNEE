"""Wait until the given competition submission refs are scored (or error), then print them."""
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

api = KaggleApi(); api.authenticate()
refs = {int(r) for r in sys.argv[1:]}
deadline = time.time() + float(os.environ.get('SWAIT_SECONDS', 6 * 3600))
while True:
    try:
        rows = {s.ref: s for s in api.competition_submissions('rsna-knee-abnormality-detection') if s.ref in refs}
        done = all(r in rows and 'PENDING' not in str(rows[r].status) for r in refs)
    except Exception as e:
        rows, done = {}, False
        print('query error', e, flush=True)
    if done or time.time() > deadline:
        break
    time.sleep(300)
for r in sorted(refs):
    s = rows.get(r)
    print(r, s and s.status, s and s.public_score, s and (s.description or '')[:60], s and getattr(s, 'error_description', ''))
