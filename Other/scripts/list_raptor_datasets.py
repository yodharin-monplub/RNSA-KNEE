"""List the public Raptor checkpoint datasets (owner dreaddevelopment) with their update times and sizes."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
seen = set()
for page in range(1, 4):
    rows = api.dataset_list(user='dreaddevelopment', page=page)
    if not rows:
        break
    for d in rows:
        if d.ref in seen:
            continue
        seen.add(d.ref)
        print(d.ref, '|', getattr(d, 'title', ''), '|', getattr(d, 'last_updated', getattr(d, 'lastUpdated', '')),
              '|', getattr(d, 'total_bytes', getattr(d, 'totalBytes', '')), '|', (getattr(d, 'subtitle', '') or '')[:160])
