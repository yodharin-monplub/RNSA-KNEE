"""Print Kaggle kernel status for given kernel slugs (owner defaults to this account)."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
for slug in sys.argv[1:]:
    ref = slug if '/' in slug else f'yodharinmonplub/{slug}'
    try:
        r = api.kernels_status(ref)
        print(ref, getattr(r, 'status', r), (getattr(r, 'failure_message', '') or '')[:300])
    except Exception as e:
        print(ref, 'ERR', e)
