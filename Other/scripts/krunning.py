"""List this account's most recently run kernels with their status (to see what holds the 2 GPU batch slots)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
for kernel in api.kernels_list(mine=True, sort_by='dateRun', page_size=15):
    ref = getattr(kernel, 'ref', '') or ''
    if not ref:
        url = getattr(kernel, 'url', '') or ''
        ref = url.split('/code/')[-1] if '/code/' in url else ''
    if not ref:
        print('no ref for', getattr(kernel, 'title', kernel))
        continue
    try:
        status = str(api.kernels_status(ref).status)
    except Exception as exc:  # noqa: BLE001
        status = f'ERR {type(exc).__name__}'
    print(ref, '|', getattr(kernel, 'last_run_time', ''), '|', status)
