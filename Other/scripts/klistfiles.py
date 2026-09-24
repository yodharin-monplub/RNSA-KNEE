"""List a kernel's output files without downloading them (tries the SDK calls that exist in this kaggle version)."""
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
ref = sys.argv[1] if '/' in sys.argv[1] else f'yodharinmonplub/{sys.argv[1]}'
for name in ('kernels_list_files', 'kernel_list_files', 'kernels_output_list'):
    fn = getattr(api, name, None)
    if fn is None:
        continue
    try:
        result = fn(ref)
        files = getattr(result, 'files', result)
        print(name, [(getattr(f, 'name', f), getattr(f, 'size', getattr(f, 'total_bytes', ''))) for f in files])
        break
    except Exception as exc:
        print(name, 'ERR', type(exc).__name__, str(exc)[:300])
else:
    print('no listing call available in this SDK:', [n for n in dir(api) if 'kernel' in n.lower()])
