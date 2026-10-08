"""Push a kernel folder with retries (flaky network / busy slots). Usage: push_any.py ACCOUNT FOLDER [ACC] [MAX_MIN]
ACC e.g. NvidiaTeslaT4 / TpuV5E8 / none (CPU). Never for training on the main account (AGENTS.md)."""
import os, sys, time
from pathlib import Path
from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', 'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org',
        'actuarialtutor': 'KAGGLE_API_KEY_krittanutsomtung@actuarialtutor.org', 'aikayui': 'KAGGLE_API_KEY_aikayui@actuarialtutor.org'}
account, folder = sys.argv[1], (OTHER / sys.argv[2]).resolve()
acc = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != 'none' else None
limit = time.time() + 60 * float(sys.argv[4] if len(sys.argv) > 4 else 30)
os.environ['KAGGLE_API_TOKEN'] = dotenv_values(OTHER / '.env')[KEYS[account]]
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
while time.time() < limit:
    try:
        text = str(api.kernels_push(str(folder), acc=acc))
    except Exception as exc:
        text = f'exception {type(exc).__name__}: {str(exc)[:160]}'
    if '"versionNumber"' in text and '"versionNumber": null' not in text:
        (folder / 'run_response.txt').write_text(text)
        print(time.strftime('%H:%M'), 'PUSHED', text[:220], flush=True); sys.exit(0)
    print(time.strftime('%H:%M'), 'not yet:', text[:160], flush=True)
    time.sleep(60)
print('TIMEOUT', flush=True); sys.exit(1)
