"""Print both accounts' Kaggle quota objects (GPU + TPU) and the status of given kernels.
usage: quota_both.py [account:kernel_ref ...]"""
import os, sys
from pathlib import Path
from dotenv import dotenv_values

OTHER = Path(__file__).resolve().parents[1]   # the project's Other/ folder
env = dotenv_values(OTHER / '.env')
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', 'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org', 'actuarialtutor': 'KAGGLE_API_KEY_krittanutsomtung@actuarialtutor.org', 'aikayui': 'KAGGLE_API_KEY_aikayui@actuarialtutor.org'}
os.environ['KAGGLE_CONFIG_DIR'] = str(OTHER / '.kaggle')
wanted = [a.split(':', 1) for a in sys.argv[1:]]
for acc, key in KEYS.items():
    os.environ['KAGGLE_API_TOKEN'] = env[key]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    try:
        q = api.quota_view()
        print(acc, 'GPU', str(q.gpu_quota).replace('\n', ' '))
        print(acc, 'TPU', str(q.tpu_quota).replace('\n', ' '))
        print(acc, 'refresh', getattr(q, 'quota_refresh_time', None) or getattr(q, 'quotaRefreshTime', None))
    except Exception as e:
        print(acc, 'quota error', type(e).__name__, e)
    for a, ref in wanted:
        if a != acc:
            continue
        try:
            s = api.kernels_status(ref)
            print(acc, ref, s.status, getattr(s, 'failure_message', None))
        except Exception as e:
            print(acc, ref, 'status error', type(e).__name__, str(e)[:200])
