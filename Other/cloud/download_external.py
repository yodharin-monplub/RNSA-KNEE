"""Run ON THE RENTAL: fetch the three PRIVATE external-data caches from Kaggle into data/external_cache."""
import os
from pathlib import Path

ROOT = Path('/workspace/rsna')
for line in (ROOT / '.env').read_text().splitlines():
    key, _, value = line.partition('=')
    os.environ[key] = value
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
for name, slug in (('mrnet', 'rsna-knee-external-mrnet-cache'), ('fastmri', 'rsna-knee-external-fastmri-plus-cache'),
                   ('kneemri', 'rsna-knee-external-kneemri-cache')):
    target = ROOT / 'data/external_cache' / name
    target.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(f'yodharinmonplub/{slug}', path=str(target), unzip=True, quiet=True)
    files = list(target.rglob('*.npz'))
    print(name, 'shards', len(files), 'GB', round(sum(f.stat().st_size for f in files) / 1e9, 2), flush=True)
    assert files
