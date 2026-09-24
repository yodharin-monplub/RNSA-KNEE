"""Download the complete visible test manifest onto the rental for a smoke run."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('.env')
os.environ.setdefault('KAGGLEHUB_CACHE', str(Path('.cache/kagglehub').resolve()))
import kagglehub

manifest = json.loads(Path('cloud/test_files.json').read_text())
assert sum(f['bytes'] for f in manifest) < 2_000_000_000


def download(file):
    kagglehub.competition_download('rsna-knee-abnormality-detection', path=file['name'],
                                   output_dir=str(Path('data').resolve()))
    if not (Path('data') / file['name']).exists():
        raise RuntimeError('Missing downloaded test file')


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(download, manifest))
print(f'TEST_READY: {len(manifest)} DICOMs', flush=True)
