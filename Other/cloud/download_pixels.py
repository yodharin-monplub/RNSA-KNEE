"""Run on the rental: fetch the pinned compact cache and verify every hash."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
load_dotenv('.env')
os.environ.setdefault('KAGGLEHUB_CACHE', str(Path('.cache/kagglehub').resolve()))
import kagglehub

root = Path('data/pixels')
root.mkdir(parents=True, exist_ok=True)
ref = 'dmitriigluzdov/rsna-knee-uint8-224-9-c130/versions/1'
for name in ['SPEC.json', 'studies.csv', 'slot_mask.npy', 'NOTICE.md', 'LICENSE-APACHE-2.0.txt']:
    kagglehub.dataset_download(ref, path=name, output_dir=str(root.resolve()))
spec = json.loads((root / 'SPEC.json').read_text())
def fetch_verified(item):
    name, expected = item
    file = root / name
    if not file.exists():
        kagglehub.dataset_download(ref, path=name, output_dir=str(root.resolve()))
    with file.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        kagglehub.dataset_download(ref, path=name, output_dir=str(root.resolve()), force_download=True)
        with file.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError(f'Checksum mismatch: {name}')
    print(f'Verified {name}', flush=True)


with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch_verified, spec['sha256'].items()))
print('PIXELS_READY', flush=True)
