"""Download the Stanford MRNet files from Redivis into data/external/mrnet (token from .env, never printed).

The client's bulk download_files() crashed ('BufferReadError'), so files are fetched one by one with retries;
existing non-empty files are skipped, so the script can simply be re-run.
"""
import base64
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
assert os.environ.get('REDIVIS_API_TOKEN'), 'REDIVIS_API_TOKEN missing in .env'
import redivis

out = ROOT / 'data/external/mrnet'
dataset = redivis.organization('aimi').dataset('mrnet_knee_mri_s:4a2c:v1_0')


def fetch(job):
    file, target = job
    if target.exists() and target.stat().st_size > 0:
        return 'skip'
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(6):
        try:
            data = file.read()
            expected = file.properties.get('md5_hash')
            if expected and base64.b64encode(hashlib.md5(data).digest()).decode() != expected:
                raise ValueError('md5 mismatch')
            temp = target.with_suffix('.part')
            temp.write_bytes(data)
            temp.replace(target)
            return 'ok'
        except Exception as exc:  # noqa: BLE001 - transient API errors are retried
            last = f'{type(exc).__name__}: {str(exc)[:100]}'
            time.sleep(min(60, 2 ** attempt))
    return 'FAIL ' + last


for split in ('valid', 'train'):
    files = list(dataset.table(split).list_files())
    # file.name is only the basename; the plane folder (axial/coronal/sagittal) is in properties['file_name'].
    jobs = [(f, out / split / f.properties['file_name']) for f in files]
    assert len({str(t) for _, t in jobs}) == len(jobs), 'target paths collide'
    print(split, 'files listed', len(jobs), 'example', files[0].properties['file_name'], flush=True)
    done = 0
    with ThreadPoolExecutor(8) as pool:
        for result in pool.map(fetch, jobs):
            done += 1
            if result.startswith('FAIL'):
                print(result, flush=True)
            if done % 250 == 0:
                print(split, done, 'of', len(jobs), flush=True)
total = [p for p in out.rglob('*.npy')]
print('npy files', len(total), 'GB', round(sum(p.stat().st_size for p in total) / 1e9, 2), flush=True)
