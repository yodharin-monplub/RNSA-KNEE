"""Download and verify the DINO cache shards and the public member weights on the rental."""
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY', ''))
from kaggle.api.kaggle_api_extended import KaggleApi

from download_dense import fetch, output_urls

WEIGHTS = 'pilkwang/rsna-knee-weights'


def main():
    api = KaggleApi(); api.authenticate()
    ready = json.loads((ROOT / 'cloud/model47_ready.json').read_text())
    for part, expected in enumerate(ready['manifests']):
        folder = ROOT / f'data/dino_shards/part{part}'
        folder.mkdir(parents=True, exist_ok=True)
        urls = output_urls(api, f'rsna-knee-dino-cache-{part}')
        response = requests.get(urls['cache_manifest.json'], timeout=(20, 60))
        response.raise_for_status()
        actual = response.json()
        assert actual == expected, 'Cache output changed since readiness verification'
        (folder / 'cache_manifest.json').write_text(json.dumps(actual, indent=2))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda shard: fetch(urls[shard['file']], folder / shard['file'],
                                              shard['bytes'], shard['sha256']), actual['shards']))
        print('Downloaded/verified dino cache part', part, flush=True)
    weights = ROOT / 'data/dino_members'
    weights.mkdir(parents=True, exist_ok=True)
    for name, size in ready['member_files'].items():
        target = weights / name
        for attempt in range(5):
            try:
                if target.is_file() and target.stat().st_size == size:
                    break
                api.dataset_download_file(WEIGHTS, name, path=str(weights), force=True, quiet=True)
                assert target.stat().st_size == size, 'member size mismatch'
                break
            except Exception as exc:  # noqa: BLE001 - retry transient Kaggle failures
                print('Member retry', name, attempt + 1, type(exc).__name__, flush=True)
                if attempt == 4:
                    raise RuntimeError('Member download failed: ' + name) from None
                time.sleep(5)
        print('Verified member', name, flush=True)
    (weights / 'manifest.json').write_text(json.dumps(ready['member_manifest']))
    print('Members ready', len(ready['member_files']), flush=True)


if __name__ == '__main__':
    main()
