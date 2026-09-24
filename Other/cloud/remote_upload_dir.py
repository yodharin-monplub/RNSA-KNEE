"""Run ON THE RENTAL: upload a whole artifact directory as a private Kaggle dataset.

Used for multi-file outputs (e.g. the twenty fine-tuned DINO members) when the rental's uplink to the
local machine is too slow for the watchdog backup.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/rsna')
run = sys.argv[1]
slug = sys.argv[2]
for line in (ROOT / '.env').read_text().splitlines():
    key, _, value = line.partition('=')
    os.environ[key] = value
from kaggle.api.kaggle_api_extended import KaggleApi

folder = ROOT / 'artifacts' / run
digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.pt'))}
(folder / 'SHA256.json').write_text(json.dumps(digests, indent=2))
(folder / 'dataset-metadata.json').write_text(json.dumps({
    'title': f'RSNA Knee {run.capitalize()} Members', 'id': f'yodharinmonplub/{slug}',
    'licenses': [{'name': 'other'}],
    'description': (f'Private {run} checkpoints fine-tuned by this project from the public DINOv2 members '
                    '(pilkwang/rsna-knee-weights). No MRI pixels, reports or patient identifiers.')}))
started = time.time()
try:
    api = KaggleApi(); api.authenticate()
    result = api.dataset_create_new(str(folder), public=False, convert_to_csv=False, quiet=True,
                                    dir_mode='zip')
    (folder / 'UPLOAD_DONE').write_text(json.dumps({'seconds': time.time() - started, 'result': str(result),
                                                    'files': len(digests)}))
    print('uploaded', result, flush=True)
except Exception as exc:
    (folder / 'UPLOAD_FAILED').write_text(f'{type(exc).__name__}: {exc}'[:500])
    raise
