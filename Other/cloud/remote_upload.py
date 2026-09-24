"""Run ON THE RENTAL: upload a run's best checkpoint to a private Kaggle dataset.

Used when the rental's link to the local machine is too slow for the watchdog backup.
Writes artifacts/<run>/UPLOAD_DONE or UPLOAD_FAILED with details.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/workspace/rsna')
run = sys.argv[1]
for line in (ROOT / '.env').read_text().splitlines():
    key, _, value = line.partition('=')
    os.environ[key] = value
from kaggle.api.kaggle_api_extended import KaggleApi

out = ROOT / 'artifacts' / run
stage = ROOT / f'upload_{run}'
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir()
shutil.copy2(out / 'best.pt', stage / f'{run}_best.pt')
shutil.copy2(out / 'history.json', stage / 'history.json')
digest = hashlib.sha256((stage / f'{run}_best.pt').read_bytes()).hexdigest()
(stage / 'dataset-metadata.json').write_text(json.dumps({
    'title': f'RSNA Knee {run.capitalize()} Weights', 'id': f'yodharinmonplub/rsna-knee-{run}-weights',
    'licenses': [{'name': 'other'}],
    'description': ('Private weak-label fine-tuned CoAtNet checkpoint from this project. Initial '
                    'model and geometry by Johnathan Wagner / Dread Development. No MRI pixels, reports or '
                    'patient identifiers. sha256 ' + digest)}))
started = time.time()
try:
    api = KaggleApi(); api.authenticate()
    result = api.dataset_create_new(str(stage), public=False, convert_to_csv=False, quiet=True)
    (out / 'UPLOAD_DONE').write_text(json.dumps({'sha256': digest, 'seconds': time.time() - started,
                                                 'result': str(result)}))
except Exception as exc:
    (out / 'UPLOAD_FAILED').write_text(f'{type(exc).__name__}: {exc}'[:500])
