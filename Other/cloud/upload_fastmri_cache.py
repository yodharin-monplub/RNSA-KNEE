"""Run ON THE RENTAL: upload artifacts/model62 (compact fastMRI+ image cache) as a PRIVATE Kaggle dataset."""
import json
import os
import time
from pathlib import Path

ROOT = Path('/workspace/rsna')
for line in (ROOT / '.env').read_text().splitlines():
    key, _, value = line.partition('=')
    os.environ[key] = value
from kaggle.api.kaggle_api_extended import KaggleApi

folder = ROOT / 'artifacts/model62'
(folder / 'dataset-metadata.json').write_text(json.dumps({
    'title': 'RSNA Knee External fastMRI Plus Cache', 'id': 'yodharinmonplub/rsna-knee-external-fastmri-plus-cache',
    'licenses': [{'name': 'other'}],
    'description': ('PRIVATE, not for redistribution. uint8 reconstructions of the fastMRI single-coil knee exams '
                    'annotated by fastMRI+ (NYU fastMRI Data Sharing Agreement accepted by the account owner; '
                    'annotations MIT/CC BY 4.0 from microsoft/fastmri-plus). For this project only.')}))
started = time.time()
api = KaggleApi(); api.authenticate()
result = api.dataset_create_new(str(folder), public=False, convert_to_csv=False, quiet=True)
(folder / 'UPLOAD_DONE').write_text(json.dumps({'seconds': time.time() - started, 'result': str(result)}))
print('uploaded', result, flush=True)
