"""Unpack KneeMRI (Rijeka, Zenodo 14789903), convert to uint8 npz shards and upload as a PRIVATE Kaggle dataset.

Licence CC BY-NC-ND 4.0: used privately for this project only, never redistributed. The volumes are Python
pickles; they are read with a restricted unpickler that only admits numpy array reconstruction (no arbitrary code).
"""
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')

source = ROOT / 'data/external/kneemri'
raw = source / 'volumes'
stage = ROOT / 'data/external/kneemri_stage'
stage.mkdir(exist_ok=True)
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
           ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
           ('numpy._core.multiarray', 'scalar'),
           ('_codecs', 'encode')}  # pure bytes helper that protocol-2 numpy pickles use; cannot execute code


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'blocked {module}.{name}')
        return super().find_class(module, name)


if not raw.exists():
    with py7zr.SevenZipFile(source / 'volumetric_data.7z') as archive:
        archive.extractall(path=raw)
    print('extracted', flush=True)
files = {p.name: p for p in raw.rglob('*.pck')}
meta = pd.read_csv(source / 'metadata.csv')
assert len(meta) == 917 and set(meta.volumeFilename) <= set(files), (len(meta), len(files))
labels, rows = {}, meta.to_dict('records')
for start in range(0, len(rows), 100):
    shard = stage / f'kneemri_{start // 100:02d}.npz'
    arrays = {}
    for row in rows[start:start + 100]:
        key = row['volumeFilename'].replace('.pck', '')
        labels[key] = {k: int(row[k]) for k in ('aclDiagnosis', 'kneeLR', 'roiX', 'roiY', 'roiZ', 'roiHeight', 'roiWidth', 'roiDepth')}
        if shard.exists():
            continue
        volume = NumpyOnly(io.BytesIO(files[row['volumeFilename']].read_bytes()), encoding='latin1').load()
        volume = np.asarray(volume, dtype=np.float32)
        assert volume.ndim == 3, volume.shape
        scale = float(np.percentile(volume, 99.5)) or 1.0
        arrays[key] = np.clip(volume / scale * 255, 0, 255).astype(np.uint8)
    if arrays:
        np.savez_compressed(shard, **arrays)
        print('wrote', shard.name, next(iter(arrays.values())).shape, flush=True)
(stage / 'labels.json').write_text(json.dumps(labels))
print('ACL classes', pd.Series([v['aclDiagnosis'] for v in labels.values()]).value_counts().to_dict(), flush=True)
(stage / 'dataset-metadata.json').write_text(json.dumps({
    'title': 'RSNA Knee External KneeMRI Cache', 'id': 'yodharinmonplub/rsna-knee-external-kneemri-cache',
    'licenses': [{'name': 'other'}],
    'description': 'PRIVATE, not for redistribution. KneeMRI dataset (Clinical Hospital Centre Rijeka; Stajduhar et al. '
                   '2017, doi 10.1016/j.cmpb.2016.12.006; CC BY-NC-ND 4.0) as uint8 shards for this project only.'}))
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print(api.dataset_create_new(str(stage), public=False, convert_to_csv=False, quiet=True))
