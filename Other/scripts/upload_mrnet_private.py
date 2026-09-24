"""Pack MRNet (already uint8 .npy, 3 planes per exam) into npz shards and upload as a PRIVATE Kaggle dataset.

Stanford's research-use agreement forbids redistribution: the dataset is private and for this project only.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

source = ROOT / 'data/external/mrnet'
stage = ROOT / 'data/external/mrnet_stage'
stage.mkdir(exist_ok=True)
labels = {}
for split in ('train', 'valid'):
    frames = []
    for task in ('abnormal', 'acl', 'meniscus'):
        # Redivis exported the first data row as a header ("_0000","_0"): read it as data and strip the underscore.
        frame = pd.read_csv(source / f'{split}-{task}.csv', header=None, names=['exam', task], dtype=str)
        frame = frame.apply(lambda column: column.str.lstrip('_'))
        frame[task] = frame[task].astype(int)
        frames.append(frame.set_index('exam'))
    table = pd.concat(frames, axis=1)
    table.index = [f'{int(i):04d}' for i in table.index]
    for exam, row in table.iterrows():
        labels[exam] = {'split': split, **{k: int(v) for k, v in row.items()}}
    exams = sorted(table.index)
    assert all((source / split / 'axial' / f'{e}.npy').exists() for e in exams), 'label row without image'
    assert len(exams) == len(list((source / split / 'axial').glob('*.npy'))), 'image without label row'
    for start in range(0, len(exams), 100):
        shard = stage / f'mrnet_{split}_{start // 100:02d}.npz'
        if shard.exists():
            continue
        arrays = {}
        for exam in exams[start:start + 100]:
            for plane in ('axial', 'coronal', 'sagittal'):
                volume = np.load(source / split / plane / f'{exam}.npy')
                assert volume.dtype == np.uint8 and volume.ndim == 3, (exam, plane, volume.dtype, volume.shape)
                arrays[f'{exam}_{plane}'] = volume
        np.savez_compressed(shard, **arrays)
        print('wrote', shard.name, flush=True)
assert len(labels) == 1250, len(labels)
(stage / 'labels.json').write_text(json.dumps(labels))
(stage / 'dataset-metadata.json').write_text(json.dumps({
    'title': 'RSNA Knee External MRNet Cache', 'id': 'yodharinmonplub/rsna-knee-external-mrnet-cache',
    'licenses': [{'name': 'other'}],
    'description': 'PRIVATE, not for redistribution. Stanford MRNet knee MRI (research use agreement accepted by the '
                   'account owner), repacked as npz shards for this project only.'}))
api = KaggleApi(); api.authenticate()
print(api.dataset_create_new(str(stage), public=False, convert_to_csv=False, quiet=False))
