"""Publish the report-label target files as a new version of the private dataset rsna-knee-gemini-relabels.

Keeps partitions.csv and every earlier targets file so existing kernels keep working; verifies each targets file
(4,407 unique StudyInstanceUIDs equal to train.csv's, 12 label columns, finite values in [0, 1]) before upload.
Usage: python scripts/upload_relabels.py "version message"
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FILES = {'partitions.csv': ROOT / 'artifacts/partitions/partitions.csv',
         'targets_g38f_low.csv': ROOT / 'data/llm_relabel/targets_g38f_low.csv',
         'targets_g38f_low_v2.csv': ROOT / 'data/llm_relabel/targets_g38f_low_v2.csv',
         'targets_g38f_low_v1v2mean.csv': ROOT / 'data/llm_relabel/targets_g38f_low_v1v2mean.csv',
         'targets_v1_oof50.csv': ROOT / 'data/llm_relabel/targets_v1_oof50.csv',
         # model71: same mix, but the out-of-fold teacher is the 5-fold ResNet34 (each fold trained on 4/5 of
         # the studies) instead of the two half-data models that failed on 2026-09-20.
         'targets_v1_oof5f50.csv': ROOT / 'data/llm_relabel/targets_v1_oof5f50.csv',
         'targets_v1_oof5f25.csv': ROOT / 'data/llm_relabel/targets_v1_oof5f25.csv'}
train_ids = set(pd.read_csv(ROOT / 'data/train.csv').StudyInstanceUID)
import time
folder = ROOT / f'artifacts/relabels_dataset_{int(time.time())}'  # fresh folder: OneDrive can lock the old one
folder.mkdir(parents=True)
for name, path in FILES.items():
    if name.startswith('targets'):
        frame = pd.read_csv(path)
        values = frame[LABELS].to_numpy(float)
        assert frame.StudyInstanceUID.is_unique and set(frame.StudyInstanceUID) == train_ids and len(frame) == 4407, name
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), name
        print(name, 'ok, mean probability', round(float(values.mean()), 4))
    shutil.copyfile(path, folder / name)
(folder / 'dataset-metadata.json').write_text(json.dumps(
    {'id': 'yodharinmonplub/rsna-knee-gemini-relabels', 'title': 'RSNA Knee Gemini Relabels',
     'licenses': [{'name': 'other'}]}), encoding='utf-8')
api = KaggleApi(); api.authenticate()
print(api.dataset_create_version(str(folder.resolve()), version_notes=sys.argv[1], quiet=True, dir_mode='skip'))
