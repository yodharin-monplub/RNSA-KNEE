"""Rebuild the private trainer kit dataset (yodharinmonplub/rsna-knee-trainer-kit) from this project's files: flat copies
named as in layout.json, so a Kaggle kernel can restore the project layout (see scripts/run_tpu_train.py).
No MRI pixels, reports or identifiers: code, split/label tables and report-derived soft targets only.

Usage: python scripts/build_trainer_kit.py [--push "version notes"]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / 'artifacts/trainer_kit'
load_dotenv(ROOT / '.env')
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')

# flat name in the dataset -> path relative to the project root (Other/); model configs live in Model/<name>/config.json
LAYOUT = {
    'train_xla.py': 'scripts/train_xla.py',
    'tpu_common.py': 'scripts/tpu_common.py',
    'xla_data.py': 'scripts/xla_data.py',
    'knee__init__.py': 'knee/__init__.py',
    'knee_dense.py': 'knee/dense.py',
    'knee_external.py': 'knee/external.py',
    'cloud__init__.py': 'cloud/__init__.py',
    'prepare_dense_cache.py': 'cloud/prepare_dense_cache.py',
    'partitions.csv': 'artifacts/partitions/partitions.csv',
    'labels.json': 'artifacts/partitions/labels.json',
    'targets_g38f_low.csv': 'data/llm_relabel/targets_g38f_low.csv',
    'targets_v1_oof5f50.csv': 'data/llm_relabel/targets_v1_oof5f50.csv',
}
MODEL_CONFIGS = ['model98', 'model100', 'model104', 'model109', 'model110']
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_YODHAIRNMONPLUB', 'krittanutsomtuas': 'KAGGLE_API_KEY_WARIN', 'actuarialtutor': 'KAGGLE_API_KEY_KRITTANUT'}


def build():
    layout = dict(LAYOUT)
    for name, rel in LAYOUT.items():
        if (ROOT / rel).exists():
            shutil.copy2(ROOT / rel, KIT / name)
        else:   # cloud/ is a namespace package locally; the kernel gets an empty __init__.py
            assert name.endswith('__init__.py'), rel
            (KIT / name).write_text('')
    for model in MODEL_CONFIGS:
        shutil.copy2(ROOT.parent / 'Model' / model / 'config.json', KIT / f'config_{model}.json')
        layout[f'config_{model}.json'] = f'{model}/config.json'
    (KIT / 'layout.json').write_text(json.dumps(layout, indent=2))
    return layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', metavar='NOTES')
    parser.add_argument('--account', default='yodharinmonplub', choices=sorted(KEYS),
                        help='owner of the kit dataset (the teammate account needs its own copy)')
    parser.add_argument('--create', action='store_true', help='first upload under this account')
    args = parser.parse_args()
    layout = build()
    meta = json.loads((KIT / 'dataset-metadata.json').read_text())
    meta['id'] = f'{args.account}/rsna-knee-trainer-kit'
    (KIT / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2))
    print('kit files', len(layout), '->', KIT, '| dataset', meta['id'])
    if args.push:
        os.environ['KAGGLE_API_TOKEN'] = os.environ[KEYS[args.account]]
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        if args.create:
            print(api.dataset_create_new(str(KIT.resolve()), public=False, quiet=True))
        else:
            print(api.dataset_create_version(str(KIT.resolve()), version_notes=args.push, quiet=True))


if __name__ == '__main__':
    main()
