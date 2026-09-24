"""Publish a timm ImageNet backbone safetensors file as a private Kaggle dataset (for offline rentals).

Usage: python scripts/upload_backbone.py <folder> <slug-suffix> <timm-name>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

folder, suffix, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = folder / f'{name}.safetensors'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(folder / 'dataset-metadata.json').write_text(json.dumps({
    'title': f'RSNA Knee Backbone {suffix}', 'id': f'yodharinmonplub/rsna-knee-backbone-{suffix}',
    'licenses': [{'name': 'other'}],
    'description': (f'Unmodified timm ImageNet pretrained weights {name} from huggingface.co/timm '
                    f'(original model licence applies). sha256 {digest}. Used to initialise training on '
                    'offline GPU rentals.')}, indent=2))
(folder / 'SHA256.txt').write_text(f'{digest}  {path.name}\n')
api = KaggleApi(); api.authenticate()
print(api.dataset_create_new(str(folder), public=False, convert_to_csv=False, quiet=True))
print(digest)
