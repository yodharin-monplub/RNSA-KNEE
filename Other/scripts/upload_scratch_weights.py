"""Upload a from-ImageNet run's best.pt as a private Kaggle dataset (package_run.py assumes a fine-tune baseline).

Usage: python scripts/upload_scratch_weights.py model65
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

run = sys.argv[1]
source = ROOT / f'artifacts/remote_{run}/{run}'
assert (source / 'TRAINING_COMPLETE').is_file()
metrics = json.loads((source / 'metrics.json').read_text())
output = ROOT / f'artifacts/{run}_dataset'
output.mkdir(exist_ok=True)
shutil.copy2(source / 'best.pt', output / f'{run}_best.pt')
shutil.copy2(ROOT / f'{run}/readme.txt', output / 'readme.txt')
digest = hashlib.sha256((output / f'{run}_best.pt').read_bytes()).hexdigest()
(output / 'dataset-metadata.json').write_text(json.dumps({
    'title': f'RSNA Knee {run.capitalize()} Weights', 'id': f'yodharinmonplub/rsna-knee-{run}-weights',
    'licenses': [{'name': 'other'}],
    'description': (f'Private {run}: timm ImageNet backbone trained fully by this project on report-derived labels '
                    f'(gold AUC {metrics["gold_auc"]:.4f}, epoch {metrics["selected_epoch"]}). No MRI pixels, reports or '
                    f'patient identifiers. sha256 {digest}')}, indent=2))
api = KaggleApi(); api.authenticate()
print(api.dataset_create_new(str(output), public=False, convert_to_csv=False, quiet=True))
print(digest)
