"""Upload the model80 fleet (this project's fully trained models) as one private Kaggle dataset.

Files: model72_best.pt (EfficientNet-B0 288 px), model71_fold{0..4}_best.pt (ResNet34 256 px, 5 report-group folds),
model73_best.pt (CoAtNet-2 256 px). Hard links, so the dataset folder costs no extra disk.
Usage: python scripts/upload_fleet_weights.py [--account yodharinmonplub|krittanutsomtuas] [--dry] [--add RUN ...] [--add-model75]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_YODHAIRNMONPLUB', 'krittanutsomtuas': 'KAGGLE_API_KEY_KRITTANUTSOMTUNG'}
FILES = {'model72_best.pt': 'artifacts/remote_model72/model72/best.pt',
         'model73_best.pt': 'artifacts/remote_model73/model73/best.pt',
         **{f'model71_fold{k}_best.pt': f'artifacts/remote_model71/model71/fold{k}/best.pt' for k in range(5)}}

parser = argparse.ArgumentParser()
parser.add_argument('--account', default='yodharinmonplub', choices=sorted(KEYS))
parser.add_argument('--dry', action='store_true')
parser.add_argument('--add-model75', action='store_true', help='alias of --add model75')
parser.add_argument('--add', action='append', default=[],
                    help='finished run (repeatable): adds artifacts/remote_RUN/RUN/best.pt as RUN_best.pt and publishes a new '
                         'version; members already in the dataset folder are kept')
args = parser.parse_args()
if args.add_model75:
    args.add.append('model75')
output = ROOT / f'artifacts/fleet_dataset_{args.account}'
kept = [f.name[:-len('_best.pt')] for f in output.glob('model*_best.pt') if f.name not in FILES] if output.exists() else []
for run in sorted(set(args.add) | set(kept)):
    assert (ROOT / f'artifacts/remote_{run}/{run}/TRAINING_COMPLETE').is_file(), f'{run} has not finished'
    FILES[f'{run}_best.pt'] = f'artifacts/remote_{run}/{run}/best.pt'
output.mkdir(exist_ok=True)
digests = {}
for name, rel in FILES.items():
    target = output / name
    if not target.exists():
        os.link(ROOT / rel, target)
    digests[name] = hashlib.sha256(target.read_bytes()).hexdigest()
(output / 'readme.txt').write_text(
    'RSNA Knee fleet weights (model80 fleet leg). Private; trained by this project from ImageNet backbones on\n'
    'report-derived labels (Gemini v1). Gold AUC (58 cases): model72 EfficientNet-B0 288 px 0.9143; model71 ResNet34\n'
    '256 px 5-fold ensemble 0.9138; model73 CoAtNet-2 256 px 0.9122; fleet mean 0.9258.\n'
    'Checkpoints: {"arch", "res", "lab", "epoch", "weak_val_bce", "training_config", "model"}; head = Raptor MIL head.\n'
    + ''.join(f'{name} sha256 {digest}\n' for name, digest in digests.items()))
(output / 'dataset-metadata.json').write_text(json.dumps({
    'title': 'RSNA Knee Fleet Weights', 'id': f'{args.account}/rsna-knee-fleet-weights',
    'licenses': [{'name': 'other'}],
    'description': 'Private: this project\'s fully trained knee MRI models (model72, model71 folds, model73) for the '
                   'model80 fleet leg. No MRI pixels, reports or patient identifiers.'}, indent=2))
print(json.dumps(digests, indent=1))
if not args.dry:
    os.environ['KAGGLE_API_TOKEN'] = dotenv_values(ROOT / '.env')[KEYS[args.account]]
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    if args.add:
        print(api.dataset_create_version(str(output), 'add ' + ', '.join(f'{r}_best.pt' for r in args.add),
                                         convert_to_csv=False, quiet=True))
    else:
        print(api.dataset_create_new(str(output), public=False, convert_to_csv=False, quiet=True))
