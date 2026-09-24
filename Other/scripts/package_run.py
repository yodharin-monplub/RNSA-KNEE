"""Stage and optionally upload a completed fine-tuned checkpoint as a private Kaggle dataset.

Usage: python scripts/package_run.py model22 --source-sha <initial checkpoint sha> [--upload]
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    'model22': 'fine-spacing (v9) CoAtNet',
    'model25': 'max-span (v5 SWA) CoAtNet, last two stages',
    'model27': 'max-span (v5 SWA) CoAtNet, full-data refit',
    'model28': 'native384 (v8 SWA) CoAtNet',
    'model31': 'native384 (v8 SWA) CoAtNet, full-data refit',
    'model34': 'max-span (v5 SWA) CoAtNet, full-data refit seed 2',
    'model32': 'native384dense (v10) CoAtNet, full-data refit',
    'model36': 'ConvNeXt-small trained from ImageNet (new family)',
    'model38': 'max-span CoAtNet, Gemini relabels',
    'model39': 'native384dense CoAtNet, Gemini relabels',
    'model40': 'native384-v8 CoAtNet, Gemini relabels',
    'model41': 'fine-spacing CoAtNet, Gemini relabels',
    'model42': 'CoAtNet-2 trained from ImageNet-12k with Gemini relabels',
    'model51': 'max-span CoAtNet, Gemini relabels, 24 windows per study',
    'model54': 'native384dense CoAtNet, Gemini relabels, 24 windows per study',
    'model55': 'native384-v8 CoAtNet, Gemini relabels, 24 windows per study',
    'model56': 'fine-spacing CoAtNet, Gemini relabels, 24 windows per study',
    'model60': 'max-span CoAtNet, hybrid labels (Gemini + Lee v4 Effusion/Baker\'s), model38 recipe',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--upload', action='store_true')
    args = p.parse_args()
    run = args.run
    source = ROOT / f'artifacts/remote_{run}/{run}'
    if not (source / 'TRAINING_COMPLETE').is_file():
        raise RuntimeError('Full training and artifact backup must complete before packaging')
    metrics = json.loads((source / 'metrics.json').read_text())
    history = json.loads((source / 'history.json').read_text())
    assert metrics['selected_epoch'] > 0, 'Epoch zero retained: fine-tuning did not help'
    assert metrics['weak_val_bce'] < history[0]['weak_val_bce']
    checkpoint = source / 'best.pt'
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    output = ROOT / f'artifacts/{run}_dataset'
    output.mkdir(exist_ok=True)
    (output / 'PACKAGE_READY.json').unlink(missing_ok=True)
    shutil.copy2(checkpoint, output / f'{run}_best.pt')
    for path in [ROOT / 'SOURCES.md', ROOT / f'{run}/readme.txt']:
        shutil.copy2(path, output / path.name)
    metadata = {
        'title': f'RSNA Knee {run.capitalize()} Weights',
        'id': f'yodharinmonplub/rsna-knee-{run}-weights',
        'licenses': [{'name': 'other'}],
        'description': (f'Private weak-label fine-tuned {DESCRIPTIONS[run]} checkpoint. Initial model and '
                        'geometry by Johnathan Wagner / Dread Development. No MRI pixels, reports, or '
                        'patient identifiers included. See SOURCES.md.'),
    }
    (output / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'PACKAGE_READY.json').write_text(json.dumps({'sha256': digest, 'metrics': metrics}, indent=2))
    print('package ready', digest)
    if args.upload:
        load_dotenv(ROOT / '.env')
        os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
        os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        staged = output / 'upload'
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(exist_ok=True)
        for name in [f'{run}_best.pt', 'SOURCES.md', 'readme.txt', 'dataset-metadata.json']:
            shutil.copy2(output / name, staged / name)
        print(api.dataset_create_new(str(staged), public=False, convert_to_csv=False, quiet=False))


if __name__ == '__main__':
    main()
