"""Download a finished TPU training kernel's outputs (best.pt, history, metrics, gold predictions, OOF predictions) into
artifacts/remote_<name>/ - the layout scripts/upload_fleet_weights.py --add <name> expects - and print the metrics.

Usage: python scripts/collect_tpu_run.py model98 [--account yodharinmonplub|krittanutsomtuas]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_YODHAIRNMONPLUB', 'krittanutsomtuas': 'KAGGLE_API_KEY_WARIN', 'actuarialtutor': 'KAGGLE_API_KEY_KRITTANUT'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--account', default='yodharinmonplub', choices=sorted(KEYS))
    args = parser.parse_args()
    os.environ['KAGGLE_API_TOKEN'] = dotenv_values(ROOT / '.env')[KEYS[args.account]]
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    ref = f'{args.account}/rsna-knee-tpu-train-{args.name}'
    dest = ROOT / f'artifacts/remote_{args.name}'
    dest.mkdir(parents=True, exist_ok=True)
    pattern = r'(\.pt$|\.json$|\.csv$|\.log$|TRAINING_COMPLETE$|TRAINING_STOPPED_EARLY$)'
    token, files = None, []
    while True:
        result = api.kernels_output(ref, path=str(dest), file_pattern=pattern, force=True, quiet=True, page_token=token)
        files += list(result[0])
        token = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if not token:
            break
    run = dest / args.name
    print('downloaded', len(files), 'files ->', dest)
    for f in ('best.pt', 'metrics.json', 'history.json', 'TRAINING_COMPLETE'):
        print(f, 'present' if (run / f).exists() else 'MISSING')
    if (run / 'metrics.json').exists():
        print(json.dumps(json.loads((run / 'metrics.json').read_text()), indent=1))
    if (run / 'best.pt').exists():
        print('best.pt sha256', hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(), (run / 'best.pt').stat().st_size)


if __name__ == '__main__':
    main()
