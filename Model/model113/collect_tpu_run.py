"""Download a finished TPU training kernel's outputs (best.pt, history, metrics, gold predictions, OOF predictions) into
artifacts/remote_<name>/ - the layout scripts/upload_fleet_weights.py --add <name> expects - and print the metrics.

Usage: python scripts/collect_tpu_run.py model98 [--account yodharinmonplub|krittanutsomtuas|actuarialtutor]
       python scripts/collect_tpu_run.py model110-f012 --account krittanutsomtuas   (k-fold / --then sessions)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
KEYS = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', 'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org', 'actuarialtutor': 'KAGGLE_API_KEY_krittanutsomtung@actuarialtutor.org', 'aikayui': 'KAGGLE_API_KEY_aikayui@actuarialtutor.org'}


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
    pattern = r'(\.pt$|\.json$|\.csv$|\.log$|TRAINING_COMPLETE$|TRAINING_STOPPED_EARLY$|PLAN_STATUS.txt$)'
    token, files = None, []
    while True:
        result = api.kernels_output(ref, path=str(dest), file_pattern=pattern, force=True, quiet=True, page_token=token)
        files += list(result[0])
        token = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        if not token:
            break
    print('downloaded', len(files), 'files ->', dest)
    if (dest / 'PLAN_STATUS.txt').exists():
        print('PLAN_STATUS', (dest / 'PLAN_STATUS.txt').read_text().strip())
    # every run folder in the output (a plain run, fold{k} folders of a k-fold session, a --then model)
    runs = sorted({f.parent for pattern in ('best.pt', 'metrics.json', 'history.json') for f in dest.rglob(pattern)
                   if not f.parent.name.endswith('_smoke')})
    for run in runs:
        print('==', run.relative_to(dest).as_posix(), '|',
              ', '.join(f'{f} {"ok" if (run / f).exists() else "MISSING"}' for f in ('best.pt', 'metrics.json', 'TRAINING_COMPLETE')))
        if (run / 'metrics.json').exists():
            print(json.dumps(json.loads((run / 'metrics.json').read_text())))
        if (run / 'best.pt').exists():
            print('best.pt sha256', hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(), (run / 'best.pt').stat().st_size)


if __name__ == '__main__':
    main()
