"""Private Kaggle artifact upload, notebook execution, and code submission."""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY', ''))
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['upload', 'run', 'status', 'output', 'submit', 'submissions'])
    p.add_argument('--version', type=int)
    p.add_argument('--pattern')
    p.add_argument('--folder', default='artifacts/kaggle_notebook')
    p.add_argument('--message')
    args = p.parse_args()
    api = KaggleApi(); api.authenticate()
    folder = ROOT / args.folder
    meta = json.loads((folder / 'kernel-metadata.json').read_text())
    kernel = meta['id']
    if args.action == 'upload':
        if not (ROOT / 'artifacts/remote/ENSEMBLE_VALIDATED').exists():
            raise RuntimeError('Real DICOM inference has not validated')
        result = api.dataset_create_new(str(ROOT / 'artifacts/kaggle_dataset'),
                                        public=False, convert_to_csv=False, quiet=False)
    elif args.action == 'run':
        result = api.kernels_push(str(folder), acc='NvidiaTeslaT4')
    elif args.action == 'status':
        result = api.kernels_status(kernel)
    elif args.action == 'output':
        output_dir = ROOT / 'artifacts/kaggle_output' if args.folder == 'artifacts/kaggle_notebook' else folder / 'output'
        result = api.kernels_output(kernel, path=str(output_dir), force=True,
                                    file_pattern=args.pattern)
    elif args.action == 'submit':
        if args.version is None:
            raise ValueError('Use the exact successfully executed notebook version')
        result = api.competition_submit_code('submission.csv',
            args.message or (meta['title'] + ': offline MRI inference.'),
            competition='rsna-knee-abnormality-detection', kernel=kernel,
            kernel_version=args.version, quiet=False)
    else:
        result = api.competition_submissions('rsna-knee-abnormality-detection')
    print(str(result))
    (folder / f'{args.action}_response.txt').write_text(str(result))
    if args.folder == 'artifacts/kaggle_notebook':
        (ROOT / f'artifacts/kaggle_{args.action}_response.txt').write_text(str(result))


if __name__ == '__main__':
    main()
