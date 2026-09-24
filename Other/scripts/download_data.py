"""Download competition metadata first; request individual images with --file."""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
if os.getenv('KAGGLE_API_KEY') and not os.getenv('KAGGLE_API_TOKEN'):
    os.environ['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_API_KEY']
os.environ.setdefault('KAGGLEHUB_CACHE', str(ROOT / '.cache/kagglehub'))

import kagglehub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', action='append', help='Exact competition file path; repeatable')
    args = parser.parse_args()
    files = args.file or ['train.csv', 'train_series.csv', 'test.csv',
                          'test_series.csv', 'sample_submission.csv']
    for name in files:
        path = kagglehub.competition_download(
            'rsna-knee-abnormality-detection', path=name,
            output_dir=str(ROOT / 'data'))
        print(f'Downloaded {name}: {path}')


if __name__ == '__main__':
    main()
