"""Submit an exact, successfully executed private notebook version to the competition."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

slug, version, message = sys.argv[1], int(sys.argv[2]), sys.argv[3]
api = KaggleApi(); api.authenticate()
status = api.kernels_status(f'yodharinmonplub/{slug}')
assert 'COMPLETE' in str(status.status), status
result = api.competition_submit_code('submission.csv', message, competition='rsna-knee-abnormality-detection',
                                     kernel=f'yodharinmonplub/{slug}', kernel_version=version, quiet=False)
print(result)
