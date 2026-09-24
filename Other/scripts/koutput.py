"""Download a kernel's output files (and log) into artifacts/<name>_notebook/output."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
slug, name = sys.argv[1], sys.argv[2]
out = ROOT / f'artifacts/{name}_notebook/output'
out.mkdir(parents=True, exist_ok=True)
pattern = sys.argv[3] if len(sys.argv) > 3 else None
api.kernels_output(f'yodharinmonplub/{slug}', path=str(out), force=True, file_pattern=pattern)
for p in sorted(out.rglob('*')):
    if p.is_file():
        print(p.relative_to(out), p.stat().st_size)
