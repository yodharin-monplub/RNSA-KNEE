"""Descriptive 58-case development predictions for the model25 checkpoint on the verified pilot cache.

Derived from scripts/validate_dense_inference.py: only the checkpoint source changes and the
raw-reference equality check is replaced by writing predictions (AUC is computed locally).
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

source = (ROOT / 'scripts/validate_dense_inference.py').read_text()
changes = {
    "files=[inputs/'raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',inputs/'datasets/dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt']":
        "files=[inputs/'rsna-knee-model25-weights/model25_best.pt',inputs/'datasets/yodharinmonplub/rsna-knee-model25-weights/model25_best.pt']",
    "    diff=float(np.max(np.abs(frame.to_numpy()-reference[LABELS].to_numpy())))\n    assert diff<.001,diff\n":
        "    diff=float(np.max(np.abs(frame.to_numpy()-reference[LABELS].to_numpy())))\n",
    "'tolerance':.001,": "'note':'model25 descriptive; diff is vs model12 raw reference, not a gate',",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
compile(source, 'model25_gold', 'exec')

api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/model25_gold_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(source)
meta = {'id': 'yodharinmonplub/rsna-knee-model25-gold-predictions', 'title': 'RSNA Knee Model25 Gold Predictions',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['yodharinmonplub/rsna-knee-model25-weights'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ['yodharinmonplub/rsna-knee-dense-cache-pilot'], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
