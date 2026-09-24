"""Per-finding gold AUC of two trained runs (artifacts/remote_<run>/<run>/gold_predictions.csv), with a paired bootstrap.

Usage: python scripts/compare_gold_predictions.py model38 model60
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
base_name, new_name = sys.argv[1], sys.argv[2]
gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID).dropna(subset=L)


def load(run):
    frame = pd.read_csv(ROOT / f'artifacts/remote_{run}/{run}/gold_predictions.csv').set_index(UID)
    return frame.loc[gold.index, L].to_numpy(float)


y = gold[L].to_numpy(float)
base, new = load(base_name), load(new_name)
macro = lambda p, rows=slice(None): float(np.mean([roc_auc_score(y[rows, j], p[rows, j]) for j in range(12)]))
print(f'{"finding":18s} {base_name:>9s} {new_name:>9s}   diff')
for j, label in enumerate(L):
    a, b = roc_auc_score(y[:, j], base[:, j]), roc_auc_score(y[:, j], new[:, j])
    print(f'{label:18s} {a:9.4f} {b:9.4f} {b - a:+7.4f}')
print(f'{"macro":18s} {macro(base):9.4f} {macro(new):9.4f} {macro(new) - macro(base):+7.4f}')
rng = np.random.default_rng(20260919)
diffs = []
while len(diffs) < 2000:
    rows = rng.integers(0, len(y), len(y))
    if all(0 < y[rows, j].sum() < len(rows) for j in range(12)):
        diffs.append(macro(new, rows) - macro(base, rows))
print('paired bootstrap 95% CI for the macro difference:', np.round(np.percentile(diffs, [2.5, 97.5]), 4))
