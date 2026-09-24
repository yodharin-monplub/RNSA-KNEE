"""Descriptive 58-case comparison of fine-tuned CoAtNet checkpoints (not an untouched holdout)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID)
sources = {
    'model25': ROOT / 'artifacts/model25_gold_notebook/output/cached_gold_predictions.csv',
    'model16': ROOT / 'artifacts/remote_model16/model16/gold_predictions.csv',
    'model20': ROOT / 'artifacts/remote_model20/model20/gold_predictions.csv',
    'model22': ROOT / 'artifacts/remote_model22/model22/gold_predictions.csv',
    'model12_cached': ROOT / 'artifacts/dense_inference_validation_notebook/output/cached_gold_predictions.csv',
}
preds = {k: pd.read_csv(v).set_index(UID) for k, v in sources.items() if v.exists()}
ids = sorted(set.intersection(*(set(p.index) for p in preds.values())))
assert len(ids) == 58, len(ids)
y = gold.loc[ids, LABELS].to_numpy()
P = {k: p.loc[ids, LABELS].to_numpy() for k, p in preds.items()}
P['model16+model25'] = (P['model16'] + P['model25']) / 2


def macro(pred, rows=slice(None)):
    return float(np.mean([roc_auc_score(y[rows, j], pred[rows, j]) for j in range(len(LABELS))]))


rng = np.random.default_rng(0)
boots = []
while len(boots) < 2000:
    rows = rng.integers(0, len(ids), len(ids))
    if all(len(set(y[rows, j])) == 2 for j in range(len(LABELS))):
        boots.append(rows)
report = {'studies': len(ids), 'auc': {k: macro(v) for k, v in P.items()}}
for a, b in [('model25', 'model16'), ('model16+model25', 'model16')]:
    d = [macro(P[a], r) - macro(P[b], r) for r in boots]
    report[f'{a}_minus_{b}_95ci'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
out = ROOT / 'artifacts/model25_development/analysis.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
