"""Descriptive model20 comparisons using fixed simple probability blends."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_models import bootstrap_auc

ROOT = Path(__file__).resolve().parents[1]
UID = 'StudyInstanceUID'
labels = list(pd.read_csv(ROOT / 'data/sample_submission.csv').columns[1:])
gold = pd.read_csv(ROOT / 'data/train.csv').dropna(subset=labels).set_index(UID)
assert len(gold) == 58 and gold.index.is_unique


def load(relative):
    frame = pd.read_csv(ROOT / relative).set_index(UID)
    assert frame.index.is_unique and set(frame.index) == set(gold.index)
    values = frame.loc[gold.index, labels].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return values


predictions = {
    'model15': load('artifacts/native384_gold_notebook/output/gold_predictions.csv'),
    'model16': load('artifacts/remote_model16/model16/gold_predictions.csv'),
    'model18': load('artifacts/model18_development/predictions.csv'),
    'model20': load('artifacts/remote_model20/model20/gold_predictions.csv'),
}
predictions['model19'] = (predictions['model16'] + predictions['model18']) / 2
predictions['equal_16_20'] = (predictions['model16'] + predictions['model20']) / 2
predictions['equal_18_20'] = (predictions['model18'] + predictions['model20']) / 2
predictions['equal_19_20'] = (predictions['model19'] + predictions['model20']) / 2

y = gold[labels].to_numpy()
indices = np.random.default_rng(20260911).integers(0, len(y), (4000, len(y)))
draws = {name: bootstrap_auc(y, values, indices) for name, values in predictions.items()}
report = {
    'notes': ('58 upstream development cases, not an untouched holdout. The three equal blends '
              'were specified together before this script was executed; no continuous or per-label '
              'weight search. Model19 is reconstructed exactly from its model16/model18 components.'),
    'models': {},
}
for name, values in predictions.items():
    report['models'][name] = {
        'macro_auc': float(roc_auc_score(y, values)),
        'paired_delta_vs_model19_95pct': np.quantile(
            draws[name] - draws['model19'], [.025, .975]).tolist(),
        'per_label_auc': dict(zip(labels, roc_auc_score(y, values, average=None).tolist())),
    }
    print(name, report['models'][name]['macro_auc'],
          report['models'][name]['paired_delta_vs_model19_95pct'])

output = ROOT / 'artifacts/model20_development'
output.mkdir(exist_ok=True)
(output / 'analysis.json').write_text(json.dumps(report, indent=2))
