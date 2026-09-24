"""Analyze the public checkpoints on gold development cases, with paired CIs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compare_models import bootstrap_auc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/coatnet_gold_notebook/output'
UID = 'StudyInstanceUID'
labels = list(pd.read_csv(ROOT / 'data/sample_submission.csv').columns[1:])
gold = pd.read_csv(ROOT / 'data/train.csv').dropna(subset=labels).set_index(UID)
assert len(gold) == 58 and gold.index.is_unique
y = gold[labels].to_numpy()
predictions = {}
for name in ['model8', 'model9', 'model10', 'model12', 'model11']:
    filename = 'model11_gold_blend.csv' if name == 'model11' else f'{name}_predictions.csv'
    frame = pd.read_csv(OUT / filename).set_index(UID)
    assert frame.index.is_unique and set(frame.index) == set(gold.index)
    pred = frame.loc[gold.index, labels].to_numpy()
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    predictions[name] = pred
assert np.allclose(predictions['model11'], .25 * predictions['model8'] +
                   .25 * predictions['model9'] + .5 * predictions['model10'], atol=1e-7)
# Fixed equal weight by training variant, specified before inspecting results.
predictions['three_family_blend'] = ((predictions['model8'] + predictions['model9']) / 2 +
                                    predictions['model10'] + predictions['model12']) / 3
indices = np.random.default_rng(20260911).integers(0, len(y), (2000, len(y)))
draws = {name: bootstrap_auc(y, pred, indices) for name, pred in predictions.items()}
report = {'notes': 'Development cases used by public checkpoint authors; not an untouched '
                  'validation set. Intervals describe sampling uncertainty, not development '
                  'bias. No per-label weight tuning. Public/private test scores are separate.',
          'studies': len(y), 'models': {}}
for name, pred in predictions.items():
    report['models'][name] = {
        'macro_auc': float(roc_auc_score(y, pred, average='macro')),
        'bootstrap_95pct': np.quantile(draws[name], [.025, .975]).tolist(),
        'difference_vs_model8': float(roc_auc_score(y, pred, average='macro') -
                                     roc_auc_score(y, predictions['model8'], average='macro')),
        'paired_difference_95pct': np.quantile(draws[name] - draws['model8'], [.025, .975]).tolist(),
        'per_label_auc': dict(zip(labels, roc_auc_score(y, pred, average=None).tolist()))}
(OUT / 'analysis.json').write_text(json.dumps(report, indent=2))
for name, metrics in report['models'].items():
    print(name, 'AUC', round(metrics['macro_auc'], 6),
          'paired delta CI', np.round(metrics['paired_difference_95pct'], 5).tolist())
