"""Fixed model12/model15 development comparison; no blend-weight search."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from compare_models import bootstrap_auc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/native384_gold_notebook/output'
labels = list(pd.read_csv(ROOT/'data/sample_submission.csv').columns[1:])
gold = pd.read_csv(ROOT/'data/train.csv').dropna(subset=labels).set_index('StudyInstanceUID')
assert len(gold) == 58 and gold.index.is_unique
predictions = {}
for name, path in [('model12', ROOT/'artifacts/coatnet_gold_notebook/output/model12_predictions.csv'),
                   ('model15', OUT/'gold_predictions.csv')]:
    f = pd.read_csv(path).set_index('StudyInstanceUID')
    assert f.index.is_unique and set(f.index) == set(gold.index)
    pred = f.loc[gold.index, labels].to_numpy()
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    predictions[name] = pred
predictions['equal_probability_blend'] = .5*predictions['model12'] + .5*predictions['model15']
y = gold[labels].to_numpy()
indices = np.random.default_rng(20260911).integers(0, len(y), (2000, len(y)))
draws = {name: bootstrap_auc(y, p, indices) for name, p in predictions.items()}
report = {'notes': '58 development cases used by publisher; not untouched holdout. '
                  'Equal probability blend fixed before observing these scores.', 'models': {}}
for name, p in predictions.items():
    result = {'macro_auc': float(roc_auc_score(y, p, average='macro')),
              'bootstrap_95pct': np.quantile(draws[name], [.025,.975]).tolist(),
              'paired_delta_vs_model12_95pct': np.quantile(draws[name]-draws['model12'], [.025,.975]).tolist(),
              'per_label_auc': dict(zip(labels, roc_auc_score(y,p,average=None).tolist()))}
    report['models'][name] = result
    print(name, result['macro_auc'], result['paired_delta_vs_model12_95pct'])
(OUT/'analysis.json').write_text(json.dumps(report,indent=2))
