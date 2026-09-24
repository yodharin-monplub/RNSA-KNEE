"""Compare fixed CoAtNet/DINOv3 candidates on gold development cases."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from compare_models import bootstrap_auc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/dinov3_gold_notebook/output'
UID = 'StudyInstanceUID'
labels = list(pd.read_csv(ROOT / 'data/sample_submission.csv').columns[1:])
gold = pd.read_csv(ROOT / 'data/train.csv').dropna(subset=labels).set_index(UID)
assert len(gold) == 58 and gold.index.is_unique
y = gold[labels].to_numpy()


def read(path):
    frame = pd.read_csv(path).set_index(UID)
    assert frame.index.is_unique and set(frame.index) == set(gold.index)
    pred = frame.loc[gold.index, labels].to_numpy()
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
    return pred


dino = read(OUT / 'gold_predictions.csv')
folds = [read(OUT / f'fold{i}_probabilities.csv') for i in range(5)]
fold_rank_mean = np.mean([(rankdata(f, axis=0)-1)/(len(y)-1) for f in folds], axis=0)
assert np.allclose(dino, fold_rank_mean, atol=1e-7)
coat = read(ROOT / 'artifacts/coatnet_gold_notebook/output/model12_predictions.csv')
coat_rank = (rankdata(coat, axis=0)-1)/(len(y)-1)
dino_rank = (rankdata(dino, axis=0)-1)/(len(y)-1)
blend = .5*coat_rank + .5*dino_rank
predictions = {'model12': coat, 'model13': dino, 'equal_rank_blend': blend}
indices = np.random.default_rng(20260911).integers(0, len(y), (2000, len(y)))
draws = {name: bootstrap_auc(y, p, indices) for name, p in predictions.items()}
report = {'note': 'Development analysis only. Public checkpoint training overlap is '
                  'not independently verified. Blend fixed before observing these scores; '
                  'no weight search. Gold cases are not an untouched holdout.', 'models': {}}
for name, pred in predictions.items():
    report['models'][name] = {
        'macro_auc': float(roc_auc_score(y, pred, average='macro')),
        'bootstrap_95pct': np.quantile(draws[name], [.025, .975]).tolist(),
        'paired_delta_vs_model12_95pct': np.quantile(draws[name]-draws['model12'], [.025, .975]).tolist(),
        'per_label_auc': dict(zip(labels, roc_auc_score(y, pred, average=None).tolist()))}
report['rank_correlation_per_label'] = {
    label: float(np.corrcoef(coat_rank[:, j], dino_rank[:, j])[0, 1])
    for j, label in enumerate(labels)}
report['fold_macro_auc'] = [float(roc_auc_score(y, p, average='macro')) for p in folds]
(OUT / 'cross_family_analysis.json').write_text(json.dumps(report, indent=2))
for name, result in report['models'].items():
    print(name, 'AUC', round(result['macro_auc'], 6),
          'paired delta CI', np.round(result['paired_delta_vs_model12_95pct'], 5).tolist())
