"""External-data step 2: model38 (baseline) vs model63 (+augmentation) vs model64 (+augmentation +external aux heads)."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
U = 'StudyInstanceUID'
gold = pd.read_csv('data/train.csv').set_index(U).dropna(subset=L)
preds = {m: pd.read_csv(f'artifacts/remote_{m}/{m}/gold_predictions.csv').set_index(U).loc[gold.index, L]
         for m in ('model38', 'model63', 'model64')}
y = gold[L].to_numpy()


def macro(p, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    values = []
    for j in range(12):
        if len(set(y[idx, j])) == 2:
            values.append(roc_auc_score(y[idx, j], p[idx, j]))
    return float(np.mean(values))


table = pd.DataFrame({m: [roc_auc_score(y[:, j], p.to_numpy()[:, j]) for j in range(12)] for m, p in preds.items()}, index=L)
table.loc['MACRO'] = table.mean()
print(table.round(4).to_string())
rng = np.random.default_rng(0)
result = {'macro': {m: macro(p.to_numpy()) for m, p in preds.items()}}
for a, b in (('model63', 'model38'), ('model64', 'model63'), ('model64', 'model38')):
    diffs = []
    for _ in range(4000):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(macro(preds[a].to_numpy(), idx) - macro(preds[b].to_numpy(), idx))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    result[f'{a}-{b}'] = {'diff': result['macro'][a] - result['macro'][b], 'ci95': [float(lo), float(hi)]}
    print(f'{a} - {b}: {result[f"{a}-{b}"]["diff"]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')
blend = (preds['model38'].to_numpy() + preds['model64'].to_numpy()) / 2
print('mean(model38, model64) macro', round(macro(blend), 4))
json.dump(result, open('artifacts/step2_comparison.json', 'w'), indent=2)
