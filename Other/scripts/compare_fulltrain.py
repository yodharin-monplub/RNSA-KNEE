"""Full-training test: ResNet34 (model65) vs ResNet34 + external data (model66) vs the CoAtNet arm, and blends."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
U = 'StudyInstanceUID'
gold = pd.read_csv('data/train.csv').set_index(U).dropna(subset=L)
y = gold[L].to_numpy()
P = {m: pd.read_csv(f'artifacts/remote_{m}/{m}/gold_predictions.csv').set_index(U).loc[gold.index, L].to_numpy()
     for m in ('model38', 'model39', 'model40', 'model41', 'model65', 'model66')}


def macro(p, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    return float(np.mean([roc_auc_score(y[idx, j], p[idx, j]) for j in range(12) if len(set(y[idx, j])) == 2]))


def rank(p):
    return np.argsort(np.argsort(p, axis=0), axis=0) / (len(p) - 1)


table = pd.DataFrame({m: [roc_auc_score(y[:, j], P[m][:, j]) for j in range(12)] for m in ('model38', 'model65', 'model66')}, index=L)
table.loc['MACRO'] = table.mean()
print(table.round(4).to_string())
coat = .70 * P['model38'] + .15 * P['model39'] + .30 * P['model40'] + .15 * P['model41']
coat = coat / 1.30
print('\nCoAtNet 4-arm blend (model43 weights):', round(macro(coat), 4))
blends = {'coat': coat}
for w in (.1, .2, .3, .4):
    for name in ('model65', 'model66'):
        blends[f'coat + {w} {name} (rank)'] = (1 - w) * rank(coat) + w * rank(P[name])
    blends[f'coat + {w} mean(65,66) (rank)'] = (1 - w) * rank(coat) + w * rank((P['model65'] + P['model66']) / 2)
for k, v in blends.items():
    print(f'{k:36s} {macro(v):.4f}')
rng = np.random.default_rng(0)
pairs = {'model66 - model65': (P['model66'], P['model65']),
         'model65 - model38': (P['model65'], P['model38']),
         'coat+0.2 mean(65,66) - coat': (blends['coat + 0.2 mean(65,66) (rank)'], rank(coat))}
for name, (a, b) in pairs.items():
    d = []
    for _ in range(800):
        idx = rng.integers(0, len(y), len(y))
        d.append(macro(a, idx) - macro(b, idx))
    print(f'{name:32s} {macro(a) - macro(b):+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')
