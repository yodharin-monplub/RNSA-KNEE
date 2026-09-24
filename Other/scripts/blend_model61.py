"""Gold CV of the CoAtNet side with the retrained heads (model61) against model43's arms.

CoAtNet side = weighted probability mean (weights normalised) of the arm views, as in the notebook:
maxspan 0.60+0.10, native384dense 0.10+0.05, native384-v8 0.20+0.10, fine-spacing 0.15. Forward-view gold
predictions stand in for both views. Retrained-head predictions come from rsna-knee-head-retrain-<arm>
(final head, seed 7); arms whose kernel has not finished fall back to the original arm.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID).dropna(subset=L)
y = gold[L].to_numpy(float)
W = {'maxspan': .70, 'dense': .15, 'v8': .30, 'fine': .15}
ORIGINAL = {'maxspan': 'model38', 'dense': 'model39', 'v8': 'model40', 'fine': 'model41'}


def auc(truth, score):
    positives = truth > .5
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return (rankdata(score)[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro(p, rows=slice(None)):
    return float(np.mean([auc(y[rows, j], p[rows, j]) for j in range(12)]))


def read(path):
    return pd.read_csv(path).set_index(UID).loc[gold.index, L].to_numpy(float)


old = {a: read(ROOT / f'artifacts/remote_{m}/{m}/gold_predictions.csv') for a, m in ORIGINAL.items()}
new = {}
for a in W:
    path = ROOT / f'artifacts/head_retrain_{a}_notebook/output/gold_predictions_{a}.csv'
    new[a] = read(path) if path.exists() else None
blend = lambda parts: sum(W[a] * parts[a] for a in W) / sum(W.values())
print(f'{"arm":9s} {"original":>9s} {"retrained":>10s}')
for a in W:
    print(f'{a:9s} {macro(old[a]):9.4f} ' + (f'{macro(new[a]):10.4f}' if new[a] is not None else '   pending'))
base = blend(old)
candidate = blend({a: (new[a] if new[a] is not None else old[a]) for a in W})
rng = np.random.default_rng(20260919)
diffs = []
while len(diffs) < 4000:
    rows = rng.integers(0, len(y), len(y))
    if all(0 < y[rows, j].sum() < len(rows) for j in range(12)):
        diffs.append(macro(candidate, rows) - macro(base, rows))
print(f'blend: model43 arms {macro(base):.4f} -> retrained heads {macro(candidate):.4f} '
      f'(diff {macro(candidate) - macro(base):+.4f}, paired bootstrap 95% CI [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}], '
      f'P(diff>0) {np.mean(np.asarray(diffs) > 0):.3f})')
print('per finding (blend):')
for j, c in enumerate(L):
    print(f'  {c:18s} {auc(y[:, j], base[:, j]):.4f} -> {auc(y[:, j], candidate[:, j]):.4f}')
