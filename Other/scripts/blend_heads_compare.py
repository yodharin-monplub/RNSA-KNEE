"""Compare a tagged head-retrain run (e.g. v2 labels) with model61's heads: per arm and in the four-arm blend.

Usage: python scripts/blend_heads_compare.py v2
Reads artifacts/head_retrain_<arm>_notebook/output/gold_predictions_<arm>.csv (model61, v1 labels) and
artifacts/head_retrain_<TAG>_<arm>_notebook/output/gold_predictions_<arm>.csv. Blend = weighted probability mean
with the notebook's view weights (maxspan 0.70, dense 0.15, v8 0.30, fine 0.15).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def rankdata(a):  # average ranks, 1-based; numpy only (scipy is blocked by this PC's app-control policy)
    a = np.asarray(a, float)
    order = a.argsort(kind='mergesort')
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    values, start = a[order], 0
    for i in range(1, len(a) + 1):
        if i == len(a) or values[i] != values[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2
            start = i
    return ranks


ROOT = Path(__file__).resolve().parents[1]
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
tag = sys.argv[1]
gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID).dropna(subset=L)
y = gold[L].to_numpy(float)
W = {'maxspan': .70, 'dense': .15, 'v8': .30, 'fine': .15}


def auc(truth, score):
    positives = truth > .5
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return (rankdata(score)[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro(p, rows=slice(None)):
    return float(np.mean([auc(y[rows, j], p[rows, j]) for j in range(12)]))


def read(path):
    return pd.read_csv(path).set_index(UID).loc[gold.index, L].to_numpy(float)


base = {a: read(ROOT / f'artifacts/head_retrain_{a}_notebook/output/gold_predictions_{a}.csv') for a in W}
new = {a: read(ROOT / f'artifacts/head_retrain_{tag}_{a}_notebook/output/gold_predictions_{a}.csv') for a in W}
blend = lambda parts: sum(W[a] * parts[a] for a in W) / sum(W.values())
print(f'{"arm":9s} {"model61":>8s} {tag:>8s} {"diff":>8s}   seeds ({tag})')
for a in W:
    report = json.loads((ROOT / f'artifacts/head_retrain_{tag}_{a}_notebook/output/head_retrain_{a}.json').read_text())
    seeds = [round(report[f'seed{s}_gold_auc'], 4) for s in (0, 1, 2)]
    print(f'{a:9s} {macro(base[a]):8.4f} {macro(new[a]):8.4f} {macro(new[a]) - macro(base[a]):+8.4f}   {seeds}')
b, n = blend(base), blend(new)
rng = np.random.default_rng(20260920)
diffs = []
while len(diffs) < 4000:
    rows = rng.integers(0, len(y), len(y))
    if all(0 < y[rows, j].sum() < len(rows) for j in range(12)):
        diffs.append(macro(n, rows) - macro(b, rows))
print(f'blend: model61 {macro(b):.4f} -> {tag} {macro(n):.4f} (diff {macro(n) - macro(b):+.4f}, paired bootstrap 95% CI '
      f'[{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}], P(diff>0) {np.mean(np.asarray(diffs) > 0):.3f})')
half = blend({a: (base[a] + new[a]) / 2 for a in W})
print(f'blend of the mean of both heads per arm: {macro(half):.4f}')
print('per finding (blend):')
for j, c in enumerate(L):
    print(f'  {c:18s} {auc(y[:, j], b[:, j]):.4f} -> {auc(y[:, j], n[:, j]):.4f}')
