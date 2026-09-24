"""Does the retrained maxspan head (rsna-knee-head-gold-cv, arms A0/A) still help inside the CoAtNet blend?

The notebook's CoAtNet side is a weighted mean of arm probabilities (weights normalised), then a rank transform.
View weights as in model43: maxspan 0.60+0.10 reversed, native384dense 0.10+0.05, native384-v8 0.20+0.10,
fine-spacing 0.15. Only forward-view gold predictions exist locally, so each arm's views share them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID).dropna(subset=L)
y = gold[L].to_numpy(float)


def run(name):
    return pd.read_csv(ROOT / f'artifacts/remote_{name}/{name}/gold_predictions.csv').set_index(UID).loc[gold.index, L].to_numpy(float)


def oof(name):
    return pd.read_csv(ROOT / f'artifacts/head_gold_cv_notebook/output/oof_{name}.csv').set_index(UID).loc[gold.index, L].to_numpy(float)


def macro(p, rows=slice(None)):
    return float(np.mean([roc_auc_score(y[rows, j], p[rows, j]) for j in range(12)]))


W = {'maxspan': .70, 'dense': .15, 'v8': .30, 'fine': .15}
arms = {'maxspan': run('model38'), 'dense': run('model39'), 'v8': run('model40'), 'fine': run('model41')}


def blend(maxspan):
    parts = dict(arms, maxspan=maxspan)
    return sum(W[k] * parts[k] for k in W) / sum(W.values())


candidates = {'model38 head (base)': arms['maxspan'], 'A0 plain-BCE head': oof('A0_weak_plain_bce'),
              'A confidence head': oof('A_weak_confidence'), 'D conf + gold4': oof('D_conf_gold4')}
rng = np.random.default_rng(20260919)
samples = []
while len(samples) < 2000:
    rows = rng.integers(0, len(y), len(y))
    if all(0 < y[rows, j].sum() < len(rows) for j in range(12)):
        samples.append(rows)
base_single, base_blend = candidates['model38 head (base)'], blend(candidates['model38 head (base)'])
print(f'{"maxspan head":22s} {"single":>8s} {"single diff 95% CI":>22s} {"blend":>8s} {"blend diff 95% CI":>22s}')
for name, pred in candidates.items():
    b = blend(pred)
    single_ci = np.percentile([macro(pred, r) - macro(base_single, r) for r in samples], [2.5, 97.5])
    blend_ci = np.percentile([macro(b, r) - macro(base_blend, r) for r in samples], [2.5, 97.5])
    print(f'{name:22s} {macro(pred):8.4f} [{single_ci[0]:+.4f}, {single_ci[1]:+.4f}]   {macro(b):8.4f} [{blend_ci[0]:+.4f}, {blend_ci[1]:+.4f}]')
