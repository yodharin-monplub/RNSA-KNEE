"""Label ensemble: mean of the v1 (own definitions) and v2 (official criteria) Gemini relabel probabilities.

Motivation: v2 agrees better with gold as a label (0.9099 vs 0.9035) but trains worse heads (blend 0.9262 vs
0.9326), mostly through MCL and Baker's, where it turns low-grade / small findings into confident negatives.
Averaging keeps those cases in the middle instead. Prints label-vs-gold agreement for v1, v2 and the mean.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
folder = ROOT / 'data/llm_relabel'
v1 = pd.read_csv(folder / 'targets_g38f_low.csv').set_index(UID)
v2 = pd.read_csv(folder / 'targets_g38f_low_v2.csv').set_index(UID).loc[v1.index]
mean = (v1[L] + v2[L]) / 2
gold = pd.read_csv(ROOT / 'data/train.csv').set_index(UID).dropna(subset=L)
print(f'{"finding":18s} {"v1":>7s} {"v2":>7s} {"mean":>7s}')
rows = []
for c in L:
    scores = [roc_auc_score(gold[c], frame.loc[gold.index, c]) for frame in (v1, v2, mean)]
    rows.append(scores)
    print(f'{c:18s} ' + ' '.join(f'{s:7.4f}' for s in scores))
print(f'{"macro":18s} ' + ' '.join(f'{s:7.4f}' for s in np.mean(rows, axis=0)))
out = mean.reset_index()[[UID] + L]
assert len(out) == 4407 and np.isfinite(out[L].to_numpy()).all()
out.to_csv(folder / 'targets_g38f_low_v1v2mean.csv', index=False)
print('wrote', folder / 'targets_g38f_low_v1v2mean.csv')
