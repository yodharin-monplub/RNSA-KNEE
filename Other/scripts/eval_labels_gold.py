"""Score a label file against the 58 gold studies (macro ROC AUC), on the studies it covers."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
g = pd.read_csv('data/train.csv').set_index('StudyInstanceUID').dropna(subset=L)
x = pd.read_csv(sys.argv[1]).set_index('StudyInstanceUID')
ids = [u for u in g.index if u in x.index]
v4 = pd.read_csv('data/weak_labels_v4/llm_labels_v4_blend.csv').set_index('StudyInstanceUID').loc[ids, L]
x = x.loc[ids, L]
for name, p in [('new', x), ('v4', v4), ('avg', (x + v4) / 2)]:
    aucs = {c: roc_auc_score(g.loc[ids, c], p[c]) for c in L}
    print(name, len(ids), round(np.mean(list(aucs.values())), 4), {c: round(a, 3) for c, a in aucs.items()})
