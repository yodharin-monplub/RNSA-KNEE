"""Build and verify targets_g38f_low_v2.csv (official-criteria relabel) and score it against the 58 gold exams."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
U = 'StudyInstanceUID'
train = pd.read_csv('data/train.csv')
v1 = pd.read_csv('data/llm_relabel/targets_g38f_low.csv').set_index(U)
v2 = pd.read_csv('data/llm_relabel/all_g38f_low_v2.csv').drop_duplicates(U).set_index(U)
out = v2.reindex(v1.index)[L]
filled = int(out.isna().any(axis=1).sum())
out = out.fillna(v1[L])
values = out.to_numpy(float)
assert list(out.index) == list(v1.index) and out.index.is_unique and len(out) == 4407 == len(train)
assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
out.reset_index()[[U] + L].to_csv('data/llm_relabel/targets_g38f_low_v2.csv', index=False)
gold = train.set_index(U).dropna(subset=L)
rows = []
for c in L:
    rows.append((c, roc_auc_score(gold[c], v1.loc[gold.index, c]), roc_auc_score(gold[c], out.loc[gold.index, c]),
                 float(v1[c].mean()), float(out[c].mean())))
table = pd.DataFrame(rows, columns=['finding', 'v1_gold_auc', 'v2_gold_auc', 'v1_mean_prob', 'v2_mean_prob']).set_index('finding')
table.loc['MACRO'] = table.mean()
print('v1-filled reports:', filled)
print(table.round(4).to_string())
print('correlation v1~v2 per finding:', {c: round(float(np.corrcoef(v1[c], out[c])[0, 1]), 3) for c in L})
