"""Out-of-fold ResNet34 predictions (model68 fold A, model69 fold B) -> OOF file, pseudo-label targets, gold checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
U = 'StudyInstanceUID'
train = pd.read_csv('data/train.csv').set_index(U)
gold = train.dropna(subset=L)
v1 = pd.read_csv('data/llm_relabel/targets_g38f_low.csv').set_index(U)


def read(run, name):
    return pd.read_csv(f'artifacts/remote_{run}/{run}/{name}').set_index(U)[L]


oof = pd.concat([read('model68', 'oof_predictions.csv'), read('model69', 'oof_predictions.csv')])
non_gold = [u for u in v1.index if u not in set(gold.index)]
# 4 studies are excluded from every partition (duplicate reports shared with gold); they keep their v1 labels.
missing = [u for u in non_gold if u not in set(oof.index)]
assert oof.index.is_unique and set(oof.index) <= set(non_gold) and len(missing) == 4, (len(oof), len(missing))
oof = pd.concat([oof, v1.loc[missing, L]])
gold_mean = (read('model68', 'gold_predictions.csv').loc[gold.index] + read('model69', 'gold_predictions.csv').loc[gold.index]) / 2
full = pd.concat([oof, gold_mean]).loc[v1.index]
full.reset_index()[[U] + L].to_csv('data/llm_relabel/oof_resnet34.csv', index=False)

targets = v1[L].copy()
targets.loc[non_gold] = 0.5 * v1.loc[non_gold, L] + 0.5 * oof.loc[non_gold, L]
values = targets.to_numpy(float)
assert list(targets.index) == list(v1.index) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
targets.reset_index()[[U] + L].to_csv('data/llm_relabel/targets_v1_oof50.csv', index=False)

print('OOF vs Gemini v1 labels on the 4,349 non-gold studies (AUC of OOF against labels thresholded at 0.5; Pearson r):')
for c in L:
    hard = (v1.loc[non_gold, c] > .5).astype(int)
    print(f'  {c:18s} auc {roc_auc_score(hard, oof.loc[non_gold, c]):.3f}  r {np.corrcoef(v1.loc[non_gold, c], oof.loc[non_gold, c])[0, 1]:.3f}  mean label {v1.loc[non_gold, c].mean():.3f}  mean oof {oof.loc[non_gold, c].mean():.3f}')

y = gold[L].to_numpy()
P = {m: pd.read_csv(f'artifacts/remote_{m}/{m}/gold_predictions.csv').set_index(U).loc[gold.index, L].to_numpy()
     for m in ('model38', 'model39', 'model40', 'model41', 'model65', 'model68', 'model69')}


def macro(p, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    return float(np.mean([roc_auc_score(y[idx, j], p[idx, j]) for j in range(12) if len(set(y[idx, j])) == 2]))


def rank(p):
    return np.argsort(np.argsort(p, axis=0), axis=0) / (len(p) - 1)


coat = (.70 * P['model38'] + .15 * P['model39'] + .30 * P['model40'] + .15 * P['model41']) / 1.30
fold_mean = (P['model68'] + P['model69']) / 2
res3 = (P['model65'] + P['model68'] + P['model69']) / 3
label_gold = v1.loc[gold.index, L].to_numpy()
print('\nGold macro AUC: model68', round(macro(P['model68']), 4), '| model69', round(macro(P['model69']), 4),
      '| mean(68,69)', round(macro(fold_mean), 4), '| model65', round(macro(P['model65']), 4), '| mean(65,68,69)', round(macro(res3), 4),
      '| CoAtNet blend', round(macro(coat), 4))
print('Pseudo-label sanity on gold (NOT used for training): labels', round(macro(label_gold), 4),
      '| 0.5 labels + 0.5 mean(68,69)', round(macro(.5 * label_gold + .5 * fold_mean), 4),
      '| 0.5 labels + 0.5 CoAtNet', round(macro(.5 * label_gold + .5 * coat), 4))
rng = np.random.default_rng(0)
for w in (.2, .3, .4):
    blend = (1 - w) * rank(coat) + w * rank(res3)
    d = [macro(blend, i) - macro(rank(coat), i) for i in (rng.integers(0, len(y), len(y)) for _ in range(600))]
    print(f'coat + {w} mean(65,68,69) rank: {macro(blend):.4f}  diff {macro(blend) - macro(coat):+.4f}  CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')
