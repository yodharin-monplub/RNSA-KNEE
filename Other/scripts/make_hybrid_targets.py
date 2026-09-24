"""Hybrid report labels: Gemini relabels, except findings where a published set is clearly better.

Rule fixed before writing the file: a finding switches source only when that source beats the Gemini
labels by at least 0.02 macro AUC on the 58 gold cases (a large margin for a 58-case estimate).
Tie-break (not gold-driven): when two published sets are within 0.005 of each other, take lee_v4,
because it scores reports that do not mention a finding 0.25 (a weak negative under the trainer's
confidence weighting 2|p-0.5|) where lee_v2 gives 0.50 (zero weight, so silent studies would not
constrain that finding at all).
Writes data/llm_relabel/targets_hybrid.csv and prints the per-finding choice.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
MARGIN = 0.02
gold = pd.read_csv('data/train.csv').set_index(UID).dropna(subset=L)


def read(path):
    return pd.read_csv(path).set_index(UID)[L].apply(pd.to_numeric, errors='coerce')


sources = {
    'gemini_low': read('data/llm_relabel/targets_g38f_low.csv'),
    'lee_v4': read('data/weak_labels_v4/llm_labels_v4_blend.csv'),
    'lee_v2': read('artifacts/sources/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v2.csv'),
}
ids = gold.index.tolist()
choice, out = {}, sources['gemini_low'].copy()
for label in L:
    scores = {name: roc_auc_score(gold.loc[ids, label], frame.loc[ids, label].fillna(0.5)) for name, frame in sources.items()}
    best = max(scores, key=scores.get)
    if best != 'lee_v4' and scores[best] - scores['lee_v4'] < 0.005:
        best = 'lee_v4'
    pick = best if scores[best] - scores['gemini_low'] >= MARGIN else 'gemini_low'
    choice[label] = (pick, {k: round(v, 3) for k, v in scores.items()})
    if pick != 'gemini_low':
        out[label] = sources[pick][label].reindex(out.index).fillna(sources['gemini_low'][label])
    print(f'{label:18s} -> {pick:10s} {choice[label][1]}')
values = out[L].to_numpy(float)
assert out.index.is_unique and len(out) == 4407 and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
hybrid_gold = out.loc[ids, L].to_numpy(float)
print('gold macro AUC: gemini', round(np.mean([roc_auc_score(gold.loc[ids, c], sources['gemini_low'].loc[ids, c]) for c in L]), 4),
      'hybrid', round(np.mean([roc_auc_score(gold.loc[ids, c], hybrid_gold[:, j]) for j, c in enumerate(L)]), 4))
out.reset_index()[[UID] + L].to_csv('data/llm_relabel/targets_hybrid.csv', index=False)
print('wrote data/llm_relabel/targets_hybrid.csv')
