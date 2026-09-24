"""Score every report-label source (and simple ensembles) against the 58 gold studies."""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
     'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'
gold = pd.read_csv('data/train.csv').set_index(UID).dropna(subset=L)


def read(path):
    frame = pd.read_csv(path).set_index(UID)
    return frame[L].apply(pd.to_numeric, errors='coerce')


sources = {
    'gemini_low': read('data/llm_relabel/targets_g38f_low.csv'),
    'lee_v4': read('data/weak_labels_v4/llm_labels_v4_blend.csv'),
    'lee_v2': read('artifacts/sources/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v2.csv'),
    'lee_v1': read('data/weak_labels/llm_labels_full.csv'),
    'gpt56sol': read('artifacts/sources/lixin73__rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv'),
    'pilkwang_v2': read('artifacts/sources/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv'),
    'raptor': read('data/raptor_labels/labels_llm_soft.csv'),
}
for name, frame in list(sources.items()):
    covered = sum(u in frame.index for u in gold.index)
    print(f'{name:12s} covers {covered} of 58 gold studies')
    if covered < 58:
        del sources[name]
ids = [u for u in gold.index if all(u in s.index for s in sources.values())]
y = gold.loc[ids, L].to_numpy()
print('gold studies covered by every remaining source:', len(ids))


def macro(frame):
    values = frame.loc[ids, L].fillna(0.5).to_numpy(float)
    return float(np.mean([roc_auc_score(y[:, j], values[:, j]) for j in range(12)]))


def per_label(frame):
    values = frame.loc[ids, L].fillna(0.5).to_numpy(float)
    return [round(roc_auc_score(y[:, j], values[:, j]), 3) for j in range(12)]


for name, frame in sources.items():
    coverage = frame.index.size
    print(f'{name:12s} n={coverage:5d} macro={macro(frame):.4f}', per_label(frame))


def rank_frame(frame):
    values = frame.loc[ids, L].fillna(0.5).to_numpy(float)
    return pd.DataFrame(np.column_stack([rankdata(values[:, j]) / len(ids) for j in range(12)]), index=ids, columns=L)


print('\nprobability averages:')
for k in (2, 3):
    for combo in itertools.combinations([n for n in ['gemini_low', 'lee_v4', 'gpt56sol', 'pilkwang_v2', 'lee_v2'] if n in sources], k):
        mean = sum(sources[c].loc[ids, L].fillna(0.5) for c in combo) / k
        print(f'  {"+".join(combo):40s} {macro(mean):.4f}')
print('\nrank averages:')
for k in (2, 3):
    for combo in itertools.combinations([n for n in ['gemini_low', 'lee_v4', 'gpt56sol', 'pilkwang_v2'] if n in sources], k):
        mean = sum(rank_frame(sources[c]) for c in combo) / k
        print(f'  {"+".join(combo):40s} {macro(mean):.4f}')
