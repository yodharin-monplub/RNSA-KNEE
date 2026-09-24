"""Build self-distillation targets: equal average of the Gemini relabels and held-out teacher predictions.

Gold studies keep the plain relabel value (they are never trained on). The 50/50 weight was fixed
before measuring anything about the result.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
UID = 'StudyInstanceUID'

labels = pd.read_csv(ROOT / 'data/llm_relabel/targets_g38f_low.csv').set_index(UID)
oof = pd.concat([pd.read_csv(ROOT / f'artifacts/remote_{m}/{m}/oof_predictions.csv').set_index(UID)
                 for m in ('model44', 'model45')])
assert oof.index.is_unique
partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv').set_index(UID)
non_gold = partition.index[partition.split != 'gold']
missing = set(non_gold) - set(oof.index)
print('non-gold studies', len(non_gold), 'with teacher predictions', len(non_gold) - len(missing))
out = labels.copy()
covered = [uid for uid in non_gold if uid in oof.index]
out.loc[covered, LABELS] = (labels.loc[covered, LABELS].to_numpy() + oof.loc[covered, LABELS].to_numpy()) / 2
values = out[LABELS].to_numpy(float)
assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
out.reset_index()[[UID] + LABELS].to_csv(ROOT / 'data/llm_relabel/targets_distill.csv', index=False)
print('wrote targets_distill.csv', out.shape)