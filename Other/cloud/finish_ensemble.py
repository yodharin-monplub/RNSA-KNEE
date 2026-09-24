import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.data import UID
from knee.train_heads import score

parts = pd.read_csv('artifacts/partitions/partitions.csv')
targets = np.load('artifacts/partitions/targets.npz')
labels = json.loads(Path('artifacts/partitions/labels.json').read_text())
validation = parts.loc[parts.split.eq('weak_val'), UID]


def read_pred(name, file, ids):
    frame = pd.read_csv(Path('artifacts') / name / file).set_index(UID)
    assert frame.index.is_unique and set(frame.index) == set(ids)
    return frame.loc[ids, labels].to_numpy()


frozen = (read_pred('model2', 'weak_val_predictions.csv', validation) +
          read_pred('model3', 'weak_val_predictions.csv', validation)) / 2
tuned = read_pred('model5', 'weak_val_predictions.csv', validation)
y = targets['weak'][parts.split.eq('weak_val')]
w = targets['confidence'][parts.split.eq('weak_val')]
losses = {}
for alpha in [0, .25, .5, .75, 1]:
    p = np.clip((1-alpha)*frozen + alpha*tuned, 1e-7, 1-1e-7)
    losses[alpha] = float(((-(y*np.log(p)+(1-y)*np.log1p(-p))*w).sum(0) / w.sum(0)).mean())
alpha = min(losses, key=losses.get)
config = {'fine_tuned_weight': alpha, 'selection': 'weak validation confidence-weighted BCE',
          'candidate_losses': losses}
Path('artifacts/ensemble_config.json').write_text(json.dumps(config, indent=2))
gold_ids = parts.loc[parts.split.eq('gold'), UID]
base = (read_pred('model2', 'gold_predictions.csv', gold_ids) +
        read_pred('model3', 'gold_predictions.csv', gold_ids)) / 2
tuned_gold = read_pred('model5', 'gold_predictions.csv', gold_ids)
pred = (1-alpha)*base + alpha*tuned_gold
metrics = score(targets['gold'][parts.split.eq('gold')], pred)
metrics.update({'model': 'model6', 'fine_tuned_weight': alpha, 'gold_n': len(gold_ids),
                'leaderboard_score': None, 'weak_val_bce': losses[alpha]})
out = Path('artifacts/model6'); out.mkdir(exist_ok=True)
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
frame = pd.DataFrame(pred, columns=labels);frame.insert(0, UID, gold_ids.to_numpy())
frame.to_csv(out / 'gold_predictions.csv', index=False)
weights = Path('artifacts/inference_bundle/model5'); weights.mkdir(exist_ok=True)
for name in ['encoder.pt', 'head.pt']:
    shutil.copy(Path('artifacts/model5') / name, weights / name)
env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HOME='/tmp/rsna-offline-empty-cache')
subprocess.run(['python','-m','knee.infer','--data','data','--weights','artifacts/inference_bundle',
                '--models','model5','--output','artifacts/model5_submission.csv'], check=True, env=env)
base_file = Path('artifacts/submission.csv')
shutil.copy(base_file, 'artifacts/model4_submission.csv')
base_frame = pd.read_csv(base_file).set_index(UID)
tuned_frame = pd.read_csv('artifacts/model5_submission.csv').set_index(UID).loc[base_frame.index]
base_frame[labels] = (1-alpha)*base_frame[labels] + alpha*tuned_frame[labels]
assert np.isfinite(base_frame[labels]).all().all()
base_frame.reset_index().to_csv(base_file, index=False)
shutil.copy('artifacts/ensemble_config.json', 'artifacts/inference_bundle/ensemble_config.json')
print(json.dumps(metrics, indent=2))
print('ENSEMBLE_VALIDATED', flush=True)
