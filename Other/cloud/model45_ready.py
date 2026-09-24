"""Readiness gates for model45: model16's inputs and recipe with the Gemini 3.8 Flash relabel targets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model16_ready

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def validate_ready():
    ready = model16_ready.validate_ready()
    config = json.loads((ROOT / 'model45/config.json').read_text())
    base = json.loads((ROOT / 'model16/config.json').read_text())
    assert config['fold'] in ('A', 'B') and config['select'] == 'last' and config['epochs'] == 1
    targets = pd.read_csv(ROOT / config['targets'])
    train = pd.read_csv(ROOT / 'data/train.csv')
    assert targets.StudyInstanceUID.is_unique and set(targets.StudyInstanceUID) == set(train.StudyInstanceUID)
    values = targets[LABELS].to_numpy(float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    budget = json.loads((ROOT / 'cloud/model45_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items() if k not in ('manifests',)}, indent=2))