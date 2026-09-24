"""Readiness gates for model60: model16's inputs and recipe with the hybrid (Gemini + Lee v4 Effusion/Baker's) targets."""
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
    config = json.loads((ROOT / 'model60/config.json').read_text())
    base = json.loads((ROOT / 'model38/config.json').read_text())
    assert {k: v for k, v in config.items() if k != 'targets'} == {k: v for k, v in base.items() if k != 'targets'}
    targets = pd.read_csv(ROOT / config['targets'])
    gemini = pd.read_csv(ROOT / base['targets']).set_index('StudyInstanceUID')
    train = pd.read_csv(ROOT / 'data/train.csv')
    assert targets.StudyInstanceUID.is_unique and set(targets.StudyInstanceUID) == set(train.StudyInstanceUID)
    values = targets[LABELS].to_numpy(float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    # Only Effusion and Baker's differ from model38's Gemini targets.
    indexed = targets.set_index('StudyInstanceUID').loc[gemini.index]
    unchanged = [c for c in LABELS if c not in ('Effusion', "Baker's")]
    assert np.allclose(indexed[unchanged].to_numpy(float), gemini[unchanged].to_numpy(float))
    budget = json.loads((ROOT / 'cloud/model60_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items() if k not in ('manifests',)}, indent=2))
