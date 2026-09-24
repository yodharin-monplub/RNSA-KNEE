"""Readiness gates for model42: model16's verified inputs with an ImageNet-initialised ConvNeXt config."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model16_ready

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    ready = model16_ready.validate_ready()
    config = json.loads((ROOT / 'model42/config.json').read_text())
    base = json.loads((ROOT / 'model16/config.json').read_text())
    targets = pd.read_csv(ROOT / config['targets'])
    train = pd.read_csv(ROOT / 'data/train.csv')
    assert targets.StudyInstanceUID.is_unique and set(targets.StudyInstanceUID) == set(train.StudyInstanceUID)
    values = targets[[c for c in targets.columns if c != 'StudyInstanceUID']].to_numpy(float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert config['init'] == 'imagenet' and config['teacher_weight'] == 0 and 'select' not in config
    budget = json.loads((ROOT / 'cloud/model42_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items() if k != 'manifests'}, indent=2))
