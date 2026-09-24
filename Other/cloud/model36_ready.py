"""Readiness gates for model36: model16's verified inputs with an ImageNet-initialised ConvNeXt config."""
import json
from pathlib import Path

import model16_ready

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    ready = model16_ready.validate_ready()
    config = json.loads((ROOT / 'model36/config.json').read_text())
    base = json.loads((ROOT / 'model16/config.json').read_text())
    assert config['targets'] == base['targets']
    assert config['init'] == 'imagenet' and config['teacher_weight'] == 0 and 'select' not in config
    budget = json.loads((ROOT / 'cloud/model36_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items() if k != 'manifests'}, indent=2))
