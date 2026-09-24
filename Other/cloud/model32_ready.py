"""Readiness gates for model32: full-data refit reusing model20's verified inputs."""
import json
from pathlib import Path

import model20_ready

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    ready = model20_ready.validate_ready()
    config = json.loads((ROOT / 'model32/config.json').read_text())
    base = json.loads((ROOT / 'model20/config.json').read_text())
    assert config['targets'] == base['targets']
    assert config['select'] == 'last' and config['include_weak_val_in_train'] and config['epochs'] == 1
    budget = json.loads((ROOT / 'cloud/model32_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items() if k not in ('manifests', 'parity', 'smoke')}, indent=2))