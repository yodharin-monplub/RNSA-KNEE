"""Readiness gates for model79: model16's verified cache and public maxspan checkpoint, full fine-tune config."""
import json
from pathlib import Path

import model16_ready

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    ready = model16_ready.validate_ready()
    config = json.loads((ROOT / 'model79/config.json').read_text())
    assert 'init' not in config and 'arch' not in config, 'model79 fine-tunes the public Raptor checkpoint'
    assert config['unfreeze_stages'] == 4 and config['teacher_weight'] == 0 and config['plain_bce'] and config['augment']
    assert (ROOT / config['targets']).exists()
    budget = json.loads((ROOT / 'cloud/model79_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in validate_ready().items() if k != 'manifests'}, indent=2))
