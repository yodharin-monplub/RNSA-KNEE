"""Readiness gates for model90: model16's verified cache, ResNet34 ImageNet weights, Gemini targets, 5 folds."""
import json
from pathlib import Path

import model16_ready

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    ready = model16_ready.validate_ready()
    config = json.loads((ROOT / 'model90/config.json').read_text())
    assert config['init'] == 'imagenet' and len(config['pretrained_sha256']) == 64 and (ROOT / config['targets']).exists()
    assert config['arch'].startswith('coatnet') and config['res'] == 384 and 'backbone_img_size' not in config and 'fold' not in config and 'n_folds' not in config
    budget = json.loads((ROOT / 'cloud/model90_budget.json').read_text())
    assert ready['prior_cost_usd'] + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    ready['config'] = config
    return ready


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in validate_ready().items() if k != 'manifests'}, indent=2))
