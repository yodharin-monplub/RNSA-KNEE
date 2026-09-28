"""Readiness gate for model100: the 96-frame 336 px cache (cloud/dense96_ready.py)."""
import json

import dense96_ready


def validate_ready():
    ready = dense96_ready.validate_ready('model100')
    config = ready['config']
    assert config['init'] == 'imagenet' and config['arch'].startswith('coatnet') and config['res'] == 384
    return ready


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in validate_ready().items() if k not in ('manifests',)}, indent=2))
