"""Readiness gate for model86: the 128-frame 256 px cache (cloud/dense128_ready.py)."""
import json

import dense128_ready


def validate_ready():
    ready = dense128_ready.validate_ready('model86')
    config = ready['config']
    assert config['init'] == 'imagenet' and config['arch'].startswith('efficientnet') and config['res'] == 256
    return ready


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in validate_ready().items() if k not in ('manifests',)}, indent=2))
