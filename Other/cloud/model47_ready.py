"""Readiness gates for model47: fine-tuning the twenty public DINOv2 members on our labels."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
GEOMETRY = json.loads((ROOT / 'cloud/dino_geometry.json').read_text()) if (
    ROOT / 'cloud/dino_geometry.json').exists() else None


def validate_ready():
    load_dotenv(ROOT / '.env')
    os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
    os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
    from kaggle.api.kaggle_api_extended import KaggleApi
    from prepare_dino_cache import EXPECTED_GEOMETRY

    config = json.loads((ROOT / 'model47/config.json').read_text())
    targets = pd.read_csv(ROOT / config['targets'])
    train = pd.read_csv(ROOT / 'data/train.csv')
    assert targets.StudyInstanceUID.is_unique and set(targets.StudyInstanceUID) == set(train.StudyInstanceUID)
    values = targets[LABELS].to_numpy(float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert 0.5 <= config['subset_fraction'] <= 1 and config['epochs'] >= 1

    manifests, ids, names, total = [], [], set(), 0
    for part in range(4):
        path = ROOT / f'artifacts/dino_cache_{part}_notebook/output/cache_manifest.json'
        manifest = json.loads(path.read_text())
        assert manifest['kind'] == 'dino_part' and str(manifest['part']) == str(part) and manifest['parts'] == 4
        assert manifest['geometry'] == EXPECTED_GEOMETRY, manifest['geometry']
        for shard in manifest['shards']:
            assert shard['file'] not in names and len(shard['sha256']) == 64 and shard['bytes'] > 0
            names.add(shard['file']); ids.extend(shard['ids'])
        assert sum(len(s['ids']) for s in manifest['shards']) == manifest['studies']
        total += manifest['compressed_bytes']
        manifests.append(manifest)
    assert len(ids) == len(set(ids)) == 4407 and set(ids) == set(train.StudyInstanceUID)

    member_manifest = json.loads((ROOT / 'artifacts/sources/pilkwang__rsna-knee-weights/manifest.json').read_text())
    assert len(member_manifest['members']) == 20
    api = KaggleApi(); api.authenticate()
    listing = {f.name: int(getattr(f, 'total_bytes', 0) or 0)
               for f in (api.dataset_list_files('pilkwang/rsna-knee-weights').files or [])}
    member_files = {}
    for member in member_manifest['members']:
        assert member['file'] in listing and listing[member['file']] > 1_000_000
        member_files[member['file']] = listing[member['file']]
    budget = json.loads((ROOT / 'cloud/model47_budget.json').read_text())
    unpacked = 4407 * 6 * 3 * 336 * 336
    assert (total + unpacked + sum(member_files.values())) / 1e9 + 15 < budget['disk_gb']
    assert (total + sum(member_files.values())) / 1e9 < budget['max_transfer_gb']
    prior_cost = 0.0
    for path in (ROOT / 'artifacts/cloud').rglob('finished.json'):
        previous = json.loads(path.read_text())
        prior_cost += max(previous.get('observed_credit_decrease_usd', 0), previous['estimated_compute_usd'])
    assert prior_cost + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    return {'manifests': manifests, 'compressed_bytes': total, 'unpacked_bytes': unpacked,
            'member_files': member_files, 'member_manifest': member_manifest, 'prior_cost_usd': prior_cost,
            'config': config}


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('manifests', 'member_manifest', 'member_files')}, indent=2))
