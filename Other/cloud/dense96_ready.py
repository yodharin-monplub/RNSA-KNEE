"""Readiness gate for runs on the 96-frame, 336 px cache (kernels rsna-knee-dense-cache-336-part{0..3}-s96).

Same checks as model16_ready (smoke/partition/targets provenance, full UID coverage, shard bookkeeping, disk, budget)
but against the 128-frame manifests, which must be downloaded first:
  artifacts/dense_cache_s96_part{p}_notebook/output/cache_manifest.json
Usage (imported by cloud/<run>_ready.py): validate_ready(run) with <run>/config.json and cloud/<run>_budget.json.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = {'slots': [['Sagittal', 1, 24], ['Sagittal', 0, 18], ['Coronal', 1, 18], ['Coronal', 0, 18], ['Axial', -1, 18]],
            'span': [.02, .98], 'crop_mm': 140, 'size': 336}


def validate_ready(run):
    smoke_root = ROOT / 'artifacts/model16_smoke_notebook/output/model16_smoke'
    provenance = json.loads((smoke_root / 'provenance.json').read_text())
    config = json.loads((ROOT / run / 'config.json').read_text())
    assert (ROOT / config['targets']).exists()
    partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv')
    assert partition.StudyInstanceUID.is_unique
    train = pd.read_csv(ROOT / 'data/train.csv')
    train_hash = hashlib.sha256((ROOT / 'data/train.csv').read_bytes()).hexdigest()
    manifests, ids, files = [], [], set()
    for part in range(4):
        path = ROOT / f'artifacts/dense_cache_s96_part{part}_notebook/output/cache_manifest.json'
        m = json.loads(path.read_text())
        assert m['kind'] == 'full_part' and m['part'] == part and m['parts'] == 4
        assert m['training_metadata_sha256'] == train_hash
        assert m['geometry'] == GEOMETRY, m['geometry']
        rows = 0
        for shard in m['shards']:
            assert Path(shard['file']).name == shard['file'] and shard['file'] not in files
            assert len(shard['sha256']) == 64 and shard['bytes'] > 0
            files.add(shard['file']); ids.extend(shard['ids']); rows += len(shard['ids'])
        assert rows == m['studies']
        assert sum(s['bytes'] for s in m['shards']) == m['compressed_bytes']
        manifests.append(m)
    assert len(ids) == len(set(ids)) == 4407 and set(ids) == set(train.StudyInstanceUID)
    compressed = sum(m['compressed_bytes'] for m in manifests)
    budget = json.loads((ROOT / f'cloud/{run}_budget.json').read_text())
    # shards + unpacked arrays + 20 GB working room
    assert (compressed + 4407 * 96 * 336 * 336) / 1e9 + 20 < budget['disk_gb'], 'raise disk_gb'
    prior_cost = 0.0
    for path in (ROOT / 'artifacts/cloud').rglob('finished.json'):
        previous = json.loads(path.read_text())
        prior_cost += max(previous.get('observed_credit_decrease_usd', 0), previous['estimated_compute_usd'])
    assert prior_cost + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    return {'manifests': manifests, 'compressed_bytes': compressed, 'prior_cost_usd': prior_cost,
            'initial_checkpoint_sha256': provenance['initial_checkpoint_sha256'],
            'training_metadata_sha256': train_hash, 'config': config}
