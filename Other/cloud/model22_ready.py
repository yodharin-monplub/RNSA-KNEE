"""Readiness gates for exact 6%-94% model22 paid training."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    parity = json.loads((ROOT / 'artifacts/dense694_inference_validation_notebook/output/cache_inference_validation.json').read_text())
    assert parity['status'] == 'PASS' and parity['studies'] == 58
    assert parity['max_abs_difference_vs_raw'] < .001
    smoke_root = ROOT / 'artifacts/model22_smoke_notebook/output/model22_smoke'
    smoke = json.loads((smoke_root / 'SMOKE_PASS.json').read_text())
    provenance = json.loads((smoke_root / 'provenance.json').read_text())
    config = json.loads((ROOT / 'model22/config.json').read_text())
    geometry = {'slots':[['Sagittal',1,18],['Sagittal',0,14],['Coronal',1,12],
                         ['Coronal',0,8],['Axial',-1,12]],
                'span':[.06,.94], 'crop_mm':140, 'size':336}
    assert provenance['config'] == config and provenance['smoke']
    assert provenance['cache']['geometry'] == geometry
    assert smoke['trainable_parameters'] > 0 and smoke['weak_training_cases']
    partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv')
    assert set(smoke['weak_training_cases']).issubset(
        set(partition.loc[partition.split == 'train', 'StudyInstanceUID']))
    assert provenance['targets_sha256'] == hashlib.sha256((ROOT / config['targets']).read_bytes()).hexdigest()
    assert provenance['partition_sha256'] == hashlib.sha256((ROOT / 'artifacts/partitions/partitions.csv').read_bytes()).hexdigest()
    train = pd.read_csv(ROOT / 'data/train.csv')
    train_hash = hashlib.sha256((ROOT / 'data/train.csv').read_bytes()).hexdigest()
    manifests, ids, files = [], [], set()
    for part in range(4):
        path = ROOT / f'artifacts/dense_cache_6_94_part{part}_notebook/output/cache_manifest.json'
        manifest = json.loads(path.read_text())
        assert manifest['kind'] == 'full_part' and manifest['part'] == part and manifest['parts'] == 4
        assert manifest['training_metadata_sha256'] == train_hash and manifest['geometry'] == geometry
        rows = 0
        for shard in manifest['shards']:
            assert Path(shard['file']).name == shard['file'] and shard['file'] not in files
            assert len(shard['sha256']) == 64 and shard['bytes'] > 0
            assert len(shard['ids']) == len(set(shard['ids']))
            files.add(shard['file']); ids.extend(shard['ids']); rows += len(shard['ids'])
        assert rows == manifest['studies']
        assert sum(shard['bytes'] for shard in manifest['shards']) == manifest['compressed_bytes']
        manifests.append(manifest)
    assert len(ids) == len(set(ids)) == 4407 and set(ids) == set(train.StudyInstanceUID)
    compressed = sum(manifest['compressed_bytes'] for manifest in manifests)
    unpacked = 4407 * 64 * 336 * 336
    budget = json.loads((ROOT / 'cloud/model22_budget.json').read_text())
    assert (compressed + unpacked) / 1e9 + 25 < budget['disk_gb']
    assert compressed / 1e9 < budget['max_transfer_gb']
    prior_cost = 0.0
    for path in (ROOT / 'artifacts/cloud').rglob('finished.json'):
        previous = json.loads(path.read_text())
        prior_cost += max(previous.get('observed_credit_decrease_usd', 0),
                          previous['estimated_compute_usd'])
    assert prior_cost + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    return {'manifests':manifests, 'compressed_bytes':compressed,
            'unpacked_bytes':unpacked, 'prior_cost_usd':prior_cost,
            'initial_checkpoint_sha256':provenance['initial_checkpoint_sha256'],
            'training_metadata_sha256':train_hash, 'parity':parity, 'smoke':smoke}


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({key:value for key,value in summary.items() if key != 'manifests'}, indent=2))

