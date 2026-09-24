"""Readiness gates for model28: native384-v8 fine-tune on the s44 (44 slots, 6%-94%, 384px) cache."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V8_SHA256 = '3db77388662d516bb0ce790dafd24e36f4991f6089af26daf546817fe1c21862'
GEOMETRY = {'slots': [['Sagittal', 1, 12], ['Sagittal', 0, 10], ['Coronal', 1, 8],
                      ['Coronal', 0, 6], ['Axial', -1, 8]],
            'span': [.06, .94], 'crop_mm': 140, 'size': 384}


def validate_ready():
    parity = json.loads((ROOT / 'artifacts/v8_parity_notebook/output/v8_parity.json').read_text())
    assert parity['status'] == 'PASS' and parity['studies'] == 58
    assert parity['max_prob_diff_cached_vs_raw'] < .001 and parity['cached_frames_after_44_nonzero'] == 0
    config = json.loads((ROOT / 'model28/config.json').read_text())
    base = json.loads((ROOT / 'model16/config.json').read_text())
    assert {k: v for k, v in config.items() if k != 'seed'} == {k: v for k, v in base.items() if k != 'seed'}
    partition = pd.read_csv(ROOT / 'artifacts/partitions/partitions.csv')
    assert set(partition.StudyInstanceUID) == set(pd.read_csv(ROOT / config['targets']).StudyInstanceUID)
    train = pd.read_csv(ROOT / 'data/train.csv')
    train_hash = hashlib.sha256((ROOT / 'data/train.csv').read_bytes()).hexdigest()
    manifests, ids, files = [], [], set()
    for part in range(4):
        path = ROOT / f'artifacts/dense_cache_384_6_94_s44_part{part}_notebook/output/cache_manifest.json'
        manifest = json.loads(path.read_text())
        assert manifest['kind'] == 'full_part' and manifest['part'] == part and manifest['parts'] == 4
        assert manifest['training_metadata_sha256'] == train_hash and manifest['geometry'] == GEOMETRY
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
    unpacked = 4407 * 64 * 384 * 384
    budget = json.loads((ROOT / 'cloud/model28_budget.json').read_text())
    assert (compressed + unpacked) / 1e9 + 25 < budget['disk_gb']
    assert compressed / 1e9 < budget['max_transfer_gb']
    prior_cost = 0.0
    for path in (ROOT / 'artifacts/cloud').rglob('finished.json'):
        previous = json.loads(path.read_text())
        prior_cost += max(previous.get('observed_credit_decrease_usd', 0), previous['estimated_compute_usd'])
    assert prior_cost + budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    return {'manifests': manifests, 'compressed_bytes': compressed, 'unpacked_bytes': unpacked,
            'prior_cost_usd': prior_cost, 'initial_checkpoint_sha256': V8_SHA256,
            'training_metadata_sha256': train_hash, 'parity': parity}


if __name__ == '__main__':
    summary = validate_ready()
    print(json.dumps({k: v for k, v in summary.items() if k != 'manifests'}, indent=2))
