"""Local readiness gates; never launch a rental for incomplete training inputs."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def validate_ready():
    parity_path = ROOT/'artifacts/dense_inference_validation_notebook/output/cache_inference_validation.json'
    parity = json.loads(parity_path.read_text())
    assert parity['status']=='PASS' and parity['studies']==58
    assert parity['max_abs_difference_vs_raw'] < .001
    smoke_root = ROOT/'artifacts/model16_smoke_notebook/output/model16_smoke'
    smoke = json.loads((smoke_root/'SMOKE_PASS.json').read_text())
    provenance = json.loads((smoke_root/'provenance.json').read_text())
    config = json.loads((ROOT/'model16/config.json').read_text())
    assert provenance['config']==config and provenance['smoke']
    assert smoke['trainable_parameters']>0 and smoke['weak_training_cases']
    partition = pd.read_csv(ROOT/'artifacts/partitions/partitions.csv')
    assert set(smoke['weak_training_cases']).issubset(set(partition.loc[partition.split=='train','StudyInstanceUID']))
    assert provenance['targets_sha256']==hashlib.sha256((ROOT/config['targets']).read_bytes()).hexdigest()
    assert provenance['partition_sha256']==hashlib.sha256((ROOT/'artifacts/partitions/partitions.csv').read_bytes()).hexdigest()
    train = pd.read_csv(ROOT/'data/train.csv')
    train_hash = hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest()
    manifests, ids, files = [], [], set()
    for part in range(4):
        path = ROOT/f'artifacts/dense_cache_part{part}_notebook/output/cache_manifest.json'
        m = json.loads(path.read_text())
        assert m['kind']=='full_part' and m['part']==part and m['parts']==4
        assert m['training_metadata_sha256']==train_hash
        assert m['geometry']==provenance['cache']['geometry']
        rows = 0
        for shard in m['shards']:
            assert Path(shard['file']).name==shard['file'] and shard['file'] not in files
            assert len(shard['sha256'])==64 and shard['bytes']>0
            files.add(shard['file']);ids.extend(shard['ids']);rows+=len(shard['ids'])
        assert rows==m['studies']
        assert sum(s['bytes'] for s in m['shards'])==m['compressed_bytes']
        manifests.append(m)
    assert len(ids)==len(set(ids))==4407 and set(ids)==set(train.StudyInstanceUID)
    compressed = sum(m['compressed_bytes'] for m in manifests)
    budget = json.loads((ROOT/'cloud/model16_budget.json').read_text())
    assert (compressed+4407*64*336*336)/1e9 + 20 < budget['disk_gb']
    # Reserve the whole next-run cap against the campaign limit.
    prior_cost = 0.0
    for path in (ROOT/'artifacts/cloud').rglob('finished.json'):
        previous = json.loads(path.read_text())
        prior_cost += max(previous.get('observed_credit_decrease_usd',0),previous['estimated_compute_usd'])
    assert prior_cost+budget['pilot_cap_usd'] <= budget['campaign_cap_usd']
    return {'manifests':manifests,'compressed_bytes':compressed,'prior_cost_usd':prior_cost,
            'initial_checkpoint_sha256':provenance['initial_checkpoint_sha256'],
            'training_metadata_sha256':train_hash}
