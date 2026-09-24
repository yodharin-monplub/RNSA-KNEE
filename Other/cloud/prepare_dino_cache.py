"""Verify the DINO cache shards and unpack them to .npy for random-access training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_GEOMETRY = {
    'slots': ['SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS', 'SAG_FLUID_NOFS', 'COR_T1', 'SAG_T1'],
    'slot_scheme': 'recovered', 'group': 3, 'n_group': 1, 'size': 336, 'crop_mm': 130.0,
    'band': [0.2, 0.8],
    'rules': {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'},
}


def prepare(manifests, train_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'verified.json').unlink(missing_ok=True)
    train = pd.read_csv(train_path)
    assert train.StudyInstanceUID.is_unique and len(train) == 4407
    metadata_hash = hashlib.sha256(Path(train_path).read_bytes()).hexdigest()
    rows, seen, parts, names = [], set(), set(), set()
    for manifest_path in map(Path, manifests):
        manifest = json.loads(manifest_path.read_text())
        assert manifest['geometry'] == EXPECTED_GEOMETRY, manifest['geometry']
        assert manifest['kind'] == 'dino_part' and manifest['parts'] == 4
        assert manifest['training_metadata_sha256'] == metadata_hash
        assert manifest['part'] not in parts
        parts.add(manifest['part'])
        count = 0
        for shard in manifest['shards']:
            name = shard['file']
            assert Path(name).name == name and name.endswith('.npz') and name not in names
            names.add(name)
            path = manifest_path.parent / name
            assert path.stat().st_size == shard['bytes']
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
            with np.load(path, allow_pickle=False) as archive:
                ids, volumes, masks = archive['ids'].tolist(), archive['volumes'], archive['masks']
            assert ids == shard['ids'] and not seen.intersection(ids)
            assert volumes.dtype == np.uint8 and volumes.shape == (len(ids), 6, 3, 336, 336)
            assert masks.shape == (len(ids), 6) and masks.any(axis=1).all()
            base = Path(name).stem
            for suffix, array in [('volumes', volumes), ('masks', masks.astype(np.float32))]:
                target = destination / f'{base}_{suffix}.npy'
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, array, allow_pickle=False)
                temporary.replace(target)
            for i, uid in enumerate(ids):
                rows.append({'StudyInstanceUID': uid, 'volume_file': f'{base}_volumes.npy',
                             'mask_file': f'{base}_masks.npy', 'row': i})
            seen.update(ids)
            count += len(ids)
            print('Verified/unpacked', name, len(ids), flush=True)
        assert count == manifest['studies']
    assert parts == {'0', '1', '2', '3'} or parts == {0, 1, 2, 3}, parts
    assert seen == set(train.StudyInstanceUID), (len(seen), len(train))
    index = pd.DataFrame(rows).set_index('StudyInstanceUID')
    index = index.loc[[uid for uid in train.StudyInstanceUID if uid in seen]].reset_index()
    index.to_csv(destination / 'index.csv', index=False)
    summary = {'studies': len(index), 'geometry': EXPECTED_GEOMETRY,
               'source_metadata_sha256': metadata_hash, 'verified_shards': len(names)}
    (destination / 'verified.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', nargs='+', required=True)
    parser.add_argument('--train', default='data/train.csv')
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    print(prepare(args.manifest, args.train, args.destination))
