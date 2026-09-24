"""Verify generated shards and unpack them for efficient random-access training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_GEOMETRY = {
    'slots': [['Sagittal', 1, 18], ['Sagittal', 0, 14], ['Coronal', 1, 12],
              ['Coronal', 0, 8], ['Axial', -1, 12]],
    'span': [.02, .98], 'crop_mm': 140, 'size': 336,
}
# native384-v8 layout: 44 frames stored at the front of the same 64-frame arrays (rest zero/masked).
SLOTS44 = [['Sagittal', 1, 12], ['Sagittal', 0, 10], ['Coronal', 1, 8],
           ['Coronal', 0, 6], ['Axial', -1, 8]]


def prepare(manifests, train_path, destination, pilot=False, size=336, span=(.02,.98), slots=64):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must never retain an old ready marker.
    (destination / 'verified.json').unlink(missing_ok=True)
    train = pd.read_csv(train_path)
    assert train.StudyInstanceUID.is_unique and len(train) == 4407
    expected_ids = set(train.StudyInstanceUID)
    metadata_hash = hashlib.sha256(Path(train_path).read_bytes()).hexdigest()
    assert size in (336, 384)
    expected_geometry = dict(EXPECTED_GEOMETRY)
    expected_geometry['size'] = size
    expected_geometry['span'] = list(span)
    if slots == 44:
        expected_geometry['slots'] = SLOTS44
    rows, seen, parts, shard_names = [], set(), set(), set()
    for manifest_path in map(Path, manifests):
        manifest = json.loads(manifest_path.read_text())
        assert manifest['geometry'] == expected_geometry
        assert manifest['kind'] == ('pilot' if pilot else 'full_part')
        if not pilot:
            assert manifest['parts'] == 4 and manifest['part'] not in parts
            assert manifest['training_metadata_sha256'] == metadata_hash
            parts.add(manifest['part'])
        manifest_count = 0
        for shard in manifest['shards']:
            filename = shard['file']
            assert Path(filename).name == filename and filename.endswith('.npz')
            assert filename not in shard_names
            shard_names.add(filename)
            path = manifest_path.parent / filename
            assert path.stat().st_size == shard['bytes']
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
            with np.load(path, allow_pickle=False) as archive:
                ids = archive['ids'].tolist()
                volumes, masks = archive['volumes'], archive['masks']
            assert ids == shard['ids'] and len(set(ids)) == len(ids)
            assert not seen.intersection(ids) and set(ids).issubset(expected_ids)
            assert volumes.dtype == np.uint8 and volumes.shape == (len(ids), 64, size, size)
            assert masks.shape == (len(ids), 64)
            assert np.array_equal(masks, volumes.reshape(len(ids), 64, -1).sum(2) > 0)
            assert masks.any(axis=1).all()
            if slots == 44:
                assert not volumes[:, 44:].any()
            basename = Path(filename).stem
            for suffix, array in [('volumes', volumes), ('masks', masks.astype(bool))]:
                target = destination / f'{basename}_{suffix}.npy'
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, array, allow_pickle=False)
                temporary.replace(target)
            for i, uid in enumerate(ids):
                rows.append({'StudyInstanceUID': uid, 'volume_file': f'{basename}_volumes.npy',
                             'mask_file': f'{basename}_masks.npy', 'row': i})
            seen.update(ids)
            manifest_count += len(ids)
            print('Verified/unpacked', filename, 'studies', len(ids), flush=True)
        assert manifest_count == manifest['studies']
    if pilot:
        assert len(seen) == 64
    else:
        assert parts == {0, 1, 2, 3} and seen == expected_ids
    index = pd.DataFrame(rows).set_index('StudyInstanceUID')
    order = [uid for uid in train.StudyInstanceUID if uid in seen]
    index = index.loc[order].reset_index()
    index.to_csv(destination / 'index.csv', index=False)
    summary = {'studies': len(index), 'pilot': pilot, 'geometry': expected_geometry,
               'source_metadata_sha256': metadata_hash, 'verified_shards': len(shard_names)}
    (destination / 'verified.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', nargs='+', required=True)
    parser.add_argument('--train', default='data/train.csv')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--size', type=int, choices=[336,384], default=336)
    parser.add_argument('--span-start', type=float, default=.02)
    parser.add_argument('--span-end', type=float, default=.98)
    parser.add_argument('--slots', type=int, choices=[64, 44], default=64)
    args = parser.parse_args()
    assert (args.span_start,args.span_end) in ((.02,.98),(.06,.94))
    print(prepare(args.manifest, args.train, args.destination, args.pilot, args.size,
                  (args.span_start,args.span_end), args.slots))
