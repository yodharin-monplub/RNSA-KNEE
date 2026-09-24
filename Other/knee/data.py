import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

UID = 'StudyInstanceUID'


def report_hash(value):
    return hashlib.sha256(' '.join(str(value).lower().split()).encode()).hexdigest()


def make_partitions(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    train = pd.read_csv(data_dir / 'train.csv')
    labels = list(pd.read_csv(data_dir / 'sample_submission.csv').columns.drop(UID))
    weak = pd.read_csv(data_dir / 'weak_labels/llm_labels_full.csv')
    assert train[UID].is_unique and weak[UID].is_unique
    assert set(train[UID]) == set(weak[UID]), 'Weak target UID mismatch'
    probs = weak.set_index(UID).loc[train[UID], labels].to_numpy(dtype=np.float32)
    assert np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
    groups = train.Report.map(report_hash)
    gold_mask = train[labels].notna().all(axis=1).to_numpy()
    excluded_groups = set(groups[gold_mask])
    eligible = np.flatnonzero(~gold_mask & ~groups.isin(excluded_groups).to_numpy())
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=2026)
                .split(eligible, groups=groups.iloc[eligible]))
    split = np.full(len(train), 'excluded', dtype=object)
    split[eligible[a]] = 'train'
    split[eligible[b]] = 'weak_val'
    split[gold_mask] = 'gold'
    assert set(groups[split == 'train']).isdisjoint(groups[split == 'weak_val'])
    assert set(groups[split == 'train']).isdisjoint(groups[split == 'gold'])
    manifest = train[[UID]].assign(split=split, report_hash=groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / 'partitions.csv', index=False)
    np.savez(out_dir / 'targets.npz', weak=probs,
             gold=train[labels].to_numpy(dtype=np.float32),
             confidence=2 * np.abs(probs - 0.5))
    (out_dir / 'labels.json').write_text(json.dumps(labels))
    return manifest


class PixelCache:
    def __init__(self, root, ids=None):
        self.root = Path(root)
        index = pd.read_csv(self.root / 'studies.csv')
        if not index[UID].is_unique:
            raise ValueError('Duplicate cache study ID')
        index['mask_row'] = np.arange(len(index))
        self.index = index if ids is None else index.set_index(UID).loc[list(ids)].reset_index()
        self.mask = np.load(self.root / 'slot_mask.npy', allow_pickle=False)
        self.shards = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        row = self.index.iloc[i]
        shard = str(row['shard'])
        if shard not in self.shards:
            self.shards[shard] = np.load(self.root / shard, mmap_mode='r', allow_pickle=False)
        pixels = np.array(self.shards[shard][int(row['row'])], copy=True)
        mask = np.asarray(self.mask[int(row['mask_row'])], dtype=bool)
        if pixels.shape != (6, 9, 224, 224) or pixels.dtype != np.uint8:
            raise ValueError(f'Unexpected pixel shape/dtype: {pixels.shape}, {pixels.dtype}')
        if not mask.any():
            raise ValueError('Study has no available MRI slots')
        return pixels, mask
