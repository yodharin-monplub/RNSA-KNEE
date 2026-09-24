"""Bounded-memory reader for the unpacked DINO member cache (6 slots x 3 slices x 336 x 336)."""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


class DinoCache:
    def __init__(self, root, ids=None, open_files=8):
        self.root = Path(root)
        self.verified = json.loads((self.root / 'verified.json').read_text())
        self.index = pd.read_csv(self.root / 'index.csv').set_index('StudyInstanceUID')
        if ids is not None:
            missing = [uid for uid in ids if uid not in self.index.index]
            assert not missing, f'{len(missing)} studies missing from the cache'
            self.index = self.index.loc[list(ids)]
        self.open_files = open_files
        self._arrays = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(self.root / name, mmap_mode='r')
            while len(self._arrays) > self.open_files:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(name)
        return self._arrays[name]

    def __getitem__(self, i):
        row = self.index.iloc[i]
        volume = np.asarray(self._array(row.volume_file)[row.row])
        mask = np.asarray(self._array(row.mask_file)[row.row])
        assert volume.shape == (6, 3, 336, 336) and volume.dtype == np.uint8
        assert mask.shape == (6,) and mask.any()
        return volume, mask
