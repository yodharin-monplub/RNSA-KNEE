"""Memory-mapped access to the verified model12-compatible dense MRI cache."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DenseCache:
    def __init__(self, root, ids=None, max_open_shards=8):
        self.root = Path(root)
        self.verified = json.loads((self.root / 'verified.json').read_text())
        frame = pd.read_csv(self.root / 'index.csv').set_index('StudyInstanceUID')
        assert frame.index.is_unique and len(frame) == self.verified['studies']
        self.index = frame if ids is None else frame.loc[list(ids)]
        self.open_arrays = OrderedDict()
        self.max_open_shards = max_open_shards
        self.size = int(self.verified['geometry']['size'])
        assert self.size in (256, 336, 384)
        # 64-frame arrays for the 64- and 44-slot layouts, 128 for the 128-slot layout
        slot_frames = sum(int(slot[2]) for slot in self.verified['geometry']['slots'])
        self.frames = 128 if slot_frames == 128 else 64
        assert max_open_shards >= 1

    def __len__(self):
        return len(self.index)

    def _open(self, filename):
        assert Path(filename).name == filename
        if filename in self.open_arrays:
            array = self.open_arrays.pop(filename)
        else:
            array = np.load(self.root / filename, mmap_mode='r', allow_pickle=False)
        self.open_arrays[filename] = array
        while len(self.open_arrays) > self.max_open_shards * 2:
            self.open_arrays.popitem(last=False)
        return array

    def __getitem__(self, i):
        row = self.index.iloc[i]
        volume = np.array(self._open(row.volume_file)[int(row.row)], copy=True)
        mask = np.array(self._open(row.mask_file)[int(row.row)], dtype=bool, copy=True)
        assert volume.shape == (self.frames, self.size, self.size) and volume.dtype == np.uint8
        assert mask.shape == (self.frames,) and mask.any()
        return volume, mask
