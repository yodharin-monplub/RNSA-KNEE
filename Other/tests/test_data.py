import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from knee.data import PixelCache, UID, make_partitions


class DataTests(unittest.TestCase):
    def test_cache_reordering_preserves_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({UID: ['a', 'b'], 'shard': ['pixels.npy'] * 2,
                          'row': [0, 1]}).to_csv(root / 'studies.csv', index=False)
            values = np.zeros((2, 6, 9, 224, 224), dtype=np.uint8)
            values[1] = 7
            np.save(root / 'pixels.npy', values)
            np.save(root / 'slot_mask.npy', np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]))
            cache = PixelCache(root, ['b', 'a'])
            pixels, mask = cache[0]
            self.assertEqual(int(pixels.max()), 7)
            self.assertEqual(mask.tolist(), [False, True, False, False, False, False])

    def test_real_partitions_have_no_report_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = make_partitions('data', tmp)
            groups = {s: set(frame.loc[frame.split.eq(s), 'report_hash'])
                      for s in ['train', 'weak_val', 'gold']}
            self.assertFalse(groups['train'] & groups['weak_val'])
            self.assertFalse(groups['train'] & groups['gold'])
            self.assertFalse(groups['weak_val'] & groups['gold'])
            self.assertEqual(int(frame.split.eq('gold').sum()), 58)
            targets = np.load(Path(tmp) / 'targets.npz')
            self.assertTrue((targets['confidence'][targets['weak'] == 0.5] == 0).all())


if __name__ == '__main__':
    unittest.main()
