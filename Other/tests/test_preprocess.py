import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from knee.preprocess import ordered_paths, render_series, slice_indices


class PreprocessTests(unittest.TestCase):
    def test_short_series_indices(self):
        self.assertEqual(slice_indices(1), [0] * 9)
        for n in range(1, 101):
            indices = slice_indices(n)
            self.assertEqual(len(indices), 9)
            self.assertTrue(all(0 <= i < n for i in indices))
        self.assertEqual(slice_indices(10), [2, 3, 4, 3, 4, 5, 4, 5, 6])

    def test_physical_order_overrides_filename_and_instance_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                p = Path(tmp) / f'{2-i}.dcm'
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = generate_uid()
                ds = FileDataset(str(p), {}, file_meta=meta, preamble=b'\0' * 128)
                ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                ds.Rows = 32; ds.Columns = 32; ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = 'MONOCHROME2'
                ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.ImagePositionPatient = [0, 0, float(i)]
                ds.InstanceNumber = 100-i; ds.PixelSpacing = [0.5, 0.5]
                ds.PixelData = (np.arange(1024).reshape(32,32) + i*100).astype(np.uint16).tobytes()
                ds.save_as(p, enforce_file_format=True); paths.append(p)
            ordered, spacing = ordered_paths(sorted(paths))
            self.assertEqual(ordered, paths)
            self.assertEqual(spacing, 0.5)
            pixels = render_series(sorted(paths))
            self.assertEqual(pixels.shape, (9, 224, 224))
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertTrue(np.array_equal(pixels[0], pixels[3]))
            self.assertGreater(float(pixels[2].mean()), float(pixels[0].mean()))


if __name__ == '__main__':
    unittest.main()
