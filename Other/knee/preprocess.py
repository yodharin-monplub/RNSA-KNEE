"""DICOM geometry compatible with Dmitrii Gluzdov's version-1 pixel cache.

Geometry attribution: Dmitrii Gluzdov and Steven Lee (Apache-2.0).
See SOURCES.md and third_party/pixel_cache/NOTICE.md. Adapted for test inference.
"""
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import zoom

SLOTS = [(plane, fluid) for fluid in [1, 0] for plane in ['Sagittal', 'Coronal', 'Axial']]


def finite_vector(value, size):
    try:
        a = np.asarray(value, dtype=float)
        return a if a.shape == (size,) and np.isfinite(a).all() else None
    except (ValueError, TypeError):
        return None


def ordered_paths(paths):
    headers = [pydicom.dcmread(p, stop_before_pixels=True, specific_tags=[
        'ImagePositionPatient', 'ImageOrientationPatient', 'SliceLocation',
        'InstanceNumber', 'PixelSpacing'], force=True) for p in paths]
    normal = None
    for header in headers:
        orientation = finite_vector(getattr(header, 'ImageOrientationPatient', None), 6)
        if orientation is not None:
            vector = np.cross(orientation[:3], orientation[3:])
            if np.linalg.norm(vector) > 1e-6:
                normal = vector / np.linalg.norm(vector)
                break
    positions = [finite_vector(getattr(h, 'ImagePositionPatient', None), 3) for h in headers]
    keys = None
    if normal is not None and all(pos is not None for pos in positions):
        keys = [float(pos @ normal) for pos in positions]
    else:
        for tag in ['SliceLocation', 'InstanceNumber']:
            values = [finite_vector([getattr(h, tag, None)], 1) for h in headers]
            if all(value is not None for value in values):
                keys = [float(value[0]) for value in values]
                break
    if keys is None:
        if len(paths) != 1:
            raise ValueError('MRI series has no usable slice ordering metadata')
        keys = [0.0]
    order = sorted(range(len(paths)), key=lambda i: (keys[i], paths[i].name))
    spacing = finite_vector(getattr(headers[order[0]], 'PixelSpacing', None), 2)
    return [paths[i] for i in order], None if spacing is None else float(spacing[0])


def slice_indices(n):
    if n < 1:
        raise ValueError('Empty MRI series')
    lo, hi = int(0.35 * (n - 1)), int(0.65 * (n - 1))
    if hi <= lo:
        lo, hi = 0, n - 1
    indices = []
    for anchor in np.linspace(lo, hi, 3).astype(int):
        start = int(np.clip(anchor - 1, 0, max(0, n - 3)))
        indices.extend(range(start, min(start + 3, n)))
    indices.extend([indices[-1]] * (9 - len(indices)))
    return indices[:9]


def render_series(paths):
    paths, spacing = ordered_paths(paths)
    indices = slice_indices(len(paths))
    decoded = {}
    for i in set(indices):
        ds = pydicom.dcmread(paths[i], force=True)
        array = ds.pixel_array.astype(np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError('Expected finite 2D DICOM pixels')
        array = array * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
        if str(getattr(ds, 'PhotometricInterpretation', '')).strip() == 'MONOCHROME1':
            array = array.max() - array
        decoded[i] = array
    volume = np.stack([decoded[i] for i in indices])
    if spacing is not None and np.isfinite(spacing) and spacing > 0:
        size = int(round(130 / spacing))
        h, w = volume.shape[-2:]
        if 16 < size < min(h, w):
            half = size // 2
            volume = volume[:, h//2-half:h//2+half, w//2-half:w//2+half]
    low, high = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
    volume = zoom(volume, (1, 224/volume.shape[1], 224/volume.shape[2]),
                  order=1, mode='nearest', prefilter=False, grid_mode=True)
    return np.clip(np.rint(volume * 255), 0, 255).astype(np.uint8)


def preprocess_study(series_root, frame):
    pixels = np.zeros((6, 9, 224, 224), dtype=np.uint8)
    mask = np.zeros(6, dtype=bool)
    for slot, (plane, fluid) in enumerate(SLOTS):
        candidates = frame.loc[frame.Anatomical_Plane.eq(plane) & frame.Fluid_Sensitive.eq(fluid)]
        choices = []
        for row in candidates.itertuples(index=False):
            paths = sorted((Path(series_root) / row.StudyInstanceUID / row.SeriesInstanceUID).glob('*.dcm'))
            choices.append(paths)
        if not choices:
            continue
        paths = max(choices, key=len)  # stable CSV order breaks ties
        if not paths:
            raise FileNotFoundError('Metadata declares a series but DICOMs are absent')
        pixels[slot] = render_series(paths)
        mask[slot] = True
    if not mask.any():
        raise ValueError('No MRI series for test study')
    return pixels, mask
