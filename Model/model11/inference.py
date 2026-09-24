"""CoAtNet inference compatible with Dread Development's public knee checkpoint.

Architecture and geometry credit: Johnathan Wagner / Dread Development.
Source: https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model
Model11: fixed probability blend of wide-dense and fine-spacing checkpoints.
This implementation uses only images and public series metadata.
"""
import gc
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
from pydicom.pixels import apply_modality_lut
from torch import nn
from torch.nn import functional as F

UID = 'StudyInstanceUID'
SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12),
         ('Coronal', 0, 8), ('Axial', -1, 12)]
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


class Classifier(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0,
                                          in_chans=3, global_pool='avg')
        dim = self.backbone.num_features
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, dim))
        self.clsb = nn.Parameter(torch.zeros(12))

    def predict_features(self, features):
        features = self.norm(features)
        attention = self.att(features).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, features)
        return (pooled * self.clsW).sum(-1) + self.clsb


def ordered_files(directory):
    records = []
    for path in directory.glob('*.dcm'):
        header = pydicom.dcmread(path, stop_before_pixels=True)
        orientation = getattr(header, 'ImageOrientationPatient', None)
        position = getattr(header, 'ImagePositionPatient', None)
        if orientation is not None and position is not None and len(orientation) == 6:
            normal = np.cross(np.asarray(orientation[:3], float), np.asarray(orientation[3:], float))
            order = float(np.dot(np.asarray(position, float), normal))
        else:
            order = float(getattr(header, 'InstanceNumber', 0) or 0)
        spacing = getattr(header, 'PixelSpacing', None)
        records.append((order, path, float(spacing[0]) if spacing is not None else .5))
    records.sort(key=lambda x: x[0])
    return records


def read_study(uid, rows, series_root):
    volume = np.zeros((64, 336, 336), dtype=np.uint8)
    used, offset, failed = set(), 0, 0
    for plane, fluid, count in SLOTS:
        candidates = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
        preferred = [r for r in candidates if int(r['Fluid_Sensitive']) == fluid] if fluid in (0,1) else []
        chosen = (preferred or candidates)
        if not chosen:
            offset += count
            continue
        row = chosen[0]; used.add(row['SeriesInstanceUID'])
        records = ordered_files(series_root / uid / row['SeriesInstanceUID'])
        if not records:
            raise FileNotFoundError('Declared series has no DICOM files')
        n = len(records); lo = int(n * .06); hi = max(lo, int(n * .94)-1)
        indices = np.rint(np.linspace(lo, hi, count)).astype(int) if n > 1 else np.zeros(count, dtype=int)
        selected = []
        for i in indices:
            _, path, spacing = records[min(i, n-1)]
            try:
                ds = pydicom.dcmread(path)
                array = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
                if array.ndim != 2 or not np.isfinite(array).all():
                    raise ValueError('Invalid decoded MRI pixels')
                if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
                    array = array.max() - array
                selected.append((array, spacing))
            except (ValueError, RuntimeError, NotImplementedError):
                selected.append((None, spacing)); failed += 1
        valid = [a for a, _ in selected if a is not None]
        if not valid:
            raise ValueError('No decodable slices in selected series')
        low, high = np.percentile(np.concatenate([a.ravel() for a in valid]), [2, 98])
        median_spacing = float(np.median([r[2] for r in records]))
        for array, spacing in selected:
            if array is not None:
                array = np.clip((array-low)/(high-low+1e-6), 0, 1)
                spacing = spacing if spacing > 0 else median_spacing
                h, w = array.shape
                size = min(int(round(140/max(spacing, 1e-3))), h, w)
                y, x = (h-size)//2, (w-size)//2
                crop = cv2.resize(array[y:y+size, x:x+size], (336,336), interpolation=cv2.INTER_AREA)
                volume[offset] = (crop * 255).astype(np.uint8)
            offset += 1
    mask = volume.reshape(64,-1).sum(1) > 0
    if not mask.any():
        raise ValueError('Study has no usable images')
    return volume, mask, failed


def windows(volume, mask, size):
    valid = np.flatnonzero(mask)
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    centers = list(range(lo+1, hi)) or [max(1, min((lo+hi)//2, len(volume)-2))]
    indices = np.arange(len(centers))
    slices = np.stack([volume[centers[i]-1:centers[i]+2] for i in indices]).astype(np.float32) / 255
    images = F.interpolate(torch.from_numpy(slices), size=(size,size), mode='bilinear', align_corners=False)
    return (images - torch.tensor([.485,.456,.406]).view(1,3,1,1)) / torch.tensor([.229,.224,.225]).view(1,3,1,1)


@torch.inference_mode()
def predict(model, images, device):
    encoded = []
    with torch.autocast(device.type, enabled=device.type=='cuda', dtype=torch.float16):
        for batch in images.split(12):
            encoded.append(model.backbone(batch.to(device)))
        logits = model.predict_features(torch.cat(encoded).unsqueeze(0))
    probabilities = logits.float().sigmoid().cpu().numpy()[0]
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite model predictions')
    return probabilities


BRANCHES = [
    ('model8', 'raptor-knee-widedense', 'raptor_ft_coatnet_v4_full.pt', .25),
    ('model9', 'raptor-knee-widedense', 'raptor_ft_coatnet_v4_full_swa.pt', .25),
    ('model10', 'raptor-knee-finespacing', 'raptor_ft_coatnet_v9_full.pt', .50),
]


def main():
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = Path('/kaggle/input')
    roots = [p for p in [inputs/'competitions/rsna-knee-abnormality-detection',
                         inputs/'rsna-knee-abnormality-detection'] if (p/'test.csv').exists()]
    assert len(roots) == 1
    root = roots[0]
    loaded = []
    for name, dataset, filename, weight in BRANCHES:
        candidates = [inputs/dataset/filename]
        candidates += list(inputs.glob(f'datasets/dreaddevelopment/{dataset}/{filename}'))
        candidates += list(inputs.glob(f'datasets/dreaddevelopment/{dataset}/*/{filename}'))
        files = [p for p in candidates if p.is_file()]
        assert len(files) == 1, files
        ck = torch.load(files[0], map_location='cpu', weights_only=True)
        assert list(ck.get('lab', LABELS)) == LABELS
        model = Classifier(ck['arch'])
        model.load_state_dict(ck['model'], strict=True)
        assert int(ck['res']) == 384
        del ck
        loaded.append((name, model.to(device).eval(), weight))
        print('Loaded', name, flush=True)
    gc.collect()
    test = pd.read_csv(root/'test.csv')
    series = pd.read_csv(root/'test_series.csv')
    sample = pd.read_csv(root/'sample_submission.csv')
    assert test[UID].is_unique and sample[UID].is_unique
    groups = {uid: frame.to_dict('records') for uid, frame in series.groupby(UID)}
    branch_results = {name: [] for name, _, _ in loaded}
    timings = {name: 0.0 for name, _, _ in loaded}
    decode_seconds = 0.0
    for i, uid in enumerate(test[UID]):
        tick = time.time()
        volume, mask, failed = read_study(uid, groups[uid], root/'test_series')
        images = windows(volume, mask, 384)
        decode_seconds += time.time() - tick
        for name, model, _ in loaded:
            tick = time.time()
            branch_results[name].append(predict(model, images, device))
            timings[name] += time.time() - tick
        if failed:
            print(json.dumps({'study': i, 'failed_slices': failed}), flush=True)
        if i % 50 == 0 or i+1 == len(test):
            print('Predicted', i+1, '/', len(test), 'seconds', round(time.time()-started, 1), flush=True)
    result = np.zeros((len(test), len(LABELS)), dtype=np.float64)
    output = Path('/kaggle/working')
    for name, _, weight in loaded:
        array = np.asarray(branch_results[name])
        result += weight * array
        branch = pd.DataFrame(array, columns=LABELS)
        branch.insert(0, UID, test[UID].to_numpy())
        branch.to_csv(output/f'{name}_predictions.csv', index=False)
    frame = pd.DataFrame(result, columns=LABELS)
    frame.insert(0, UID, test[UID].to_numpy())
    submission = sample[[UID]].merge(frame, on=UID, validate='one_to_one')[sample.columns]
    values = submission[LABELS].to_numpy()
    assert len(submission) == len(test) and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    submission.to_csv(output/'submission.csv', index=False)
    runtime = {'studies': len(test), 'elapsed_seconds': time.time()-started,
               'decode_seconds': decode_seconds, 'branch_seconds': timings,
               'peak_gpu_gb': torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else None,
               'weights': {name: weight for name, _, weight in loaded}}
    (output/'runtime.json').write_text(json.dumps(runtime, indent=2))
    print(json.dumps(runtime), flush=True)


if __name__ == '__main__':
    main()
