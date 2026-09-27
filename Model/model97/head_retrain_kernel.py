"""CoAtNet inference compatible with Dread Development's public knee checkpoint.

Architecture and geometry credit: Johnathan Wagner / Dread Development.
Source: https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model
Model12: max-span checkpoint with documented 2%-98% sampling.
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
        n = len(records); lo = int(n * .02); hi = max(lo, int(n * .98)-1)
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



ARM = 'model75'
WEIGHT_FILE = 'model75_best.pt'
KNOWN_GOLD = 0.920393
TARGETS_FILE = 'targets_g38f_low.csv'
OUTPUT_PREFIX = 'model97'
MAX_WINDOWS = 62


def extract_features(model, device, inputs, size):
    import hashlib
    manifests = sorted(p for p in inputs.rglob('cache_manifest.json'))
    assert len(manifests) == 4, manifests
    ids, feats, masks = [], [], []
    for path in manifests:
        manifest = json.loads(path.read_text())
        print('cache geometry', manifest['geometry'], flush=True)
        for shard in manifest['shards']:
            shard_path = path.parent / shard['file']
            with shard_path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == shard['sha256']
            with np.load(shard_path, allow_pickle=False) as archive:
                for i, uid in enumerate(archive['ids'].tolist()):
                    images = windows(archive['volumes'][i], archive['masks'][i], size)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        encoded = torch.cat([model.backbone(chunk.to(device)) for chunk in images.split(12)])
                    n = min(len(encoded), MAX_WINDOWS)
                    block = np.zeros((MAX_WINDOWS, encoded.shape[1]), np.float16)
                    block[:n] = encoded[:n].float().cpu().numpy().astype(np.float16)
                    window_mask = np.zeros(MAX_WINDOWS, np.float32)
                    window_mask[:n] = 1
                    ids.append(uid); feats.append(block); masks.append(window_mask)
        print('features', len(ids), flush=True)
    return ids, np.stack(feats), np.stack(masks)


class Head(nn.Module):
    """model12's prediction head (norm + attention pooling + per-label linear), trainable alone."""
    def __init__(self, dim, state):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, dim))
        self.clsb = nn.Parameter(torch.zeros(12))
        self.load_state_dict(state, strict=True)

    def forward(self, features, mask):
        h = self.norm(features.float())
        attention = self.att(h).masked_fill(mask.unsqueeze(-1) < .5, -1e4).softmax(dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def main():
    from sklearn.metrics import roc_auc_score
    started = time.time()
    # Head training on cached features is small enough for a CPU kernel (used when the account's two GPU batch
    # slots are taken); only the feature extraction needs the GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = Path('/kaggle/input')
    out = Path('/kaggle/working')
    comp = next(p.parent for p in inputs.rglob('train.csv') if (p.parent / 'test_series.csv').exists())
    weights = list(inputs.rglob(WEIGHT_FILE))
    assert len(weights) == 1, weights
    ck = torch.load(weights[0], map_location='cpu', weights_only=True)
    model = Classifier(ck['arch']); model.load_state_dict(ck['model'], strict=True)
    size = int(ck['res'])
    head_state = {k: v for k, v in ck['model'].items() if not k.startswith('backbone.')}
    cached = list(inputs.rglob('window_features_fp16.npz'))
    if cached:
        with np.load(cached[0], allow_pickle=False) as archive:
            ids, feats, masks = archive['ids'].tolist(), archive['features'], archive['masks']
        print('reused features from', cached[0], flush=True)
    else:
        assert device.type == 'cuda', 'feature extraction needs a GPU kernel'
        model = model.to(device).eval()
        ids, feats, masks = extract_features(model, device, inputs, size)
        np.savez(out / 'window_features_fp16.npz', ids=np.asarray(ids), features=feats, masks=masks)
        model = model.cpu()
    torch.cuda.empty_cache()
    print('feature array', feats.shape, round(time.time() - started), flush=True)

    target_paths = list(inputs.rglob(TARGETS_FILE))
    assert len(target_paths) == 1, target_paths
    labels = pd.read_csv(target_paths[0]).set_index(UID)
    assert labels.index.is_unique and np.isfinite(labels[LABELS].to_numpy(float)).all()
    partition = pd.read_csv(next(inputs.rglob('partitions.csv'))).set_index(UID)
    gold = pd.read_csv(comp / 'train.csv').set_index(UID)
    position = {uid: i for i, uid in enumerate(ids)}
    gold_ids = sorted(u for u in partition.index[partition.split == 'gold'] if u in position)
    weak_ids = [u for u in partition.index if u in position and u not in set(gold_ids)]
    assert len(gold_ids) == 58 and len(weak_ids) > 4000
    y_weak = torch.from_numpy(labels.loc[weak_ids, LABELS].to_numpy(np.float32))
    y_gold = gold.loc[gold_ids, LABELS].to_numpy(np.float32)
    weak_index = np.asarray([position[u] for u in weak_ids])
    gold_index = np.asarray([position[u] for u in gold_ids])
    feats_t, masks_t = torch.from_numpy(feats), torch.from_numpy(masks)
    macro = lambda p: float(np.mean([roc_auc_score(y_gold[:, j], p[:, j]) for j in range(12)]))

    def predict(head_module):
        head_module.eval()
        with torch.no_grad():
            return torch.sigmoid(head_module(feats_t[gold_index].to(device), masks_t[gold_index].to(device))).cpu().numpy()

    base = macro(predict(Head(feats.shape[-1], head_state).to(device)))
    report = {'arm': ARM, 'own_head_on_cached_features': base, 'known_gold_auc': KNOWN_GOLD}
    print(json.dumps(report), flush=True)
    assert abs(base - KNOWN_GOLD) < 0.002, 'cached features do not reproduce the arm'

    def train_head(seed, epochs=6):
        torch.manual_seed(seed)
        head_module = Head(feats.shape[-1], head_state).to(device)
        optimizer = torch.optim.AdamW(head_module.parameters(), lr=3e-4, weight_decay=2e-3)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(weak_index))
            head_module.train()
            for start in range(0, len(order), 128):
                chunk = order[start:start + 128]
                logits = head_module(feats_t[weak_index[chunk]].to(device), masks_t[weak_index[chunk]].to(device))
                loss = F.binary_cross_entropy_with_logits(logits, y_weak[chunk].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head_module.parameters(), 2.0)
                optimizer.step()
        return head_module

    seed_predictions = []
    for seed in (0, 1, 2):
        seed_predictions.append(predict(train_head(seed)))
        report[f'seed{seed}_gold_auc'] = macro(seed_predictions[-1])
        print(json.dumps({f'seed{seed}_gold_auc': report[f'seed{seed}_gold_auc']}), flush=True)
    report['seed_mean'] = float(np.mean([report[f'seed{s}_gold_auc'] for s in (0, 1, 2)]))
    final = train_head(7)
    final_prediction = predict(final)
    report['final_head_gold_auc'] = macro(final_prediction)
    report['final_per_label'] = {c: float(roc_auc_score(y_gold[:, j], final_prediction[:, j])) for j, c in enumerate(LABELS)}
    pd.DataFrame(final_prediction, index=gold_ids, columns=LABELS).rename_axis(UID).to_csv(out / f'gold_predictions_{ARM}.csv')

    # Full checkpoint in the arm's own layout, head replaced; verify it end to end through Classifier on gold features.
    new_state = dict(ck['model'])
    for key, value in final.state_dict().items():
        assert key in new_state and new_state[key].shape == value.shape, key
        new_state[key] = value.detach().cpu().to(new_state[key].dtype)
    checkpoint = dict(ck); checkpoint['model'] = new_state
    checkpoint['head_retrain'] = {'source': WEIGHT_FILE, 'targets': TARGETS_FILE,
                                  'recipe': 'plain BCE, lr 3e-4, wd 2e-3, bs 128, 6 epochs, seed 7'}
    report['targets'] = TARGETS_FILE
    torch.save(checkpoint, out / f'{OUTPUT_PREFIX}_{ARM}_best.pt')
    verify = Classifier(ck['arch']); verify.load_state_dict(new_state, strict=True); verify = verify.to(device).eval()
    with torch.no_grad():
        f = feats_t[gold_index[:8]].to(device).float(); m = masks_t[gold_index[:8]]
        rows = [torch.sigmoid(verify.predict_features(f[i:i + 1, :int(m[i].sum())])).cpu().numpy()[0] for i in range(8)]
    report['checkpoint_max_abs_diff'] = float(np.abs(np.stack(rows) - final_prediction[:8]).max())
    assert report['checkpoint_max_abs_diff'] < 1e-4, report['checkpoint_max_abs_diff']
    report['elapsed_seconds'] = time.time() - started
    (out / f'head_retrain_{ARM}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


main()
