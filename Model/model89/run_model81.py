"""model81: this project's fully trained fleet ALONE as a submission - a leaderboard calibration for the fleet.

fleet = mean probability of model72 (EfficientNet-B0 288 px, gold 0.9143), model71 (ResNet34 256 px, 5 folds,
0.9138) and model73 (CoAtNet-2 256 px, 0.9122); fleet gold AUC 0.9258 (58 cases). Forum teams report 5-fold
EfficientNet-B0 ensembles at 0.94+ public; model81 measures where ours stands, which sets how much weight the fleet
deserves in the final blend (model80 blends it into the public Speedy pipeline at w = 0.30).
Same code as model80's fleet leg (model12 reader geometry, both GPUs, strict loads with SHA-256), but the fleet's
probabilities are the submission; a study the fleet cannot read gets the fleet mean (neutral fill). No fallback: a
failure fails the run. Weights: private dataset yodharinmonplub/rsna-knee-fleet-weights.

Usage: python scripts/run_model81.py [--push] [--name modelNN --title T --fleet "[('m', ['m_best.pt']), ...]"]
"""
import argparse
import ast
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org', ''))
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
NAME = 'model81'
TITLE = 'RSNA Knee Model81 Fleet Only'
# Speedy's pinned image: the one model77/model80 already ran the same timm checkpoints on.
DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
MODEL_DIR = ROOT.parent / 'Model' / NAME
FLEET_SRC = "_M80_FLEET = [('model72', ['model72_best.pt']),\n              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)]),\n              ('model73', ['model73_best.pt'])]"

CELL = '''# ---- model81: this project's fully trained fleet ALONE (no public models) - a leaderboard calibration ----
# fleet = mean probability of model72 (EfficientNet-B0, 288 px), model71 (ResNet34, 256 px, 5 folds averaged) and
# model73 (CoAtNet-2, 256 px), read with this project's own reader (model12 geometry: 64 frames, 2-98% span, 140 mm,
# 336 px). Same code as model80's fleet leg; here the fleet's probabilities ARE the submission. A study the fleet
# cannot read gets the fleet's mean probability over the readable studies (neutral fill).
import glob as _m81_glob
import os as _m81_os
_M80_SAFETY_S = 1800
_M80_FLEET = [('model72', ['model72_best.pt']),
              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)]),
              ('model73', ['model73_best.pt'])]
_m81_roots = [p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                          '/kaggle/input/rsna-knee-abnormality-detection') if _m81_os.path.exists(p + '/test.csv')]
assert len(_m81_roots) == 1, _m81_roots
_KE_NS = {'find_test_root': lambda: _m81_roots[0]}


def _m80_fleet_leg():
    import glob as _g
    import hashlib as _h
    import os as _o
    import threading as _th
    import time as _t
    from concurrent.futures import ThreadPoolExecutor as _Pool
    from pathlib import Path as _P
    import cv2 as _cv2
    import numpy as _np
    import pandas as _pd
    import pydicom as _dcm
    import timm as _timm
    import torch as _torch
    from torch import nn as _nn
    from torch.nn import functional as _F
    try:
        from pydicom.pixels import apply_modality_lut as _lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut as _lut
    started = _t.time()
    uptime = float(open('/proc/uptime').read().split()[0])
    start_ticks = int(open('/proc/self/stat').read().split(')')[-1].split()[19])
    age = uptime - start_ticks / _o.sysconf('SC_CLK_TCK')
    deadline = started + 9 * 3600 - age - _M80_SAFETY_S
    print(f'[model80] fleet leg starts at notebook age {age / 3600:.2f} h; must finish within '
          f'{(deadline - started) / 60:.0f} min', flush=True)
    if deadline - started < 600:
        raise TimeoutError('less than 10 minutes left for the fleet leg')
    LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
              'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
    SLOTS = [('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)]
    sub_path = _P('/kaggle/working/submission.csv')
    pub = _pd.read_csv(_P(_KE_NS['find_test_root']()) / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
    labels = [c for c in pub.columns if c != 'StudyInstanceUID']
    if sorted(labels) != sorted(LABELS) or pub.StudyInstanceUID.duplicated().any():
        raise RuntimeError(f'unexpected submission schema {pub.columns.tolist()}')
    uids = pub.StudyInstanceUID.tolist()
    root = _P(_KE_NS['find_test_root']())
    series_root = root / 'test_series' if (root / 'test_series').is_dir() else root / 'test_images'
    series = _pd.read_csv(root / 'test_series.csv')
    series['StudyInstanceUID'] = series['StudyInstanceUID'].astype(str)
    series['SeriesInstanceUID'] = series['SeriesInstanceUID'].astype(str)
    groups = {uid: frame.to_dict('records') for uid, frame in series.groupby('StudyInstanceUID')}

    class Classifier(_nn.Module):   # model12.inference.Classifier
        def __init__(self, arch, img_size=None):
            super().__init__()
            extra = {'img_size': img_size} if img_size else {}
            self.backbone = _timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3,
                                               global_pool='avg', **extra)
            dim = self.backbone.num_features
            self.norm = _nn.LayerNorm(dim)
            self.att = _nn.Sequential(_nn.Linear(dim, 256), _nn.Tanh(), _nn.Dropout(.2), _nn.Linear(256, 12))
            self.clsW = _nn.Parameter(_torch.zeros(12, dim))
            self.clsb = _nn.Parameter(_torch.zeros(12))

        def predict_features(self, features):
            features = self.norm(features)
            attention = self.att(features).softmax(dim=1)
            pooled = _torch.einsum('bkn,bkf->bnf', attention, features)
            return (pooled * self.clsW).sum(-1) + self.clsb

    def ordered_files(directory):   # model12.inference.ordered_files
        records = []
        for path in directory.glob('*.dcm'):
            header = _dcm.dcmread(path, stop_before_pixels=True)
            orientation = getattr(header, 'ImageOrientationPatient', None)
            position = getattr(header, 'ImagePositionPatient', None)
            if orientation is not None and position is not None and len(orientation) == 6:
                normal = _np.cross(_np.asarray(orientation[:3], float), _np.asarray(orientation[3:], float))
                order = float(_np.dot(_np.asarray(position, float), normal))
            else:
                order = float(getattr(header, 'InstanceNumber', 0) or 0)
            spacing = getattr(header, 'PixelSpacing', None)
            records.append((order, path, float(spacing[0]) if spacing is not None else .5))
        records.sort(key=lambda x: x[0])
        return records

    def read_study(uid):   # model12.inference.read_study
        rows = groups.get(uid, [])
        volume = _np.zeros((64, 336, 336), dtype=_np.uint8)
        used, offset = set(), 0
        for plane, fluid, count in SLOTS:
            candidates = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
            preferred = [r for r in candidates if int(r['Fluid_Sensitive']) == fluid] if fluid in (0, 1) else []
            chosen = (preferred or candidates)
            if not chosen:
                offset += count
                continue
            row = chosen[0]; used.add(row['SeriesInstanceUID'])
            records = ordered_files(series_root / uid / row['SeriesInstanceUID'])
            if not records:
                raise FileNotFoundError('Declared series has no DICOM files')
            n = len(records); lo = int(n * .02); hi = max(lo, int(n * .98) - 1)
            indices = _np.rint(_np.linspace(lo, hi, count)).astype(int) if n > 1 else _np.zeros(count, dtype=int)
            selected = []
            for i in indices:
                _, path, spacing = records[min(i, n - 1)]
                try:
                    ds = _dcm.dcmread(path)
                    array = _lut(ds.pixel_array, ds).astype(_np.float32)
                    if array.ndim != 2 or not _np.isfinite(array).all():
                        raise ValueError('Invalid decoded MRI pixels')
                    if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
                        array = array.max() - array
                    selected.append((array, spacing))
                except (ValueError, RuntimeError, NotImplementedError):
                    selected.append((None, spacing))
            valid = [a for a, _ in selected if a is not None]
            if not valid:
                raise ValueError('No decodable slices in selected series')
            low, high = _np.percentile(_np.concatenate([a.ravel() for a in valid]), [2, 98])
            median_spacing = float(_np.median([r[2] for r in records]))
            for array, spacing in selected:
                if array is not None:
                    array = _np.clip((array - low) / (high - low + 1e-6), 0, 1)
                    spacing = spacing if spacing > 0 else median_spacing
                    h, w = array.shape
                    size = min(int(round(140 / max(spacing, 1e-3))), h, w)
                    y, x = (h - size) // 2, (w - size) // 2
                    crop = _cv2.resize(array[y:y + size, x:x + size], (336, 336), interpolation=_cv2.INTER_AREA)
                    volume[offset] = (crop * 255).astype(_np.uint8)
                offset += 1
        mask = volume.reshape(64, -1).sum(1) > 0
        if not mask.any():
            raise ValueError('Study has no usable images')
        return volume, mask

    def windows(volume, mask, size):   # model12.inference.windows
        valid = _np.flatnonzero(mask)
        if len(valid) < 3:
            valid = _np.arange(3)
        lo, hi = int(valid.min()), int(valid.max())
        centers = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, len(volume) - 2))]
        slices = _np.stack([volume[c - 1:c + 2] for c in centers]).astype(_np.float32) / 255
        images = _F.interpolate(_torch.from_numpy(slices), size=(size, size), mode='bilinear', align_corners=False)
        return (images - _torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / _torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)

    def find(name):
        hits = sorted({_o.path.realpath(p) for pattern in (f'/kaggle/input/datasets/*/rsna-knee-fleet-weights/{name}',
                                         f'/kaggle/input/rsna-knee-fleet-weights/{name}',
                                         f'/kaggle/input/*/rsna-knee-fleet-weights/{name}',
                                         f'/kaggle/input/*/*/rsna-knee-fleet-weights/{name}')
                       for p in _g.glob(pattern)})
        if len(hits) != 1:
            raise RuntimeError(f'model80: expected one {name}, found {hits}')
        return hits[0]

    for gpu in range(_torch.cuda.device_count()):
        with _torch.cuda.device(gpu):
            _torch.cuda.empty_cache()
    gpus = list(range(min(2, _torch.cuda.device_count())))
    if not gpus:
        raise RuntimeError('no GPU')
    checkpoints = []   # (member, res, file, arch, img_size, state)
    for member, files in _M80_FLEET:
        for name in files:
            path = find(name)
            digest = _h.sha256()
            with open(path, 'rb') as handle:
                for block in iter(lambda: handle.read(8 << 20), b''):
                    digest.update(block)
            ck = _torch.load(path, map_location='cpu', weights_only=False)
            if list(ck.get('lab', LABELS)) != LABELS:
                raise RuntimeError(f'{name}: label order differs')
            img_size = (ck.get('training_config') or {}).get('backbone_img_size')
            checkpoints.append((member, int(ck['res']), name, ck['arch'], img_size, ck['model']))
            print(f'[model80] checkpoint {path} sha256={digest.hexdigest()} arch={ck["arch"]} res={ck["res"]} '
                  f'epoch={ck.get("epoch")}', flush=True)
            del ck
    models = {}
    for gpu in gpus:
        models[gpu] = []
        for member, res, name, arch, img_size, state in checkpoints:
            model = Classifier(arch, img_size)
            model.load_state_dict(state, strict=True)
            models[gpu].append((member, res, model.eval().to(_torch.device(f'cuda:{gpu}'))))
    del checkpoints
    members = [m for m, _ in _M80_FLEET]
    n = len(uids)
    member_probs = {m: _np.full((n, 12), _np.nan, _np.float32) for m in members}
    failed, stop, done = [], _th.Event(), [0]
    lock = _th.Lock()

    @_torch.inference_mode()
    def predict(model, images, device):   # model12.inference.predict
        encoded = []
        with _torch.autocast('cuda', dtype=_torch.float16):
            for batch in images.split(12):
                encoded.append(model.backbone(batch.to(device)))
            logits = model.predict_features(_torch.cat(encoded).unsqueeze(0))
        probabilities = logits.float().sigmoid().cpu().numpy()[0]
        if not _np.isfinite(probabilities).all():
            raise ValueError('Nonfinite model predictions')
        return probabilities

    def run_gpu(gpu):
        device = _torch.device(f'cuda:{gpu}')
        order = list(range(gpu, n, len(gpus)))
        with _Pool(max_workers=2) as pool, _torch.cuda.device(device):
            pending = {j: pool.submit(read_study, uids[order[j]]) for j in range(min(3, len(order)))}
            for j, i in enumerate(order):
                if stop.is_set():
                    break
                future = pending.pop(j)
                if j + 3 < len(order):
                    pending[j + 3] = pool.submit(read_study, uids[order[j + 3]])
                try:
                    volume, mask = future.result()
                    by_res, sums = {}, {}
                    for member, res, model in models[gpu]:
                        if res not in by_res:
                            by_res[res] = windows(volume, mask, res)
                        sums.setdefault(member, []).append(predict(model, by_res[res], device))
                    for member, values in sums.items():
                        member_probs[member][i] = _np.mean(values, axis=0)
                except Exception as error:
                    with lock:
                        failed.append(f'{uids[i]}: {type(error).__name__}: {str(error)[:200]}')
                with lock:
                    done[0] += 1
                    count = done[0]
                now = _t.time()
                if now > deadline:
                    stop.set()
                elif count >= 40 and now + (now - started) / count * (n - count) > deadline:
                    stop.set()   # the projected finish is past the limit: give up early, keep Speedy
            for future in pending.values():
                future.cancel()

    with _Pool(max_workers=len(gpus)) as pool:
        for job in [pool.submit(run_gpu, gpu) for gpu in gpus]:
            job.result()
    del models
    seconds = _t.time() - started
    if stop.is_set():
        raise TimeoutError(f'fleet leg stopped by the time guard after {done[0]}/{n} studies ({seconds:.0f}s)')
    fleet = _np.mean([member_probs[m] for m in members], axis=0)
    ok = _np.isfinite(fleet).all(axis=1)
    if ok.sum() < max(1, int(0.97 * n)):
        raise RuntimeError(f'fleet leg read only {int(ok.sum())}/{n} studies; first failures {failed[:3]}')
    order = [LABELS.index(c) for c in labels]
    fleet = fleet[:, order]
    fill = _np.nanmean(_np.where(ok[:, None], fleet, _np.nan), axis=0)
    fleet[~ok] = fill
    out = pub[['StudyInstanceUID']].copy()
    for j, c in enumerate(labels):
        out[c] = fleet[:, j]
    values = out[labels].to_numpy(_np.float64)
    if not _np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError('fleet produced invalid values')
    _np.savez_compressed('/kaggle/working/model81_fleet.npz', study_uids=_np.asarray(uids), labels=_np.asarray(LABELS),
                         **{m: member_probs[m] for m in members})
    tmp = sub_path.with_name('.submission_model81.csv')
    out.to_csv(tmp, index=False)
    _o.replace(tmp, sub_path)
    print(f'[model81] fleet submission written: {int(ok.sum())}/{n} studies read by the fleet, {len(failed)} failed '
          f'(neutral fill); members {members}; {seconds:.0f}s on {len(gpus)} GPUs', flush=True)


_m80_fleet_leg()
'''


def build(name=NAME, title=TITLE, fleet=None):
    """fleet: optional replacement for the member list, e.g. "[('model84', ['model84_best.pt']), ...]"; every file must
    be in the fleet dataset."""
    cell = CELL
    if fleet:
        members = ast.literal_eval(fleet)
        assert all(isinstance(m, str) and all(f.endswith('.pt') for f in files) for m, files in members)
        cell = cell.replace(FLEET_SRC, '_M80_FLEET = ' + repr(members))
        assert repr(members) in cell
    nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                     'source': cell.splitlines(keepends=True)}],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 4}
    meta = {'id': f'yodharinmonplub/{title.lower().replace(" ", "-")}', 'title': title, 'code_file': 'notebook.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
            'enable_tpu': False, 'enable_internet': False, 'keywords': ['gpu'],
            'dataset_sources': ['yodharinmonplub/rsna-knee-fleet-weights'], 'kernel_sources': [],
            'competition_sources': ['rsna-knee-abnormality-detection'], 'model_sources': [],
            'docker_image': DOCKER, 'machine_shape': 'NvidiaTeslaT4'}
    compile(cell, name, 'exec')
    folder = ROOT / f'artifacts/{name}_notebook'
    model_dir = ROOT.parent / 'Model' / name
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    shutil.copy2(Path(__file__).resolve(), model_dir / 'run_model81.py')
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--name', default=NAME)
    parser.add_argument('--title', default=TITLE)
    parser.add_argument('--fleet', help='replacement member list as a Python literal')
    args = parser.parse_args()
    assert len(args.title) <= 50, 'Kaggle caps titles at 50 characters'
    folder, meta = build(args.name, args.title, args.fleet)
    print('built', folder, '| kernel', meta['id'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta['machine_shape'])
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
