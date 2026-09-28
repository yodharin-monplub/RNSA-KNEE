"""Fleet leg v2: model80's final-level rank leg with a per-member reader geometry ('s64' 64 frames @ 336 px,
's128' 128 frames @ 256 px). Builders substitute _M80_W and the _M80_FLEET list. Derived from the submitted
model80 leg by scratch make_fleet_leg_v2.py; tested on CPU with fake readers."""

FLEET_LEG_V2 = '''
# ---- model80: this project's fully trained models as a final-level RANK leg (the d4-blend recipe) ----
# fleet = mean probability of model72 (EfficientNet-B0, 288 px), model71 (ResNet34, 256 px, 5 folds averaged) and
# model73 (CoAtNet-2, 256 px); final = rank((1 - w) * rank(Speedy) + w * rank(fleet)), w fixed in advance.
# Inputs are built exactly as each member was trained: model12's reader at the member's geometry - 's64' = 64 frames
# (18/14/12/8/12) at 336 px, 's128' = 128 frames (32/24/24/24/24) at 256 px; 2-98% span, 140 mm crop.
# Fail-safe: Speedy's submission.csv is already written. It is replaced only after the fleet has predicted the
# studies inside the 9-hour limit; any error or time shortfall keeps Speedy's file.
_M80_W = 0.30
_M80_SAFETY_S = 1800
_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),
              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)], 's64'),
              ('model73', ['model73_best.pt'], 's64')]


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
    GEOMETRIES = {'s64': ([('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8),
                           ('Axial', -1, 12)], 336),
                  's128': ([('Sagittal', 1, 32), ('Sagittal', 0, 24), ('Coronal', 1, 24), ('Coronal', 0, 24),
                            ('Axial', -1, 24)], 256),
                  's96': ([('Sagittal', 1, 24), ('Sagittal', 0, 18), ('Coronal', 1, 18), ('Coronal', 0, 18),
                           ('Axial', -1, 18)], 336)}
    used_geometries = sorted({g for _, _, g in _M80_FLEET})
    assert set(used_geometries) <= set(GEOMETRIES), used_geometries
    sub_path = _P('/kaggle/working/submission.csv')
    pub = _pd.read_csv(sub_path, dtype={'StudyInstanceUID': str})
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

    def read_geometry(uid, geometry):   # model12.inference.read_study, at a member's geometry
        slots, cache_size = GEOMETRIES[geometry]
        frames = sum(count for _, _, count in slots)
        rows = groups.get(uid, [])
        volume = _np.zeros((frames, cache_size, cache_size), dtype=_np.uint8)
        used, offset = set(), 0
        for plane, fluid, count in slots:
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
                    crop = _cv2.resize(array[y:y + size, x:x + size], (cache_size, cache_size),
                                       interpolation=_cv2.INTER_AREA)
                    volume[offset] = (crop * 255).astype(_np.uint8)
                offset += 1
        mask = volume.reshape(frames, -1).sum(1) > 0
        if not mask.any():
            raise ValueError('Study has no usable images')
        return volume, mask

    def read_study(uid):   # every geometry the fleet needs (DICOM ordering is re-read per geometry)
        return {geometry: read_geometry(uid, geometry) for geometry in used_geometries}

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
    for member, files, geometry in _M80_FLEET:
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
            checkpoints.append((member, int(ck['res']), name, ck['arch'], img_size, ck['model'], geometry))
            print(f'[model80] checkpoint {path} sha256={digest.hexdigest()} arch={ck["arch"]} res={ck["res"]} '
                  f'epoch={ck.get("epoch")}', flush=True)
            del ck
    models = {}
    for gpu in gpus:
        models[gpu] = []
        for member, res, name, arch, img_size, state, geometry in checkpoints:
            model = Classifier(arch, img_size)
            model.load_state_dict(state, strict=True)
            models[gpu].append((member, res, geometry, model.eval().to(_torch.device(f'cuda:{gpu}'))))
    del checkpoints
    members = [m for m, _, _ in _M80_FLEET]
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
                    volumes = future.result()
                    by_key, sums = {}, {}
                    for member, res, geometry, model in models[gpu]:
                        if (geometry, res) not in by_key:
                            by_key[(geometry, res)] = windows(*volumes[geometry], res)
                        sums.setdefault(member, []).append(predict(model, by_key[(geometry, res)], device))
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
    r_pub = pub[labels].rank(method='average', pct=True).to_numpy(_np.float64)
    r_fleet = r_pub.copy()   # a study the fleet could not read keeps its Speedy rank
    r_fleet[ok] = _pd.DataFrame(fleet[ok]).rank(method='average', pct=True).to_numpy(_np.float64)
    mixed = (1.0 - _M80_W) * r_pub + _M80_W * r_fleet
    out = pub.copy()
    out[labels] = _pd.DataFrame(mixed, columns=labels).rank(method='average', pct=True).to_numpy(_np.float64)
    values = out[labels].to_numpy(_np.float64)
    if not _np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError('fleet blend produced invalid values')
    rho = float(_np.mean([_pd.Series(r_pub[ok, j]).corr(_pd.Series(r_fleet[ok, j])) for j in range(len(labels))]))
    _np.savez_compressed('/kaggle/working/model80_fleet.npz', study_uids=_np.asarray(uids), labels=_np.asarray(LABELS),
                         **{m: member_probs[m] for m in members})
    _P('/kaggle/working/submission_speedy.csv').write_bytes(sub_path.read_bytes())
    tmp = sub_path.with_name('.submission_model80.csv')
    out.to_csv(tmp, index=False)
    _o.replace(tmp, sub_path)
    print(f'[model80] fleet leg blended at w={_M80_W}: {int(ok.sum())}/{n} studies read by the fleet, '
          f'{len(failed)} failed (kept Speedy rank); members {members}; geometries {used_geometries}; '
          f'{seconds:.0f}s on {len(gpus)} GPUs; '
          f'mean per-label rank corr Speedy vs fleet {rho:.3f}; rewrote submission.csv', flush=True)


try:
    _m80_fleet_leg()
except Exception as _m80_error:
    print(f'[model80] fleet leg FAILED ({type(_m80_error).__name__}: {str(_m80_error)[:300]}); '
          f'Speedy submission.csv kept', flush=True)
'''
