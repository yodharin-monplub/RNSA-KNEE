"""CPU test: fleet leg v6 with a ConvNeXt-S member (model140/141 architecture), checkpoint in the trainer format with
the pretrained timm backbone and a random head; harness from test_fleet_leg_v6.py. Original notes: CPU tests of fleet leg v6 (scripts/fleet_leg_v6.py = v5 + partial blend on a time shortfall); harness from
test_fleet_leg_v4.py (v3 notes: (A) the v2 checks with real checkpoints + real windows/predict/blend and a fake reader that
honours the requested geometries; (B) missing checkpoint -> FAILED, file untouched; (C) deadline passed -> Speedy kept;
(D) priority dropping: slow fake members and a tight budget -> the last member is dropped, the rest still blend;
(E) budget too small even for the first member -> leg abandoned, Speedy kept)."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

OTHER = Path(r'C:\Users\yodha\OneDrive\resources hp\coding\RSNA Knee Abnormality Detection\Other')
sys.path.insert(0, str(OTHER / 'scripts'))
from fleet_leg_v3 import FLEET_LEG_V3
from fleet_leg_v4 import FLEET_LEG_V4
from fleet_leg_v5 import FLEET_LEG_V5
from fleet_leg_v6 import FLEET_LEG_V6

code3 = FLEET_LEG_V3.lstrip()
code4 = FLEET_LEG_V4.lstrip()
code5 = FLEET_LEG_V5.lstrip()
code6 = FLEET_LEG_V6.lstrip()
code = code3
code = code.replace("_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),", "_M80_FLEET = [('model72', ['model72_best.pt'], 's128'),")
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
       'Synovitis', "Baker's", 'Contusion', 'Fracture']

tmp = Path(tempfile.mkdtemp(prefix='m80cnx_'))
weights = tmp / 'input/datasets/me/rsna-knee-fleet-weights'
weights.mkdir(parents=True)
src = {'model72_best.pt': 'artifacts/remote_model72/model72/best.pt', 'model73_best.pt': 'artifacts/remote_model73/model73/best.pt'}
src.update({f'model71_fold{k}_best.pt': f'artifacts/remote_model71/model71/fold{k}/best.pt' for k in range(5)})
for name, rel in src.items():
    os.link(OTHER / rel, weights / name)
(tmp / 'working').mkdir()
comp = tmp / 'comp'; comp.mkdir()
requested = []
FAST = [False]


def setup(n):
    uids = [f'1.2.3.{i}' for i in range(n)]
    pd.DataFrame({'StudyInstanceUID': uids * 2, 'SeriesInstanceUID': [u + '.1' for u in uids] + [u + '.2' for u in uids],
                  'Anatomical_Plane': ['Sagittal'] * n + ['Coronal'] * n, 'Fluid_Sensitive': [1] * (2 * n)}).to_csv(comp / 'test_series.csv', index=False)
    rng = np.random.default_rng(0)
    speedy = pd.DataFrame(rng.random((n, 12)), columns=LAB[::-1]).rank(pct=True)
    speedy.insert(0, 'StudyInstanceUID', uids)
    speedy.to_csv(tmp / 'working/submission.csv', index=False)
    return uids, speedy, (tmp / 'working/submission.csv').read_bytes()


def fake_read(uid, geometries):
    requested.append(tuple(geometries))
    if uid == '1.2.3.4':
        raise ValueError('No decodable slices in selected series')
    r = np.random.default_rng(int(uid.split('.')[-1]))
    out = {}
    for g, (frames, size) in {'s64': (64, 336), 's128': (128, 256), 's96': (96, 336)}.items():
        if FAST[0]:
            size = 8   # windows() is bypassed in the fast tests: only the mask matters
        if g in geometries:
            volume = (r.random((frames, size, size)) * 255).astype(np.uint8)
            volume[frames - 10:] = 0
            out[g] = (volume, volume.reshape(frames, -1).sum(1) > 0)
    return out


COST = {'model72': 0.005, 'model71': 0.01, 'model73': 0.1}   # fake seconds per call (model71: 5 checkpoints)


def fake_predict(member):
    time.sleep(COST[member])
    return np.random.default_rng(abs(hash(member)) % 1000).random(12).astype(np.float32)


def run(extra_subs=(), fast=False, c=None):
    FAST[0] = fast
    c = c or code
    subs = [("_torch.device(f'cuda:{gpu}')", "_torch.device('cpu')"),
            ('_torch.cuda.device_count()', '2'),
            ('with _Pool(max_workers=_M80_READERS) as pool, _torch.cuda.device(device):', 'with _Pool(max_workers=_M80_READERS) as pool:'),
            ("_torch.autocast('cuda', dtype=_torch.float16)", "_torch.autocast('cpu', dtype=_torch.bfloat16)"),
            ('with _torch.cuda.device(gpu):\n            _torch.cuda.empty_cache()', 'pass'),
            ("uptime = float(open('/proc/uptime').read().split()[0])", 'uptime = 0.0'),
            ("start_ticks = int(open('/proc/self/stat').read().split(')')[-1].split()[19])", 'start_ticks = 0'),
            ("age = uptime - start_ticks / _o.sysconf('SC_CLK_TCK')", 'age = 3600.0'),
            ('return read_study(uid, geometries)', 'return _fake_read(uid, geometries)'),
            ('    import cv2 as _cv2\n', '    _cv2 = None\n'),
            ('    import pydicom as _dcm\n', '    _dcm = None\n'),
            ('    try:\n        from pydicom.pixels import apply_modality_lut as _lut\n    except ImportError:\n'
             '        from pydicom.pixel_data_handlers.util import apply_modality_lut as _lut\n', ''),
            ('/kaggle/input', (tmp / 'input').as_posix()),
            ('/kaggle/working', (tmp / 'working').as_posix())]
    if fast:   # no real inference: per-member fake cost, so ~100+ studies run in seconds
        subs += [('by_key[(geometry, res)] = windows(*volumes[geometry], res)', 'by_key[(geometry, res)] = None'),
                 ('sums.setdefault(member, []).append(predict(model, by_key[(geometry, res)][::stride], device))',
                  'sums.setdefault(member, []).append(_fake_predict(member))')]
    for a, b in subs + list(extra_subs):
        assert a in c, a
        c = c.replace(a, b)
    ns = {'_KE_NS': {'find_test_root': lambda: str(comp)}, '_fake_read': fake_read, '_fake_predict': fake_predict}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(c, 'model80_leg_v3', 'exec'), ns)
    return buf.getvalue()





import torch, timm
sys.path.insert(0, str(OTHER / 'scripts'))
from tpu_common import Classifier
model = Classifier('convnext_small.fb_in22k_ft_in1k_384')
model.backbone = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', pretrained=True, num_classes=0, in_chans=3,
                                   global_pool='avg', pretrained_cfg_overlay={'file': str(OTHER / 'data/weights/convnext_small.fb_in22k_ft_in1k_384.safetensors')})
torch.save({'model': model.state_dict(), 'arch': 'convnext_small.fb_in22k_ft_in1k_384', 'res': 384, 'epoch': 0,
            'training_config': {'arch': 'convnext_small.fb_in22k_ft_in1k_384'}}, weights / 'model140_best.pt')
FLEET = "_M80_FLEET = [('model140', ['model140_best.pt'], 's96', 2)]"
c = code6.replace("""_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),
              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)], 's64'),
              ('model73', ['model73_best.pt'], 's64')]""", FLEET)
assert FLEET in c
uids, speedy, before = setup(3)
out = run(c=c)
assert 'fleet leg blended' in out and "members ['model140']" in out and 'FAILED' not in out, out[-900:]
assert "strides {'model140': 2}" in out and "geometries ['s96']" in out, out[-900:]
# (with 3 studies at w 0.30 Speedy's order always dominates, so the file itself may not change; check the member output)
npz = dict(np.load(tmp / 'working/model80_fleet.npz'))
assert npz['model140'].shape == (3, 12) and np.isfinite(npz['model140']).all(), npz['model140']
print('model140 (ConvNeXt-S, random head) predictions', np.round(npz['model140'][:, :4], 3).tolist())
print([l for l in out.splitlines() if 'checkpoint' in l or 'blended' in l][-2:])
shutil.rmtree(tmp, ignore_errors=True)
print('CONVNEXT LEG TEST PASS')
