"""CPU tests of fleet leg v3: (A) the v2 checks with real checkpoints + real windows/predict/blend and a fake reader that
honours the requested geometries; (B) missing checkpoint -> FAILED, file untouched; (C) deadline passed -> Speedy kept;
(D) priority dropping: slow fake members and a tight budget -> the last member is dropped, the rest still blend;
(E) budget too small even for the first member -> leg abandoned, Speedy kept."""
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

code3 = FLEET_LEG_V3.lstrip()
code4 = FLEET_LEG_V4.lstrip()
code = code3
code = code.replace("_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),", "_M80_FLEET = [('model72', ['model72_best.pt'], 's128'),")
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
       'Synovitis', "Baker's", 'Contusion', 'Fracture']

tmp = Path(tempfile.mkdtemp(prefix='m80v3_'))
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
    for g, (frames, size) in {'s64': (64, 336), 's128': (128, 256)}.items():
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
                 ('sums.setdefault(member, []).append(predict(model, by_key[(geometry, res)], device))',
                  'sums.setdefault(member, []).append(_fake_predict(member))')]
    for a, b in subs + list(extra_subs):
        assert a in c, a
        c = c.replace(a, b)
    ns = {'_KE_NS': {'find_test_root': lambda: str(comp)}, '_fake_read': fake_read, '_fake_predict': fake_predict}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(c, 'model80_leg_v3', 'exec'), ns)
    return buf.getvalue()



# (V4-1) all strides 1 (3-tuples): v4 must write exactly v3's submission
import numpy as np, pandas as pd
uids, speedy, before = setup(3)
out3 = run(c=code3.replace("_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),", "_M80_FLEET = [('model72', ['model72_best.pt'], 's128'),"))
sub3 = (tmp / 'working/submission.csv').read_bytes(); npz3 = dict(np.load(tmp / 'working/model80_fleet.npz'))
(tmp / 'working/submission.csv').write_bytes(before)
code4s = code4.replace("_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),", "_M80_FLEET = [('model72', ['model72_best.pt'], 's128'),")
out4 = run(c=code4s)
sub4 = (tmp / 'working/submission.csv').read_bytes(); npz4 = dict(np.load(tmp / 'working/model80_fleet.npz'))
assert sub3 == sub4, 'v4 with stride 1 differs from v3'
assert all(np.array_equal(npz3[k], npz4[k], equal_nan=True) for k in ('model72', 'model71', 'model73'))
assert "strides {'model72': 1, 'model71': 1, 'model73': 1}" in out4, out4[-600:]
print('(V4-1) stride 1 == v3 OK')
# (V4-2) model73 at stride 2: only model73's predictions change; the log shows the strides
(tmp / 'working/submission.csv').write_bytes(before)
out5 = run(c=code4s.replace("('model73', ['model73_best.pt'], 's64')]", "('model73', ['model73_best.pt'], 's64', 2)]"))
npz5 = dict(np.load(tmp / 'working/model80_fleet.npz'))
assert "strides {'model72': 1, 'model71': 1, 'model73': 2}" in out5, out5[-600:]
assert np.array_equal(npz4['model72'], npz5['model72'], equal_nan=True) and np.array_equal(npz4['model71'], npz5['model71'], equal_nan=True)
assert not np.allclose(npz4['model73'], npz5['model73'], equal_nan=True)
print('(V4-2) stride 2 OK: max |model73 diff|', float(np.nanmax(np.abs(npz4['model73'] - npz5['model73']))))
shutil.rmtree(tmp, ignore_errors=True)
print('ALL FLEET LEG V4 TESTS PASS')
