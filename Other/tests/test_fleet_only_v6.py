"""CPU test of the v6 FLEET-ONLY notebook cell (scripts/run_fleet_only.py --leg v6): real checkpoints, fake reader,
one s128x336 member at stride 2 + one s64 member; the submission must be the fleet mean probability. Harness from
test_fleet_leg_v6.py. Original notes: CPU tests of fleet leg v6 (scripts/fleet_leg_v6.py = v5 + partial blend on a time shortfall); harness from
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

tmp = Path(tempfile.mkdtemp(prefix='m80fo6_'))
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
    for g, (frames, size) in {'s64': (64, 336), 's128': (128, 256), 's128x336': (128, 336)}.items():
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





from run_fleet_only import cell_for
cell = cell_for("[('model73', ['model73_best.pt'], 's128x336', 2), ('model72', ['model72_best.pt'], 's64')]", 'v6')
real = tmp / 'input/competitions/rsna-knee-abnormality-detection'
real.mkdir(parents=True)
uids, speedy, before = setup(3)
import shutil as _sh
_sh.copy(comp / 'test_series.csv', real / 'test_series.csv')
pd.DataFrame({'StudyInstanceUID': uids}).to_csv(real / 'test.csv', index=False)
ss = pd.DataFrame({'StudyInstanceUID': uids}); [ss.insert(len(ss.columns), c, 0.5) for c in LAB]
ss.to_csv(real / 'sample_submission.csv', index=False)
(tmp / 'working/submission.csv').unlink()
out = run(c=cell.replace("_KE_NS = {'find_test_root': lambda: _fo_roots[0]}", "_KE_NS = {'find_test_root': lambda: _fo_roots[0]}"))
assert '[fleet-only] submission written: 3/3 studies read by the fleet, 0 failed' in out, out[-1200:]
sub = pd.read_csv(tmp / 'working/submission.csv', dtype={'StudyInstanceUID': str})
npz = dict(np.load(tmp / 'working/fleet_only.npz'))
mean = (npz['model73'] + npz['model72']) / 2
order = [list(npz['labels']).index(c) for c in sub.columns[1:]]
assert sub.StudyInstanceUID.tolist() == uids and np.allclose(sub.iloc[:, 1:].to_numpy(float), mean[:, order], atol=1e-6)
print([l for l in out.splitlines() if 'fleet-only' in l][-1][:220])
shutil.rmtree(tmp, ignore_errors=True)
print('FLEET-ONLY V6 TEST PASS')
