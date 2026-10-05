"""CPU tests of fleet leg v6 (scripts/fleet_leg_v6.py = v5 + partial blend on a time shortfall); harness from
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

tmp = Path(tempfile.mkdtemp(prefix='m80v6_'))
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




S128 = ("_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),", "_M80_FLEET = [('model72', ['model72_best.pt'], 's128'),")
code5s, code6s = code5.replace(*S128), code6.replace(*S128)

# (V6-1) a run that finishes in time: v6 writes exactly v5's submission and member predictions
uids, speedy, before = setup(3)
out5 = run(c=code5s)
sub5 = (tmp / 'working/submission.csv').read_bytes(); npz5 = dict(np.load(tmp / 'working/model80_fleet.npz'))
(tmp / 'working/submission.csv').write_bytes(before)
out6 = run(c=code6s)
sub6 = (tmp / 'working/submission.csv').read_bytes(); npz6 = dict(np.load(tmp / 'working/model80_fleet.npz'))
assert sub5 == sub6, 'v6 differs from v5 on a run that finished in time'
assert all(np.array_equal(npz5[k], npz6[k], equal_nan=True) for k in ('model72', 'model71', 'model73'))
assert 'PARTIAL' not in out6 and ' 2/3 studies read by the fleet, 1 failed' not in out6
print('(V6-1) in-time run == v5 OK')

# (V6-2) the first member alone cannot finish (v3 test E with a slower model72): v5 abandons the leg, v6 keeps model72
# running to the deadline and blends the complete studies; the unfinished ones keep Speedy's order
tight = [("deadline = started + 9 * 3600 - age - _M80_SAFETY_S", "deadline = started + 10 ** 9"),
         ("    with _Pool(max_workers=len(gpus)) as pool:\n        for job in",
          "    deadline = _t.time() + _M80_TEST_BUDGET\n    with _Pool(max_workers=len(gpus)) as pool:\n        for job in")]
budget = [('_M80_SAFETY_S = 1800\n', '_M80_SAFETY_S = 1800\n_M80_TEST_BUDGET = 9.0\n')]
uids, speedy, before = setup(200)
COST['model72'] = 0.2
out5 = run(budget + tight, fast=True, c=code5s)
assert 'fleet leg FAILED (TimeoutError' in out5 and (tmp / 'working/submission.csv').read_bytes() == before, out5[-900:]
out6 = run(budget + tight, fast=True, c=code6s)
assert 'continuing to the deadline for a partial blend' in out6 and 'PARTIAL (time)' in out6, out6[-1200:]
assert 'dropped model73' in out6 and 'dropped model71' in out6, out6[-1200:]
final = pd.read_csv(tmp / 'working/submission.csv', dtype={'StudyInstanceUID': str})
npz = dict(np.load(tmp / 'working/model80_fleet.npz'))
covered = np.isfinite(npz['model72']).all(axis=1)
assert 0.2 * 200 <= covered.sum() < 200, covered.sum()
assert final.StudyInstanceUID.tolist() == speedy.StudyInstanceUID.tolist()
labels = [c for c in final.columns if c != 'StudyInstanceUID']
for c in labels:   # unfinished studies: same order as Speedy's
    a = final.loc[~covered, c].rank().to_numpy(); b = speedy.loc[~covered, c].rank().to_numpy()
    assert np.array_equal(a, b), c
assert not np.allclose(final[labels].to_numpy(), speedy[labels].to_numpy())
print('(V6-2) partial blend OK:', int(covered.sum()), '/ 200 complete;', [l for l in out6.splitlines() if 'PARTIAL' in l][0][:160])

# (V6-3) below _M80_MIN_PARTIAL complete -> Speedy's file kept
(tmp / 'working/submission.csv').write_bytes(before)
COST['model72'] = 1.0
out7 = run(budget + tight, fast=True, c=code6s)
assert 'fleet leg FAILED (TimeoutError' in out7 and 'complete (<' in out7, out7[-900:]
assert (tmp / 'working/submission.csv').read_bytes() == before
print('(V6-3) too little coverage -> Speedy kept OK')
shutil.rmtree(tmp, ignore_errors=True)
print('ALL FLEET LEG V6 TESTS PASS')
