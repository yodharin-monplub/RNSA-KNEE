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

code = FLEET_LEG_V3.lstrip()
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


def run(extra_subs=(), fast=False):
    FAST[0] = fast
    c = code
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


# (A) real models, 6 studies, one unreadable
uids, speedy, before = setup(6)
out = run()
print(out[-900:])
after = pd.read_csv(tmp / 'working/submission.csv', dtype={'StudyInstanceUID': str})
assert after.columns.tolist() == ['StudyInstanceUID'] + LAB[::-1] and after.StudyInstanceUID.tolist() == uids
assert np.isfinite(after[LAB].to_numpy()).all() and (tmp / 'working/submission_speedy.csv').read_bytes() == before
assert '[model80] fleet leg blended at w=0.3: 5/6 studies' in out and out.count('sha256=') == 7
assert "members ['model72', 'model71', 'model73']; dropped for time []; geometries ['s128', 's64']" in out
assert set(requested) == {('s128', 's64')}, set(requested)
fleet = np.load(tmp / 'working/model80_fleet.npz')
assert np.isnan(fleet['model72'][4]).all() and np.isfinite(fleet['model73'][[0, 1, 2, 3, 5]]).all()
print('(A) blend OK')

# (B) missing checkpoint -> FAILED, file untouched
(tmp / 'working/submission.csv').write_bytes(before)
out = run([("('model73', ['model73_best.pt'], 's64')", "('model73', ['model73_missing.pt'], 's64')")])
assert 'fleet leg FAILED' in out and (tmp / 'working/submission.csv').read_bytes() == before, out[-500:]
print('(B) failure path OK')

# (C) deadline already passed -> TimeoutError -> kept
out = run([('age = 3600.0', 'age = 9 * 3600.0')])
assert 'fleet leg FAILED (TimeoutError' in out and (tmp / 'working/submission.csv').read_bytes() == before
print('(C) time guard OK')

# (D) priority dropping: 200 studies, 0.155 s/study of fake member time on 2 threads (~15 s), budget 9 s -> model73
#     (the slowest, last in priority) is dropped; model72 + model71 blend all readable studies
uids, speedy, before = setup(200)
requested.clear()
# the test budget starts when the study loop starts (checkpoint loading on CPU takes ~9 s by itself)
tight = [("deadline = started + 9 * 3600 - age - _M80_SAFETY_S", "deadline = started + 10 ** 9"),
         ("    with _Pool(max_workers=len(gpus)) as pool:\n        for job in",
          "    deadline = _t.time() + _M80_TEST_BUDGET\n    with _Pool(max_workers=len(gpus)) as pool:\n        for job in")]
out = run([('_M80_SAFETY_S = 1800\n', '_M80_SAFETY_S = 1800\n_M80_TEST_BUDGET = 9.0\n')] + tight, fast=True)
print(out[-700:])
assert "dropped model73 after" in out and "dropped model71" not in out, out[-1200:]
assert "members ['model72', 'model71']; dropped for time ['model73']" in out and '199/200 studies' in out
assert (tmp / 'working/submission.csv').read_bytes() != before
assert set(requested) == {('s128', 's64')}, set(requested)   # s64 stays needed by model71
print('(D) dropping OK')

# (E) the first member alone is too slow: model73 and model71 are dropped, then the leg is abandoned, Speedy kept
(tmp / 'working/submission.csv').write_bytes(before)
COST['model72'] = 0.1
out = run([('_M80_SAFETY_S = 1800\n', '_M80_SAFETY_S = 1800\n_M80_TEST_BUDGET = 9.0\n')] + tight, fast=True)
assert 'fleet leg FAILED (TimeoutError' in out and (tmp / 'working/submission.csv').read_bytes() == before, out[-1200:]
assert 'dropped model73' in out and 'dropped model71' in out, out[-1200:]
print('(E) abandon OK:', [l for l in out.splitlines() if 'dropped' in l or 'FAILED' in l][-3:])
shutil.rmtree(tmp, ignore_errors=True)
print('ALL FLEET LEG V3 TESTS PASS')
