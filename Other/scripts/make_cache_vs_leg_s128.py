"""Free CPU kernel on the teammate account: are the 128-frame 336 px TRAINING cache volumes (model135's cache,
krittanutsomtuas/rsna-knee-dense-cache-336-part{0..3}-s128) identical to what fleet leg v5's reader builds from the
DICOMs at geometry 's128x336'? For the 58 gold studies: load their rows from the cache shards, rebuild them with the
leg's read_study(uid, ['s128x336', 's96']) (code cut verbatim from fleet_leg_v5.py; asking for s96 too checks the
one-pass reader with two geometries), compare exactly. Usage: python make_cache_vs_leg_s128.py"""
import json
import sys
import textwrap
from pathlib import Path

OTHER = Path(r'C:\Users\yodha\OneDrive\resources hp\coding\RSNA Knee Abnormality Detection\Other')
sys.path.insert(0, str(OTHER / 'scripts'))
from fleet_leg_v5 import FLEET_LEG_V5 as L


def block(first, stop):
    a = L.index(first)
    return textwrap.dedent(L[a:L.index(stop, a)])


leg = '\n'.join([block("    GEOMETRIES = {", "    # v4: (member, files, geometry[, stride])"),
                 block("    def ordered_files(directory):", "    def timed_read(uid, geometries):")])
assert "'s128x336'" in leg and 'def read_study' in leg

KERNEL = r'''
import json, os, time
from pathlib import Path
import numpy as _np, pandas as _pd, pydicom as _dcm, cv2 as _cv2
try:
    from pydicom.pixels import apply_modality_lut as _lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut as _lut
print('versions pydicom', _dcm.__version__, 'cv2', _cv2.__version__, 'numpy', _np.__version__, flush=True)
IN = Path('/kaggle/input')


def find(name, max_depth=7):
    hits, base = [], len(IN.parts)
    for folder, dirs, files in os.walk(IN):
        depth = len(Path(folder).parts) - base
        if depth >= max_depth or (depth == 1 and Path(folder).name == 'competitions'):
            dirs[:] = []
        if name in files:
            hits.append(Path(folder) / name)
    return sorted(hits)


comp = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                              '/kaggle/input/rsna-knee-abnormality-detection') if Path(p, 'train.csv').exists())
series = _pd.read_csv(comp / 'train_series.csv')
series['StudyInstanceUID'] = series['StudyInstanceUID'].astype(str)
series['SeriesInstanceUID'] = series['SeriesInstanceUID'].astype(str)
groups = {uid: frame.to_dict('records') for uid, frame in series.groupby('StudyInstanceUID')}
series_root = comp / 'train_series' if (comp / 'train_series').is_dir() else comp / 'train_images'
exec(__LEG__)
partition = _pd.read_csv(find('partitions.csv')[0]).set_index('StudyInstanceUID')
gold = [str(u) for u in partition.index[partition.split == 'gold']]
manifests = [m for m in find('cache_manifest.json') if '-s128' in str(m)]
print('manifests', [str(m) for m in manifests], flush=True)
where = {}
for m in manifests:
    man = json.loads(m.read_text())
    print('geometry', json.dumps(man['geometry']), flush=True)
    assert [tuple(s) for s in man['geometry']['slots']] == list(GEOMETRIES['s128x336'][0])
    assert man['geometry']['size'] == GEOMETRIES['s128x336'][1]
    for shard in man['shards']:
        for row, uid in enumerate(shard['ids']):
            if uid in gold:
                where[uid] = (str(m.parent / shard['file']), row)
print('gold studies found in cache', len(where), 'of', len(gold), flush=True)
rows, cache = [], {}
t0 = time.time()
for uid, (path, row) in sorted(where.items(), key=lambda x: x[1]):
    if path not in cache:
        cache.clear()
        cache[path] = _np.load(path)
    z = cache[path]
    assert str(z['ids'][row]) == uid
    cv, cm = z['volumes'][row], z['masks'][row]
    both = read_study(uid, ['s128x336', 's96'])
    lv, lm = both['s128x336']
    diff = _np.abs(cv.astype(int) - lv.astype(int))
    frames = [int(f) for f in _np.flatnonzero(diff.reshape(len(diff), -1).max(1) > 0)]
    s96_alone = read_study(uid, ['s96'])['s96']
    r = {'uid': uid, 'identical': bool(_np.array_equal(cv, lv) and _np.array_equal(cm, lm)),
         'mask_equal': bool(_np.array_equal(cm, lm)), 'max_pixel_diff': int(diff.max()),
         'n_frames_differing': len(frames), 'frames_differing': frames[:12],
         's96_same_with_and_without_s128x336': bool(_np.array_equal(both['s96'][0], s96_alone[0]) and _np.array_equal(both['s96'][1], s96_alone[1]))}
    rows.append(r)
    print(json.dumps(r), flush=True)
summary = {'n': len(rows), 'identical': sum(r['identical'] for r in rows),
           'max_pixel_diff': max(r['max_pixel_diff'] for r in rows) if rows else None,
           's96_consistent': sum(r['s96_same_with_and_without_s128x336'] for r in rows),
           'seconds': round(time.time() - t0)}
print('SUMMARY', json.dumps(summary), flush=True)
Path('/kaggle/working/cache_vs_leg_s128.json').write_text(json.dumps({'summary': summary, 'rows': rows}, indent=1))
'''
code = KERNEL.replace('__LEG__', repr(leg))
compile(code, 'cache_vs_leg_s128', 'exec')
folder = OTHER / 'artifacts/cache_vs_leg_s128_notebook'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'check.py').write_text(code, encoding='utf-8')
meta = {'id': 'krittanutsomtuas/rsna-knee-cache-vs-leg-s128', 'title': 'RSNA Knee Cache Vs Leg S128', 'code_file': 'check.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': False, 'dataset_sources': ['krittanutsomtuas/rsna-knee-trainer-kit'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': [f'krittanutsomtuas/rsna-knee-dense-cache-336-part{i}-s128' for i in range(4)], 'model_sources': []}
(folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('built', folder)
