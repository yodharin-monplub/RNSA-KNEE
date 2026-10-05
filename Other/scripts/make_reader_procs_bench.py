"""Free CPU kernel (teammate account): is fleet leg v5's DICOM reading (one-pass read_study, s64 + s96) GIL-bound?
Times a ThreadPool(4) - the leg's current scheme, 2 readers per GPU - against a spawn ProcessPool(4) and (8) running the
SAME reader code (written to a module file), each on its own disjoint set of cold training studies (no page-cache help),
and checks on a further 8 studies that a process returns byte-identical volumes and masks to an in-process call.
Usage: python make_reader_procs_bench.py"""
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


reader = '\n'.join([block("    GEOMETRIES = {", "    # v4: (member, files, geometry[, stride])"),
                    block("    def ordered_files(directory):", "    def timed_read(uid, geometries):")])
assert 'def read_study' in reader
MODULE = '''import numpy as _np
import pydicom as _dcm
import cv2 as _cv2
from pathlib import Path
try:
    from pydicom.pixels import apply_modality_lut as _lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut as _lut
groups = {}
series_root = None
''' + reader + '''

def read_one(uid, rows, root, geometries):
    global series_root
    series_root = Path(root)
    groups[uid] = rows
    try:
        return read_study(uid, geometries)
    finally:
        groups.pop(uid, None)
'''
compile(MODULE, 'm80reader', 'exec')

KERNEL = r'''
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd


def main():
    Path('/kaggle/working/m80reader.py').write_text(__MODULE__)
    sys.path.insert(0, '/kaggle/working')
    import m80reader
    print('cpus', os.cpu_count(), flush=True)
    comp = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                                  '/kaggle/input/rsna-knee-abnormality-detection') if Path(p, 'train.csv').exists())
    series = pd.read_csv(comp / 'train_series.csv')
    series['StudyInstanceUID'] = series['StudyInstanceUID'].astype(str)
    series['SeriesInstanceUID'] = series['SeriesInstanceUID'].astype(str)
    groups = {uid: frame.to_dict('records') for uid, frame in series.groupby('StudyInstanceUID')}
    root = str(comp / 'train_series' if (comp / 'train_series').is_dir() else comp / 'train_images')
    uids = sorted(groups)
    rng = np.random.default_rng(20261004)
    rng.shuffle(uids)
    G = ['s64', 's96']
    sets = {'threads4': uids[0:60], 'procs4': uids[60:120], 'procs8': uids[120:180], 'threads4_again': uids[180:240]}
    report = {}
    for name, chunk in sets.items():
        t0 = time.time()
        if name.startswith('threads'):
            with ThreadPoolExecutor(4) as pool:
                out = list(pool.map(lambda u: m80reader.read_one(u, groups[u], root, G), chunk))
        else:
            with ProcessPoolExecutor(int(name[-1]), mp_context=mp.get_context('spawn')) as pool:
                out = list(pool.map(m80reader.read_one, chunk, [groups[u] for u in chunk], [root] * len(chunk), [G] * len(chunk)))
        dt = time.time() - t0
        report[name] = {'studies': len(chunk), 'seconds': round(dt, 1), 's_per_study': round(dt / len(chunk), 3)}
        print(name, report[name], flush=True)
    check = uids[240:248]
    with ProcessPoolExecutor(4, mp_context=mp.get_context('spawn')) as pool:
        viaproc = list(pool.map(m80reader.read_one, check, [groups[u] for u in check], [root] * len(check), [G] * len(check)))
    same = 0
    for u, p in zip(check, viaproc):
        d = m80reader.read_one(u, groups[u], root, G)
        same += all(np.array_equal(d[g][0], p[g][0]) and np.array_equal(d[g][1], p[g][1]) for g in G)
    report['process_vs_inprocess_identical'] = f'{same}/{len(check)}'
    print('SUMMARY', json.dumps(report), flush=True)
    Path('/kaggle/working/reader_procs_bench.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
'''
code = KERNEL.replace('__MODULE__', repr(MODULE))
compile(code, 'bench', 'exec')
folder = OTHER / 'artifacts/reader_procs_bench_notebook'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'bench.py').write_text(code, encoding='utf-8')
meta = {'id': 'krittanutsomtuas/rsna-knee-reader-procs-bench', 'title': 'RSNA Knee Reader Procs Bench', 'code_file': 'bench.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': False, 'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': [], 'model_sources': []}
(folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('built', folder)
