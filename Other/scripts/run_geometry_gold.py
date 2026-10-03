"""Gold (58-case) test of Raptor slice geometry for the model61 head-retrained CoAtNet arms (CPU kernel).

The Raptor author documents the fine-spacing corpus as 80 slices/study over 2-98% (slots 22/18/15/10/15,
78 windows) and reports "density is the lever" (44 -> 64 -> 80 slices: LB 0.917 -> 0.928 -> 0.932). Our
fine-spacing arm runs at 64 slices over 6-94% with 62 windows (EXPERIMENTS.md, geometry bug). Geometry is
inference-only (the classifier pools over any number of 3-slice windows), so each variant just re-decodes the
gold studies with different Raptor globals and reuses the exact model61 checkpoints.

Usage: python scripts/run_geometry_gold.py fine|maxspan [--push]
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY') or os.environ['KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')

SLOTS80 = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15)]
SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
# Same 80-slice density but with the extra slices on the axial plane (PF OA headroom, EXPERIMENTS.md).
SLOTS80AX = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 28)]

VARIANTS = {
    'fine': ('model61_fine_best.pt', 'rsna-knee-head-retrain-fine', 336, {
        'cur_64_06_94': (SLOTS64, (0.06, 0.94), 62),
        'raptor_80_02_98': (SLOTS80, (0.02, 0.98), 78),
        's64_02_98': (SLOTS64, (0.02, 0.98), 62),
        'ax80_02_98': (SLOTS80AX, (0.02, 0.98), 78),
    }),
    # model70 follow-up: the other three arms also run a reversed-window view (windows.flip(1), as in the notebook);
    # the fine-spacing arm has none. Same decode, so one kernel scores both views at the adopted 80-slice geometry.
    'finerev': ('model61_fine_best.pt', 'rsna-knee-head-retrain-fine', 336, {
        'raptor_80_02_98': (SLOTS80, (0.02, 0.98), 78),
        'raptor_80_02_98_rev': (SLOTS80, (0.02, 0.98), 78, True),
    }),
    'maxspan': ('model61_maxspan_best.pt', 'rsna-knee-head-retrain-maxspan', 336, {
        'cur_64_02_98': (SLOTS64, (0.02, 0.98), 62),
        'raptor_80_02_98': (SLOTS80, (0.02, 0.98), 78),
        'ax80_02_98': (SLOTS80AX, (0.02, 0.98), 78),
    }),
}

BODY = r'''
import json as _json, os as _os, time as _time
import numpy as _np, pandas as _pd, torch as _torch
from sklearn.metrics import roc_auc_score as _auc
ns = {'__name__': '_ke_raptor'}
exec(compile(_KE_SRC, '<raptor>', 'exec'), ns)
CFG = _json.loads(__CFG__)
t0 = _time.time()
_torch.set_num_threads(max(1, _os.cpu_count() or 1))
dev = 'cuda' if _torch.cuda.is_available() else 'cpu'
root = next(p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                        '/kaggle/input/rsna-knee-abnormality-detection') if _os.path.isfile(p + '/train.csv'))
sdir = next(root + '/' + d for d in ('train_series', 'train_images') if _os.path.isdir(root + '/' + d))
LAB = ns['LAB']
train = _pd.read_csv(root + '/train.csv')
train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
gold = train.dropna(subset=LAB).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
ids = sorted(gold.index)
assert len(ids) == 58, len(ids)
ser = _pd.read_csv(root + '/train_series.csv')
for c in ('StudyInstanceUID', 'SeriesInstanceUID'):
    ser[c] = ser[c].astype(str)
ser = ser[ser['StudyInstanceUID'].isin(ids)]
series = {k: f.to_dict('records') for k, f in ser.groupby('StudyInstanceUID')}
hits = [p for p in (f"/kaggle/input/notebooks/yodharinmonplub/{CFG['slug']}/{CFG['file']}",
                    f"/kaggle/input/{CFG['slug']}/{CFG['file']}") if _os.path.isfile(p)]
assert len(hits) == 1, hits
model, res = ns['load_model'](hits[0], 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, dev)
print('device', dev, '| threads', _torch.get_num_threads(), '| res', res, '| series dir', sdir, flush=True)
reader = ns['_make_reader']()
y = gold.loc[ids, LAB].to_numpy().astype(float)
report = {}
for name, (slots, span, k, *rest) in CFG['variants'].items():
    reverse = bool(rest and rest[0])
    ns['IMG'] = CFG['img']; ns['SLOTS'] = [tuple(s) for s in slots]; ns['MAXS'] = sum(s[2] for s in slots)
    ns['SPAN_LO'], ns['SPAN_HI'] = span; ns['K_EVAL'] = k
    probs = _np.full((len(ids), len(LAB)), 0.5, _np.float32); fallbacks = 0
    for i, uid in enumerate(ids):
        try:
            vol, mask = ns['build_study'](uid, series, sdir, reader)
            wins = ns['eval_windows'](vol, mask, k=k, res=res, norm=ns['NORM'])
            probs[i] = ns['infer_probs'](model, wins.flip(1).contiguous() if reverse else wins, dev)
        except Exception as e:
            fallbacks += 1
            print('FALLBACK', name, uid[:16], type(e).__name__, e, flush=True)
        if (i + 1) % 10 == 0:
            print(f'  {name} {i + 1}/58 | {_time.time() - t0:.0f}s', flush=True)
    per = {LAB[j]: float(_auc(y[:, j], probs[:, j])) for j in range(len(LAB))}
    report[name] = {'macro': float(_np.mean(list(per.values()))), 'per_finding': per, 'fallbacks': fallbacks}
    _pd.DataFrame(probs, index=_pd.Index(ids, name='StudyInstanceUID'), columns=LAB).to_csv(f'/kaggle/working/{name}.csv')
    print(f"[{name}] gold macro AUC {report[name]['macro']:.4f} | fallbacks {fallbacks} | {_time.time() - t0:.0f}s", flush=True)
report['elapsed_seconds'] = _time.time() - t0
_os.makedirs('/kaggle/working', exist_ok=True)
open('/kaggle/working/geometry_gold.json', 'w').write(_json.dumps(report, indent=2))
print(_json.dumps({k: (v['macro'] if isinstance(v, dict) else v) for k, v in report.items()}, indent=2), flush=True)
'''


def ke_source():
    nb = json.loads((ROOT / 'artifacts/model61_notebook/notebook.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        if '_KE_SRC = ' in src:
            start = src.index('_KE_SRC = ')
            end = src.index("_KE_NS = {'__name__'")
            return src[start:end]
    raise RuntimeError('no _KE_SRC cell in the model61 notebook')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=sorted(VARIANTS))
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    file, slug, img, variants = VARIANTS[a.arm]
    cfg = {'file': file, 'slug': slug, 'img': img, 'variants': variants}
    code = ke_source() + '\n' + BODY.replace('__CFG__', repr(json.dumps(cfg)))
    compile(code, 'geometry', 'exec')
    out = ROOT / f'artifacts/geometry_gold_{a.arm}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inference.py').write_text(code, encoding='utf-8')
    meta = {'id': f'yodharinmonplub/rsna-knee-geometry-gold-{a.arm}', 'title': f'RSNA Knee Geometry Gold {a.arm}',
            'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': False, 'enable_internet': False, 'dataset_sources': [],
            'competition_sources': ['rsna-knee-abnormality-detection'],
            'kernel_sources': [f'yodharinmonplub/{slug}'], 'model_sources': []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print('built', out)
    if a.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(out))
        (out / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
