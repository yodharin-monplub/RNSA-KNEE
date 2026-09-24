"""Cached-vs-public-reader parity for the native384-v8 geometry (44 slots, 6%-94%, 384px).

Uses the public Raptor arm code (_KE_SRC from mattiaangeli/bend-the-knee-to-the-dinosaurs, verbatim)
to decode the 58 development studies from raw DICOMs, and compares both pixels and v8 predictions
against this project's s44 pilot cache. Gate for model28 paid training.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

nb = json.loads((ROOT / 'artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs/'
                        'bend-the-knee-to-the-dinosaurs.ipynb').read_text(encoding='utf-8'))
cell = next(''.join(c['source']) for c in nb['cells'] if '_KE_SRC = ' in ''.join(c['source']))
line = next(l for l in cell.splitlines() if l.startswith('_KE_SRC = '))
import ast
KE_SRC = ast.literal_eval(line[len('_KE_SRC = '):])

BODY = r'''
import hashlib, io, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
NS = {'__name__': 'public_raptor'}
exec(compile(KE_SRC, '<raptor>', 'exec'), NS)
LAB = NS['LAB']

def our_windows(volume, mask, size):
    # scripts/train_model16.py training/eval windows (model12.inference.windows), verbatim logic
    import torch.nn.functional as F
    valid = np.flatnonzero(mask)
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    centers = list(range(lo+1, hi)) or [max(1, min((lo+hi)//2, len(volume)-2))]
    slices = np.stack([volume[c-1:c+2] for c in centers]).astype(np.float32) / 255
    images = F.interpolate(torch.from_numpy(slices), size=(size, size), mode='bilinear', align_corners=False)
    return (images - torch.tensor([.485,.456,.406]).view(1,3,1,1)) / torch.tensor([.229,.224,.225]).view(1,3,1,1)

def main():
    started = time.time()
    inputs = Path('/kaggle/input')
    comp = next(p for p in [inputs/'competitions/rsna-knee-abnormality-detection', inputs/'rsna-knee-abnormality-detection'] if (p/'train.csv').exists())
    caches = [p for p in inputs.rglob('cache_manifest.json')]
    assert len(caches) == 1, caches
    cache = caches[0].parent
    manifest = json.loads(caches[0].read_text())
    assert manifest['geometry']['span'] == [.06, .94] and manifest['geometry']['size'] == 384
    assert sum(s[2] for s in manifest['geometry']['slots']) == 44
    weights = [p for p in inputs.rglob('raptor_ft_coatnet_v8_full_swa.pt')]
    assert len(weights) == 1, weights
    dev = 'cuda'
    model, res = NS['load_model'](str(weights[0]), 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, dev)
    NS['IMG'] = 384; NS['SLOTS'] = list(NS['_SLOTS44']); NS['MAXS'] = 44
    NS['SPAN_LO'], NS['SPAN_HI'] = .06, .94
    train = pd.read_csv(comp/'train.csv')
    gold = train.dropna(subset=LAB).set_index('StudyInstanceUID')
    series = pd.read_csv(comp/'train_series.csv')
    series['StudyInstanceUID'] = series['StudyInstanceUID'].astype(str)
    series['SeriesInstanceUID'] = series['SeriesInstanceUID'].astype(str)
    records = {k: f.to_dict('records') for k, f in series.groupby('StudyInstanceUID')}
    reader = NS['_make_reader']()
    pix_diff, raw_p, cached_p, ours_p, extra_nonzero = 0, {}, {}, {}, 0
    for shard in manifest['shards']:
        path = cache/shard['file']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == shard['sha256']
        with np.load(path, allow_pickle=False) as a:
            ids, volumes, masks = a['ids'].tolist(), a['volumes'], a['masks']
        for i, uid in enumerate(ids):
            if uid not in gold.index:
                continue
            vol, mask = NS['build_study'](uid, records, str(comp/'train_series'), reader)
            cvol, cmask = volumes[i], masks[i]
            extra_nonzero += int(cvol[44:].any())
            pix_diff = max(pix_diff, int(np.abs(vol.astype(int) - cvol[:44].astype(int)).max()))
            raw_p[uid] = NS['infer_probs'](model, NS['eval_windows'](vol, mask, k=42, res=res), dev)
            cached_p[uid] = NS['infer_probs'](model, NS['eval_windows'](cvol[:44], cmask[:44].astype(np.uint8), k=42, res=res), dev)
            ours_p[uid] = NS['infer_probs'](model, our_windows(cvol, cmask, res), dev)
        print('studies', len(raw_p), round(time.time()-started), flush=True)
    ids = sorted(raw_p)
    assert len(ids) == 58
    y = gold.loc[ids, LAB].to_numpy()
    R, C, O = (np.stack([d[u] for u in ids]) for d in (raw_p, cached_p, ours_p))
    auc = lambda P: float(np.mean([roc_auc_score(y[:, j], P[:, j]) for j in range(12)]))
    report = {'studies': 58, 'max_pixel_abs_diff': pix_diff, 'cached_frames_after_44_nonzero': extra_nonzero,
              'max_prob_diff_cached_vs_raw': float(np.abs(C - R).max()),
              'max_prob_diff_trainwindows_vs_raw': float(np.abs(O - R).max()),
              'gold_auc_raw_public': auc(R), 'gold_auc_cached_public': auc(C), 'gold_auc_cached_trainwindows': auc(O),
              'elapsed_seconds': time.time() - started}
    report['status'] = 'PASS' if (report['max_prob_diff_cached_vs_raw'] < 1e-3 and extra_nonzero == 0) else 'FAIL'
    pd.DataFrame(R, index=pd.Index(ids, name='StudyInstanceUID'), columns=LAB).to_csv('/kaggle/working/v8_raw_gold_predictions.csv')
    Path('/kaggle/working/v8_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

main()
'''

source = f'KE_SRC = {KE_SRC!r}\n' + BODY
compile(source, 'v8_parity', 'exec')
api = KaggleApi(); api.authenticate()
out = ROOT / 'artifacts/v8_parity_notebook'
out.mkdir(parents=True, exist_ok=True)
(out / 'inference.py').write_text(source)
meta = {'id': 'yodharinmonplub/rsna-knee-v8-cache-parity', 'title': 'RSNA Knee V8 Cache Parity',
        'code_file': 'inference.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'dataset_sources': ['dreaddevelopment/raptor-knee-native384'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ['yodharinmonplub/rsna-knee-dense-cache-384-6-94-pilot-s44'], 'model_sources': []}
(out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print(api.kernels_push(str(out), acc='NvidiaTeslaT4'))
