"""Free-CPU kernel on the teammate account: is a WEIGHT SOUP of the model110 folds (uniform average of the 4 fold
checkpoints, same ImageNet init + recipe, different data folds) as good as their prediction ensemble?
If yes, one leg member (17 min full / 8.5 min stride 2 on the hidden set) carries a 4-fold ensemble.
Evaluates on the 58 gold studies through fleet leg v3's own reader/windows/Classifier/predict (CPU fp32, even windows
= stride 2): soup with averaged BatchNorm statistics, and soup with BatchNorm statistics re-estimated on 16 non-gold
training studies. Reference (from the TPU runs' gold_predictions.csv): each fold and the 4-fold mean ensemble.
Usage: python make_soup_check.py [--push]"""
import json
import os
import sys
import textwrap
from pathlib import Path

OTHER = Path(r'C:\Users\yodha\OneDrive\resources hp\coding\RSNA Knee Abnormality Detection\Other')
sys.path.insert(0, str(OTHER / 'scripts'))
from fleet_leg_v3 import FLEET_LEG_V3 as L


def block(first, stop):
    a = L.index(first)
    return textwrap.dedent(L[a:L.index(stop, a)])


parts = [block("    GEOMETRIES = {", "    used_geometries"),
         block("    class Classifier(_nn.Module):", "    def ordered_files(directory):"),
         block("    def ordered_files(directory):", "    def timed_read(uid, geometries):"),
         block("    def windows(volume, mask, size):", "    def find(name):"),
         block("    @_torch.inference_mode()", "    def run_gpu(gpu):")]
leg_code = '\n'.join(parts).replace("with _torch.autocast('cuda', dtype=_torch.float16):", "with _contextlib.nullcontext():")
assert 'nullcontext' in leg_code and 'def read_study' in leg_code and 'def predict' in leg_code

KERNEL = r'''
import contextlib as _contextlib, json, os, time
from pathlib import Path
import numpy as _np, pandas as _pd, pydicom as _dcm, cv2 as _cv2, timm as _timm, torch as _torch
from torch import nn as _nn
from torch.nn import functional as _F
from sklearn.metrics import roc_auc_score
try:
    from pydicom.pixels import apply_modality_lut as _lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut as _lut
_torch.set_num_threads(os.cpu_count() or 4)
IN = Path('/kaggle/input')
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
          'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
T_START = time.time()


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
partition.index = partition.index.astype(str)
gold_ids = partition.index[partition.split == 'gold'].tolist()
calib_ids = sorted(partition.index[partition.split == 'train'].tolist())[:16]
truth = _pd.read_csv(comp / 'train.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[gold_ids, LABELS]
macro = lambda p: float(_np.mean([roc_auc_score(truth[c], p[:, j]) for j, c in enumerate(LABELS)]))
spear = lambda a, b: float(_np.mean([_pd.Series(a[:, j]).corr(_pd.Series(b[:, j]), method='spearman') for j in range(12)]))

report = {'gold_studies': len(gold_ids), 'calibration_studies': len(calib_ids)}
paths, golds = {}, {}
for k in range(4):
    hits = [p for p in find('best.pt') if p.as_posix().endswith(f'model110/fold{k}/best.pt')]
    ghits = [p for p in find('gold_predictions.csv') if p.as_posix().endswith(f'model110/fold{k}/gold_predictions.csv')]
    assert len(hits) == 1, (k, hits)
    paths[k] = hits[0]
    if ghits:
        golds[k] = _pd.read_csv(ghits[0], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[gold_ids, LABELS].to_numpy()
for k, g in golds.items():
    report[f'fold{k}_gold_auc_tpu_eval'] = macro(g)
if len(golds) == 4:
    ens = _np.mean([golds[k] for k in range(4)], axis=0)
    report['ensemble4_gold_auc_tpu_eval'] = macro(ens)
print(json.dumps(report), flush=True)

cks = [_torch.load(paths[k], map_location='cpu', weights_only=False) for k in range(4)]
arch, res = cks[0]['arch'], int(cks[0]['res'])
img_size = (cks[0].get('training_config') or {}).get('backbone_img_size')
assert all(c['arch'] == arch and int(c['res']) == res for c in cks)
keys = list(cks[0]['model'].keys())
assert all(list(c['model'].keys()) == keys for c in cks)
soup = {}
for key in keys:
    tensors = [c['model'][key] for c in cks]
    if tensors[0].is_floating_point():
        soup[key] = _torch.stack([t.float() for t in tensors]).mean(0).to(tensors[0].dtype)
    else:
        soup[key] = tensors[0].clone()
# how far apart are the folds? (relative L2 distance of each fold to the soup over the backbone)
dist = {}
for k in range(4):
    num = sum(float(((cks[k]['model'][key].float() - soup[key].float()) ** 2).sum()) for key in keys if soup[key].is_floating_point())
    den = sum(float((soup[key].float() ** 2).sum()) for key in keys if soup[key].is_floating_point())
    dist[f'fold{k}'] = (num / den) ** 0.5
report['relative_l2_fold_to_soup'] = dist
print('relative L2 distance fold -> soup', dist, flush=True)
# the heads (norm/att/cls) start from random init in every fold -> far apart in weight space; besides the averaged
# head, apply each fold's own head to the soup backbone's window features and average the 4 probabilities
heads = [{key: c['model'][key] for key in keys if not key.startswith('backbone.')} for c in cks]
soup_head = {key: soup[key] for key in keys if not key.startswith('backbone.')}
del cks

model = Classifier(arch, img_size)
model.load_state_dict(soup, strict=True)
model.eval()

t0 = time.time()
vols = {uid: read_study(uid, ['s96'])['s96'] for uid in gold_ids}
print('read gold', round(time.time() - t0), 's', flush=True)


@_torch.inference_mode()
def head_probs(encoded, head_state):
    model.load_state_dict(head_state, strict=False)
    return model.predict_features(encoded).float().sigmoid().numpy()[0]


def gold_eval(tag):
    variants = {'avg_head': [], 'head_ensemble4': []} | {f'fold{k}_head': [] for k in range(4)}
    t = time.time()
    for i, uid in enumerate(gold_ids):
        images = windows(*vols[uid], res)[0::2]
        with _torch.inference_mode():
            encoded = _torch.cat([model.backbone(b) for b in images.split(12)]).unsqueeze(0)
        per_head = [head_probs(encoded, h) for h in heads]
        variants['avg_head'].append(head_probs(encoded, soup_head))
        variants['head_ensemble4'].append(_np.mean(per_head, axis=0))
        for k in range(4):
            variants[f'fold{k}_head'].append(per_head[k])
        if i % 10 == 0:
            print(tag, i, f'{time.time() - t:.0f}s', flush=True)
    model.load_state_dict(soup_head, strict=False)
    out = {}
    for name, rows in variants.items():
        p = _np.stack(rows)
        out[name] = p
        report[f'{tag}_{name}_gold_auc'] = macro(p)
        if len(golds) == 4:
            report[f'{tag}_{name}_vs_ensemble_spearman'] = spear(p, ens)
        _pd.DataFrame(p, index=gold_ids, columns=LABELS).to_csv(f'/kaggle/working/{tag}_{name}_gold_even.csv')
    print(json.dumps(report), flush=True)
    return out


gold_eval('soup')

# BatchNorm statistics re-estimated for the soup backbone (cumulative average over 16 non-gold training studies)
bns = [m for m in model.modules() if isinstance(m, _nn.modules.batchnorm._BatchNorm)]
report['batchnorm_layers'] = len(bns)
_torch.save({'arch': arch, 'res': res, 'model': soup, 'training_config': {'backbone_img_size': img_size},
             'soup_of': [str(paths[k]) for k in range(4)], 'batchnorm': 'averaged', 'fold_heads': heads},
            '/kaggle/working/model110_soup4.pt')
if bns:
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    t = time.time()
    with _torch.no_grad():
        for i, uid in enumerate(calib_ids):
            volume, mask = read_study(uid, ['s96'])['s96']
            for batch in windows(volume, mask, res)[0::2].split(12):
                model.backbone(batch)
    model.eval()
    print('BN re-estimated on', len(calib_ids), 'studies', round(time.time() - t), 's', flush=True)
    gold_eval('soup_bnfix')
    _torch.save({'arch': arch, 'res': res, 'model': model.state_dict(), 'training_config': {'backbone_img_size': img_size},
                 'soup_of': [str(paths[k]) for k in range(4)], 'batchnorm': 're-estimated on 16 train studies, even windows',
                 'fold_heads': heads}, '/kaggle/working/model110_soup4_bnfix.pt')
report['seconds'] = round(time.time() - T_START)
print('SUMMARY', json.dumps(report), flush=True)
Path('/kaggle/working/soup_check.json').write_text(json.dumps(report, indent=1))
'''
code = KERNEL.replace('__LEG__', repr(leg_code))
compile(code, 'soup', 'exec')
folder = OTHER / 'artifacts/soup_check_notebook'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'soup_check.py').write_text(code, encoding='utf-8')
meta = {'id': 'krittanutsomtuas/rsna-knee-soup-check', 'title': 'RSNA Knee Soup Check', 'code_file': 'soup_check.py',
        'language': 'python', 'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
        'enable_internet': False, 'dataset_sources': ['krittanutsomtuas/rsna-knee-trainer-kit'],
        'competition_sources': ['rsna-knee-abnormality-detection'],
        'kernel_sources': ['krittanutsomtuas/rsna-knee-tpu-train-model110-f012',
                           'krittanutsomtuas/rsna-knee-tpu-train-model110-f1234-then-model112'], 'model_sources': []}
(folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
print('built', folder)
