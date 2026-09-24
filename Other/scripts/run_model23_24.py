"""Build/push model23 (exact public 0.941 reproduction) and model24 (our fine-tuned CoAtNet swap).

model23: mattiaangeli/bend-the-knee-to-the-dinosaurs, unchanged apart from kernel id.
model24: identical, except the Raptor arm loads our model16 checkpoint in place of
maxspan v5 (forward and reversed arms) and our model20 checkpoint in place of
native384dense v10. Geometry, weights, and every other branch are unchanged.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ['KAGGLE_API_KEY'])
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from kaggle.api.kaggle_api_extended import KaggleApi

SOURCE = ROOT / 'artifacts/sources/public_notebooks/mattiaangeli__bend-the-knee-to-the-dinosaurs'
NOTEBOOK = 'bend-the-knee-to-the-dinosaurs.ipynb'
ANCHOR = "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n"

SWAP = '''
# ---- model24: substitute this project's weak-label fine-tuned checkpoints ----
import os as _m24_os
_M24_FILES = {
    'model16_best.pt': 'rsna-knee-model16-weights',
    'model20_best.pt': 'rsna-knee-model20-weights',
}
_M24_ARM_FILE = {'maxspan-v5': 'model16_best.pt', 'maxspan-v5-reverse': 'model16_best.pt',
                 'native384dense-v10': 'model20_best.pt'}
for _m24_arm in _KE_NS['ARMS']:
    if _m24_arm['name'] in _M24_ARM_FILE:
        _m24_arm['file'] = _M24_ARM_FILE[_m24_arm['name']]
assert sum(a['file'] in _M24_FILES for a in _KE_NS['ARMS']) == 3, _KE_NS['ARMS']
_m24_public_find = _KE_NS['find_weight_file']

def _m24_find_weight_file(fname):
    if fname not in _M24_FILES:
        return _m24_public_find(fname)
    slug = _M24_FILES[fname]
    hits = [p for p in (f'/kaggle/input/datasets/yodharinmonplub/{slug}/{fname}',
                        f'/kaggle/input/{slug}/{fname}') if _m24_os.path.isfile(p)]
    if len(hits) != 1:
        raise RuntimeError(f'expected one fine-tuned checkpoint {fname}, found {hits}')
    return hits[0]

_KE_NS['find_weight_file'] = _m24_find_weight_file
print('[model24] Raptor arms:', [(a['name'], a['file'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model26: model24 plus a fifth arm, our fine-tuned fine-spacing checkpoint (model22),
# at model10's scored geometry (64 slices, 6%-94%, 336px intermediate, 384px input).
EXTRA_ARM = '''
_M24_FILES['model22_best.pt'] = 'rsna-knee-model22-weights'
_KE_NS['ARMS'].append({"name": "finespacing-ft22", "file": "model22_best.pt",
    "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "img": 336,
    "slots": _KE_NS['_SLOTS64'], "span": (0.06, 0.94), "k_eval": 62, "reverse": False, "w": 0.15})
print('[model26] Raptor arms:', [(a['name'], a['file'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model29 (declared before model26/27/28 results): model26 with the full-data refit (model27) in the
# maxspan forward arm and the fine-tuned native384-v8 (model28) in the v8 arm.
ALL_FINETUNED = '''
_M24_FILES['model27_best.pt'] = 'rsna-knee-model27-weights'
_M24_FILES['model28_best.pt'] = 'rsna-knee-model28-weights'
for _m29_arm in _KE_NS['ARMS']:
    if _m29_arm['name'] == 'maxspan-v5':
        _m29_arm['file'] = 'model27_best.pt'
    elif _m29_arm['name'] == 'native384-v8':
        _m29_arm['file'] = 'model28_best.pt'
assert [a['file'] for a in _KE_NS['ARMS']] == ['model27_best.pt', 'model20_best.pt', 'model16_best.pt',
                                              'model28_best.pt', 'model22_best.pt'], _KE_NS['ARMS']
print('[model29] Raptor arms:', [(a['name'], a['file'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model30: model29 plus reversed-slice views of the fine-tuned native384dense (model20) and
# native384-v8 (model28) arms, weights 0.05 / 0.10 fixed before any model29/model30 score.
# To pay for the extra views, the Raptor main loop is replaced by a grouped loop that decodes each
# study once per geometry and runs every arm sharing it (per-arm maths and final blend unchanged).
REVERSE_ARMS = '''
for _m30_src, _m30_w in (('native384dense-v10', 0.05), ('native384-v8', 0.10)):
    _m30_arm = dict(next(a for a in _KE_NS['ARMS'] if a['name'] == _m30_src))
    _m30_arm.update(name=_m30_src + '-reverse', reverse=True, w=_m30_w)
    _KE_NS['ARMS'].append(_m30_arm)
print('[model30] Raptor arms:', [(a['name'], a['file'], a['reverse'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model33 (declared before model29-32 results): model30 with the full-data refits in the native384
# views: model31 (v8, both views) and model32 (native384dense, both views).
REFIT_NATIVE = '''
_M24_FILES['model31_best.pt'] = 'rsna-knee-model31-weights'
_M24_FILES['model32_best.pt'] = 'rsna-knee-model32-weights'
for _m33_arm in _KE_NS['ARMS']:
    if _m33_arm['name'].startswith('native384-v8'):
        _m33_arm['file'] = 'model31_best.pt'
    elif _m33_arm['name'].startswith('native384dense-v10'):
        _m33_arm['file'] = 'model32_best.pt'
assert [a['file'] for a in _KE_NS['ARMS']] == ['model27_best.pt', 'model32_best.pt', 'model16_best.pt',
    'model31_best.pt', 'model22_best.pt', 'model32_best.pt', 'model31_best.pt'], _KE_NS['ARMS']
print('[model33] Raptor arms:', [(a['name'], a['file'], a['reverse'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model35 (declared before model29-34 scores): model33 with the maxspan forward weight 0.60 split
# equally between the two full-data refit seeds, model27 and model34 (same decode group).
SEED_SPLIT = '''
_M24_FILES['model34_best.pt'] = 'rsna-knee-model34-weights'
_m35_fwd = next(a for a in _KE_NS['ARMS'] if a['name'] == 'maxspan-v5')
assert _m35_fwd['file'] == 'model27_best.pt' and abs(_m35_fwd['w'] - 0.60) < 1e-9
_m35_fwd['w'] = 0.30
_m35_seed2 = dict(_m35_fwd)
_m35_seed2.update(name='maxspan-v5-seed2', file='model34_best.pt', w=0.30)
_KE_NS['ARMS'].insert(1, _m35_seed2)
print('[model35] Raptor arms:', [(a['name'], a['file'], a['reverse'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model43: every Raptor view uses this project's Gemini-relabel retrains (model38 maxspan,
# model39 native384dense, model40 native384-v8, model41 fine-spacing). Weights as model30.
GEMINI_ARMS = '''
for _m43_file, _m43_slug in (('model38_best.pt', 'rsna-knee-model38-weights'),
                             ('model39_best.pt', 'rsna-knee-model39-weights'),
                             ('model40_best.pt', 'rsna-knee-model40-weights'),
                             ('model41_best.pt', 'rsna-knee-model41-weights')):
    _M24_FILES[_m43_file] = _m43_slug
_M43_BY_ARM = {'maxspan-v5': 'model38_best.pt', 'maxspan-v5-reverse': 'model38_best.pt',
               'native384dense-v10': 'model39_best.pt', 'native384dense-v10-reverse': 'model39_best.pt',
               'native384-v8': 'model40_best.pt', 'native384-v8-reverse': 'model40_best.pt',
               'finespacing-ft22': 'model41_best.pt'}
for _m43_arm in _KE_NS['ARMS']:
    _m43_arm['file'] = _M43_BY_ARM[_m43_arm['name']]
assert len(_KE_NS['ARMS']) == 7
print('[model43] Raptor arms:', [(a['name'], a['file'], a['reverse'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model61: model43 with only the prediction heads retrained (scripts/run_head_retrain.py: frozen backbone,
# full window bags, plain soft-label BCE on the Gemini relabels, no gold). Same backbones, views, geometry
# and weights as model43; each view loads the checkpoint written by its arm's head-retrain kernel.
HEAD_RETRAIN_ARMS = '''
_M61_KERNEL = {'model61_maxspan_best.pt': 'rsna-knee-head-retrain-maxspan',
               'model61_dense_best.pt': 'rsna-knee-head-retrain-dense',
               'model61_v8_best.pt': 'rsna-knee-head-retrain-v8',
               'model61_fine_best.pt': 'rsna-knee-head-retrain-fine'}
_m61_previous_find = _KE_NS['find_weight_file']

def _m61_find_weight_file(fname):
    if fname not in _M61_KERNEL:
        return _m61_previous_find(fname)
    slug = _M61_KERNEL[fname]
    hits = [p for p in (f'/kaggle/input/notebooks/yodharinmonplub/{slug}/{fname}',
                        f'/kaggle/input/{slug}/{fname}') if _m24_os.path.isfile(p)]
    if len(hits) != 1:
        raise RuntimeError(f'expected one head-retrained checkpoint {fname}, found {hits}')
    return hits[0]

_KE_NS['find_weight_file'] = _m61_find_weight_file
_M61_BY_ARM = {'maxspan-v5': 'model61_maxspan_best.pt', 'maxspan-v5-reverse': 'model61_maxspan_best.pt',
               'native384dense-v10': 'model61_dense_best.pt', 'native384dense-v10-reverse': 'model61_dense_best.pt',
               'native384-v8': 'model61_v8_best.pt', 'native384-v8-reverse': 'model61_v8_best.pt',
               'finespacing-ft22': 'model61_fine_best.pt'}
for _m61_arm in _KE_NS['ARMS']:
    _m61_arm['file'] = _M61_BY_ARM[_m61_arm['name']]
assert len(_KE_NS['ARMS']) == 7
for _m61_file in sorted(set(_M61_BY_ARM.values())):
    print('[model61] found', _m61_find_weight_file(_m61_file), flush=True)
print('[model61] Raptor arms:', [(a['name'], a['file'], a['reverse'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''

# model48: model43 plus our fine-tuned DINOv2 members (model47). The only change on the transformer
# side is where find_weights looks; the manifest, member ids, configs and fingerprint gate are the
# public ones (our checkpoints carry recomputed fingerprints, so the 20/20 check still applies).
MEMBER_ROOTS_OLD = """        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights'),
        Path('/kaggle/input/rsna-knee-weights'),
"""
MEMBER_ROOTS_NEW = """        Path('/kaggle/input/datasets/yodharinmonplub/rsna-knee-model47-members'),
        Path('/kaggle/input/rsna-knee-model47-members'),
"""

# model53: model43 with a higher base weight for the CoAtNet side of the final per-finding blend.
# Rationale: our CoAtNet views are now trained on better labels than the public ones, so the side that
# improved should carry more weight. The per-finding exceptions (ACL/Medial Meniscus/Lateral Meniscus/
# Lateral OA/Fracture) are left exactly as published.
COATNET_WEIGHT_OLD = '_coatnet_weight = {label: 0.60 for label in _blend_labels}'
COATNET_WEIGHT_NEW = '_coatnet_weight = {label: 0.70 for label in _blend_labels}'

GROUPED_MAIN = '''
def _m30_main():
    import gc as _gc, os as _os, time as _time
    import numpy as _np, pandas as _pd, torch as _torch
    ns = _KE_NS
    t0 = _time.time()
    dev = "cuda" if _torch.cuda.is_available() else "cpu"
    root = ns['find_test_root']()
    tsdir = root + "/test_series"
    if not _os.path.isdir(tsdir):
        tsdir = root + "/test_images"
    test = _pd.read_csv(root + "/test.csv")
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = _pd.read_csv(root + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    series = {key: frame.to_dict("records") for key, frame in tser.groupby("StudyInstanceUID")}
    LAB = ns['LAB']
    sub_cols = ["StudyInstanceUID"] + LAB
    sample = _os.path.join(root, "sample_submission.csv")
    if _os.path.exists(sample):
        sub_cols = list(_pd.read_csv(sample, nrows=1).columns)
    reader = ns['_make_reader']()
    arms = ns['ARMS']
    n_study = len(test_ids)
    arm_probs = [_np.full((n_study, len(LAB)), 0.5, _np.float32) for _ in arms]
    groups = {}
    for index, arm in enumerate(arms):
        key = (int(arm["img"]), tuple(tuple(s) for s in arm["slots"]), tuple(float(v) for v in arm["span"]),
               int(arm["k_eval"]))
        groups.setdefault(key, []).append(index)
    for (img, slots, span, k_eval), members in groups.items():
        ns["IMG"] = img
        ns["SLOTS"] = [tuple(s) for s in slots]
        ns["MAXS"] = sum(s[2] for s in slots)
        ns["SPAN_LO"], ns["SPAN_HI"] = span
        ns["K_EVAL"] = k_eval
        models = {}
        for index in members:
            arm = arms[index]
            if arm["file"] not in models:
                models[arm["file"]] = ns['load_model'](ns['find_weight_file'](arm["file"]), arm["arch"], arm["res"], dev)
        print(f"[group] img {img} slices {ns['MAXS']} span {span} windows {k_eval} | arms "
              f"{[arms[i]['name'] for i in members]} | {_time.time() - t0:.0f}s", flush=True)
        for study_index, study_uid in enumerate(test_ids):
            try:
                volume, mask = ns['build_study'](study_uid, series, tsdir, reader)
            except Exception as error:
                print(f"  [group] study {study_index} {study_uid[:16]} FALLBACK ({type(error).__name__}: {error})", flush=True)
                continue
            windows_by_res = {}
            for index in members:
                arm = arms[index]
                model, resolution = models[arm["file"]]
                try:
                    if resolution not in windows_by_res:
                        windows_by_res[resolution] = ns['eval_windows'](volume, mask, k=k_eval, res=resolution, norm=ns['NORM'])
                    windows = windows_by_res[resolution]
                    if bool(arm.get("reverse", False)):
                        windows = windows.flip(1).contiguous()
                    arm_probs[index][study_index] = ns['infer_probs'](model, windows, dev)
                except Exception as error:
                    print(f"  [arm {index}] study {study_index} {study_uid[:16]} FALLBACK ({type(error).__name__}: {error})", flush=True)
            del volume, mask, windows_by_res
            if (study_index + 1) % 100 == 0 or study_index + 1 == n_study:
                print(f"  [group] {study_index + 1}/{n_study} | {_time.time() - t0:.0f}s", flush=True)
        del models
        _gc.collect()
        if str(dev).startswith("cuda"):
            _torch.cuda.empty_cache()
    weights = _np.array([float(arm.get("w", 1.0)) for arm in arms], dtype=_np.float64)
    weights /= weights.sum()
    print(f"[blend] global probability mean w={dict(zip([arm['name'] for arm in arms], weights.round(4)))}", flush=True)
    probability_blend = _np.tensordot(weights, _np.stack([_np.clip(values, 0, 1) for values in arm_probs]), axes=(0, 0))
    ranks = ns['rankpct'](probability_blend)
    if not _np.isfinite(ranks).all():
        ranks[~_np.isfinite(ranks)] = 0.5
    submission = _pd.DataFrame(ranks.astype(_np.float32), columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    submission = submission[sub_cols]
    assert submission["StudyInstanceUID"].tolist() == test_ids
    assert _np.isfinite(submission[LAB].values).all()
    submission.to_csv("/kaggle/working/_raptor.csv", index=False)
    ns['_M30_ARM_PROBS'] = arm_probs
    print("wrote /kaggle/working/_raptor.csv |", len(submission), "rows", flush=True)
    print(f"DONE {_time.time() - t0:.0f}s", flush=True)

_m30_public_main = _KE_NS['main']
_KE_NS['main'] = _m30_main
'''

# model37+: same grouped loop, but DICOM decoding is prefetched by a thread pool in bounded
# chunks (inference stays sequential and in study order, so outputs are unchanged).
GROUPED_MAIN_PREFETCH = GROUPED_MAIN.replace(
    """        for study_index, study_uid in enumerate(test_ids):
            try:
                volume, mask = ns['build_study'](study_uid, series, tsdir, reader)
            except Exception as error:
                print(f"  [group] study {study_index} {study_uid[:16]} FALLBACK ({type(error).__name__}: {error})", flush=True)
                continue
""",
    """        from concurrent.futures import ThreadPoolExecutor as _Pool

        def _decode(uid):
            try:
                return ns['build_study'](uid, series, tsdir, reader), None
            except Exception as error:
                return None, error

        def _decoded():
            with _Pool(max_workers=_M37_DECODE_THREADS) as pool:
                for start in range(0, n_study, 4 * _M37_DECODE_THREADS):
                    chunk = test_ids[start:start + 4 * _M37_DECODE_THREADS]
                    yield from zip(range(start, start + len(chunk)), chunk, pool.map(_decode, chunk))

        for study_index, study_uid, (decoded, decode_error) in _decoded():
            if decoded is None:
                error = decode_error
                print(f"  [group] study {study_index} {study_uid[:16]} FALLBACK ({type(error).__name__}: {error})", flush=True)
                continue
            volume, mask = decoded
""")
assert GROUPED_MAIN_PREFETCH != GROUPED_MAIN
GROUPED_MAIN_PREFETCH = '\n_M37_DECODE_THREADS = 6\n' + GROUPED_MAIN_PREFETCH

# Validation-only: after the grouped main, rerun the public per-arm loop, recording every
# infer_probs output in call order, and require identical arm probabilities.
GROUPED_CHECK = '''
_KE_NS['main']()
_m30_grouped = [p.copy() for p in _KE_NS['_M30_ARM_PROBS']]
_m30_calls = []
_m30_public_infer = _KE_NS['infer_probs']
def _m30_record(model, xwins, device):
    out = _m30_public_infer(model, xwins, device)
    _m30_calls.append(out)
    return out
_KE_NS['infer_probs'] = _m30_record
_m30_public_main()
_KE_NS['infer_probs'] = _m30_public_infer
import numpy as _m30_np
_m30_n = len(_m30_grouped[0])
_m30_public = _m30_np.stack(_m30_calls).reshape(len(_m30_grouped), _m30_n, -1)
_m30_diff = float(_m30_np.abs(_m30_public - _m30_np.stack(_m30_grouped)).max())
print('[model30-check] grouped vs public per-arm max abs prob diff', _m30_diff, flush=True)
assert _m30_diff < 1e-3, _m30_diff
open('/kaggle/working/grouped_check.txt', 'w').write(f'max_abs_diff {_m30_diff}\\n')
'''


# model67 (declared before any leaderboard result): model61 plus model65, a ResNet34 trained fully from ImageNet
# (different family; better than the CoAtNet arms on Baker's, Synovitis, Lateral/Medial OA on the 58 gold cases).
# Same geometry as the maxspan arm, so it shares that decode group; weight 0.30 fixed a priori (gold rank-blend
# tests at 0.2-0.4 gave 0.9314-0.9326 vs 0.9305).
RESNET_ARM = '''
_M24_FILES['model65_best.pt'] = 'rsna-knee-model65-weights'
_KE_NS['ARMS'].append({"name": "resnet34-fulltrain-65", "file": "model65_best.pt",
    "arch": "resnet34.a1_in1k", "res": 256, "img": 336,
    "slots": _KE_NS['_SLOTS64'], "span": (0.02, 0.98), "k_eval": 62, "reverse": False, "w": 0.30})
print('[model67] Raptor arms:', [(a['name'], a['file'], a['w']) for a in _KE_NS['ARMS']], flush=True)
'''


# model70: the fine-spacing arm at the geometry its author documents for the v9 corpus (80 slices over 2-98%,
# slots 22/18/15/10/15, 78 windows) instead of 64 slices over 6-94% / 62 windows. Gold test 2026-09-22
# (scripts/run_geometry_gold.py fine): arm 0.9181 -> 0.9252, four-arm blend 0.9326 -> 0.9340; ADOPT under the
# pre-declared rule. Its decode group changes; every other arm is untouched.
FINE80 = '''
for _m70_arm in _KE_NS['ARMS']:
    if _m70_arm['name'] == 'finespacing-ft22':
        _m70_arm.update(slots=[("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10),
                               ("Axial", -1, 15)], span=(0.02, 0.98), k_eval=78)
assert sum(a['k_eval'] == 78 for a in _KE_NS['ARMS']) == 1
print('[model70] Raptor arms:', [(a['name'], a['file'], sum(s[2] for s in a['slots']), a['span'], a['k_eval'], a['w'])
                                 for a in _KE_NS['ARMS']], flush=True)
'''

COAT_ROOT_OLD ="competition_root=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')"
COAT_ROOT_NEW = ("competition_root=next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', "
                 "'/kaggle/input/rsna-knee-abnormality-detection') if Path(p, 'test.csv').is_file())")


def build(name):
    # model67 = model61's exact patch set plus one extra arm; 'kind' selects the inherited patches.
    kind = 'model61' if name in ('model67', 'model70') else name
    nb = json.loads((SOURCE / NOTEBOOK).read_text(encoding='utf-8'))
    meta = json.loads((SOURCE / 'kernel-metadata.json').read_text())
    # The public CoAtNet child hard-codes the new-style competition mount; accounts that still get the
    # legacy /kaggle/input/<slug> layout (teammate krittanutsomtuas, model67 v1 on 2026-09-22) fail there.
    for cell in nb['cells']:
        text = ''.join(cell['source'])
        if COAT_ROOT_OLD in text:
            assert text.count(COAT_ROOT_OLD) == 1
            cell['source'] = text.replace(COAT_ROOT_OLD, COAT_ROOT_NEW).splitlines(keepends=True)
    later = ('model29', 'model30', 'model30check', 'model33', 'model35', 'model37check', 'model43', 'model48', 'model53', 'model61')
    if kind == 'model53':
        hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and COATNET_WEIGHT_OLD in ''.join(c['source'])]
        assert len(hits) == 1
        text = ''.join(hits[0]['source'])
        assert text.count(COATNET_WEIGHT_OLD) == 1
        hits[0]['source'] = text.replace(COATNET_WEIGHT_OLD, COATNET_WEIGHT_NEW).splitlines(keepends=True)
    if kind in ('model48', 'model49'):
        # Point the DINOv2 member loader at this project's fine-tuned members.
        hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and MEMBER_ROOTS_OLD in ''.join(c['source'])]
        assert len(hits) == 1
        text = ''.join(hits[0]['source'])
        assert text.count(MEMBER_ROOTS_OLD) == 1
        hits[0]['source'] = text.replace(MEMBER_ROOTS_OLD, MEMBER_ROOTS_NEW).splitlines(keepends=True)
        meta['dataset_sources'] = meta['dataset_sources'] + ['yodharinmonplub/rsna-knee-model47-members']
    if kind in ('model24', 'model26') + later:
        hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and ANCHOR in ''.join(c['source'])]
        assert len(hits) == 1
        text = ''.join(hits[0]['source'])
        assert text.count(ANCHOR) == 1
        patch = SWAP + (EXTRA_ARM if kind in ('model26',) + later else '') + (
            ALL_FINETUNED if kind in later else '')
        if kind in ('model30', 'model30check', 'model33', 'model35', 'model37check', 'model43', 'model48', 'model53', 'model61'):
            prefetch = kind in ('model37check', 'model43', 'model48', 'model53', 'model61')
            patch += REVERSE_ARMS + (REFIT_NATIVE if kind in ('model33', 'model35', 'model37check') else '') + (
                SEED_SPLIT if kind in ('model35', 'model37check') else '') + (
                GEMINI_ARMS if kind in ('model43', 'model48', 'model53', 'model61') else '') + (
                HEAD_RETRAIN_ARMS if kind == 'model61' else '') + (
                RESNET_ARM if name == 'model67' else '') + (
                FINE80 if name == 'model70' else '') + (
                GROUPED_MAIN_PREFETCH if prefetch else GROUPED_MAIN) + (
                GROUPED_CHECK if kind in ('model30check', 'model37check') else '')
        hits[0]['source'] = text.replace(ANCHOR, ANCHOR + patch).splitlines(keepends=True)
        meta['dataset_sources'] = meta['dataset_sources'] + [
            'yodharinmonplub/rsna-knee-model16-weights', 'yodharinmonplub/rsna-knee-model20-weights']
        if kind in ('model26',) + later:
            meta['dataset_sources'].append('yodharinmonplub/rsna-knee-model22-weights')
        if kind in later:
            meta['dataset_sources'] += ['yodharinmonplub/rsna-knee-model27-weights',
                                        'yodharinmonplub/rsna-knee-model28-weights']
        if kind in ('model33', 'model35', 'model37check', 'model43', 'model48', 'model53'):
            meta['dataset_sources'] += ['yodharinmonplub/rsna-knee-model31-weights',
                                        'yodharinmonplub/rsna-knee-model32-weights']
        if kind in ('model35', 'model37check'):
            meta['dataset_sources'].append('yodharinmonplub/rsna-knee-model34-weights')
        if kind in ('model43', 'model48', 'model53'):
            meta['dataset_sources'] += [f'yodharinmonplub/rsna-knee-model{i}-weights' for i in (38, 39, 40, 41)]
        if name == 'model67':
            meta['dataset_sources'].append('yodharinmonplub/rsna-knee-model65-weights')
        if kind == 'model61':
            meta['kernel_sources'] = list(meta.get('kernel_sources', [])) + [
                f'yodharinmonplub/rsna-knee-head-retrain-{arm}' for arm in ('maxspan', 'dense', 'v8', 'fine')]
    titles = {'model23': 'RSNA Knee Model23 Public Dinosaurs Repro',
              'model24': 'RSNA Knee Model24 Dinosaurs FineTuned CoAtNet',
              'model26': 'RSNA Knee Model26 Dinosaurs Five Arm FineTuned',
              'model29': 'RSNA Knee Model29 Dinosaurs All FineTuned Arms',
              'model30': 'RSNA Knee Model30 Dinosaurs Grouped Reverse Views',
              'model30check': 'RSNA Knee Model30 Grouped Loop Check',
              'model33': 'RSNA Knee Model33 Dinosaurs Refit Native Views',
              'model35': 'RSNA Knee Model35 Dinosaurs Two Seed Maxspan',
              'model37check': 'RSNA Knee Model37 Prefetch Decode Check',
              'model43': 'RSNA Knee Model43 Gemini Relabel Arms',
              'model48': 'RSNA Knee Model48 FineTuned Members And Arms',
              'model49': 'RSNA Knee Model49 FineTuned Members Only',
              'model53': 'RSNA Knee Model53 Higher CoAtNet Weight',
              'model61': 'RSNA Knee Model61 Retrained Heads',
              'model67': 'RSNA Knee Model67 Plus Full Trained ResNet34',
              'model70': 'RSNA Knee Model70 Fine Spacing 80 Slices'}
    slug = titles[name].lower().replace(' ', '-')
    new_meta = copy.deepcopy(meta)
    for key in ('id_no',):
        new_meta.pop(key, None)
    new_meta.update({'id': f'yodharinmonplub/{slug}', 'title': titles[name],
                     'code_file': 'notebook.ipynb', 'is_private': True})
    for folder in (ROOT / name, ROOT / f'artifacts/{name}_notebook'):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
    (ROOT / f'artifacts/{name}_notebook/kernel-metadata.json').write_text(json.dumps(new_meta, indent=2))
    (ROOT / name / 'kernel-metadata.json').write_text(json.dumps(new_meta, indent=2))
    # compile check for every code cell (skip IPython magics)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if not any(l.lstrip().startswith(('!', '%')) for l in src.splitlines()):
                compile(src, name, 'exec')
    return ROOT / f'artifacts/{name}_notebook'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name', choices=['model23', 'model24', 'model26', 'model29', 'model30', 'model30check', 'model33', 'model35', 'model37check', 'model43', 'model48', 'model49', 'model53', 'model61', 'model67', 'model70'])
    p.add_argument('--push', action='store_true')
    args = p.parse_args()
    folder = build(args.name)
    print('built', folder)
    if args.push:
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc='NvidiaTeslaT4')
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
