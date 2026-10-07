"""Speedy profiling notebooks (2026-10-06): run model76's Speedy pipeline (no fleet leg) on N TRAINING studies posing as
the test set, to time its stages at scale and to measure the A5 stage's bf16 autocast on T4 (forum 744230: T4 has no
bf16 units, PyTorch emulates it ~50x slower; Speedy keeps AMP_PREF = 'bf16' for A5) against fp16.

How: a first cell builds /tmp/fakecomp (test.csv / test_series.csv / sample_submission.csv for the N studies,
test_series/<uid> symlinked to train_series/<uid>, every other competition file symlinked) and sets RSNA_COMP_ROOT,
which Speedy's find_root() / find_test_root() honour everywhere (the legacy /kaggle/input/rsna-knee-abnormality-detection
mount does not exist, so the fake root is the only match). /tmp is not saved as output.
Variant 'bf16' = Speedy unchanged; 'fp16' = AMP_PREF 'fp16' in the A5 cell. Compare the per-line log times and the two
submission.csv files (Spearman per label). Needs a GPU T4 x2 session (~30 min each for N = 64).

Usage: python scripts/make_speedy_profile.py --variant bf16|fp16 [--n 64] [--account krittanutsomtuas] [--push]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_model76 as base

PROFILE_CELL = '''
# ---- Speedy profiling (this project, 2026-10-06; Other/scripts/make_speedy_profile.py) ----
# N TRAINING studies pose as the test set in /tmp/fakecomp; RSNA_COMP_ROOT points every Speedy stage at it.
import os as _pf_os
import time as _pf_time
from pathlib import Path as _PfPath
import pandas as _pf_pd
_PF_N = __N__
_pf_real = next(_PfPath(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                                     '/kaggle/input/rsna-knee-abnormality-detection') if _PfPath(p, 'train.csv').exists())
_pf_fake = _PfPath('/tmp/fakecomp')
(_pf_fake / 'test_series').mkdir(parents=True, exist_ok=True)
_pf_train = _pf_pd.read_csv(_pf_real / 'train.csv', dtype={'StudyInstanceUID': str})
_pf_all = sorted(_pf_train.StudyInstanceUID)
_pf_ids = _pf_all[::max(1, len(_pf_all) // _PF_N)][:_PF_N]
_pf_series = _pf_pd.read_csv(_pf_real / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
_pf_pd.DataFrame({'StudyInstanceUID': _pf_ids}).to_csv(_pf_fake / 'test.csv', index=False)
_pf_series[_pf_series.StudyInstanceUID.isin(_pf_ids)].to_csv(_pf_fake / 'test_series.csv', index=False)
_pf_labels = [c for c in _pf_pd.read_csv(_pf_real / 'sample_submission.csv', nrows=1).columns if c != 'StudyInstanceUID']
_pf_sub = _pf_pd.DataFrame({'StudyInstanceUID': _pf_ids})
for _pf_c in _pf_labels:
    _pf_sub[_pf_c] = 0.5
_pf_sub.to_csv(_pf_fake / 'sample_submission.csv', index=False)
for _pf_uid in _pf_ids:
    _pf_os.symlink(_pf_real / 'train_series' / _pf_uid, _pf_fake / 'test_series' / _pf_uid)
for _pf_f in _pf_real.iterdir():
    if _pf_f.name not in ('test.csv', 'test_series.csv', 'sample_submission.csv', 'test_series', 'test_images'):
        _pf_os.symlink(_pf_f, _pf_fake / _pf_f.name)
_pf_os.environ['RSNA_COMP_ROOT'] = str(_pf_fake)
_PF_T0 = _pf_time.time()
print(f'[profile] {len(_pf_ids)} train studies pose as the test set at {_pf_fake}; variant __VARIANT__', flush=True)
'''

SUMMARY_CELL = '''
# ---- Speedy profiling summary ----
print(f'[profile] Speedy on {_PF_N} studies took {(_pf_time.time() - _PF_T0) / 60:.1f} min (variant __VARIANT__)', flush=True)
'''


def build(variant, n, account):
    folder76, meta76 = base.build()
    nb = json.loads((folder76 / 'notebook.ipynb').read_text(encoding='utf-8'))
    code = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    if variant == 'fp16':
        hits = [i for i in code if "AMP_PREF = 'bf16'" in ''.join(nb['cells'][i]['source'])]
        assert len(hits) == 1, hits
        src = ''.join(nb['cells'][hits[0]]['source']).replace("AMP_PREF = 'bf16'", "AMP_PREF = 'fp16'", 1)
        nb['cells'][hits[0]]['source'] = src.splitlines(keepends=True)
    cell = PROFILE_CELL.lstrip('\n').replace('__N__', str(int(n))).replace('__VARIANT__', variant)
    compile(cell, 'profile', 'exec')
    nb['cells'].insert(code[0], {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                                 'source': cell.splitlines(keepends=True)})
    tail = SUMMARY_CELL.lstrip('\n').replace('__VARIANT__', variant)
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                        'source': tail.splitlines(keepends=True)})
    title = f'RSNA Knee Speedy Profile {variant.upper()} N{n}'
    meta = dict(meta76)
    meta.update({'id': f'{account}/{title.lower().replace(" ", "-")}', 'title': title})
    folder = base.ROOT / f'artifacts/speedy_profile_{variant}_n{n}_notebook'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['bf16', 'fp16'], required=True)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--account', default='krittanutsomtuas', choices=['krittanutsomtuas', 'yodharinmonplub'])
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    folder, meta = build(args.variant, args.n, args.account)
    print('built', folder, '| kernel', meta['id'], '| accelerator', meta.get('machine_shape'))
    if args.push:
        keys = {'yodharinmonplub': 'KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org',
                'krittanutsomtuas': 'KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org'}
        os.environ['KAGGLE_API_TOKEN'] = os.environ[keys[args.account]]
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta.get('machine_shape', 'NvidiaTeslaT4'))
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
