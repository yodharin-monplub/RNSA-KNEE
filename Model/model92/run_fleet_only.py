"""Fleet-ONLY submission notebook (no public models) built from fleet leg v2, so members may use either reader geometry:
's64' = 64 frames @ 336 px (model72/71/73/75) or 's128' = 128 frames @ 256 px (model86). Same transformation as
model81 (built from the v1 leg): read the study list from sample_submission, write the fleet's mean probability as
submission.csv, neutral-fill unreadable studies, fail loudly on errors.

Usage: python scripts/run_fleet_only.py --name model91 --title "RSNA Knee Model91 Model86 Alone" \
         --fleet "[('model86', ['model86_best.pt'], 's128')]" [--push]
"""
import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(ROOT / '.env')
os.environ.setdefault('KAGGLE_API_TOKEN', os.environ.get('KAGGLE_API_KEY_YODHAIRNMONPLUB', ''))
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.kaggle')
from fleet_leg_v2 import FLEET_LEG_V2

DOCKER = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
FLEET_SRC = """_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),
              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)], 's64'),
              ('model73', ['model73_best.pt'], 's64')]"""


def cell_for(fleet):
    s = FLEET_LEG_V2.lstrip('\n')
    head_end = s.index('def _m80_fleet_leg():')
    body = s[head_end:]
    members = ast.literal_eval(fleet)
    assert all(g in ('s64', 's128') and all(f.endswith('.pt') for f in files) for _, files, g in members)
    header = ("# ---- fleet ONLY (no public models): this project's fully trained models, mean probability ----\n"
              "# Each member is read at its training geometry ('s64' 64 frames @ 336 px, 's128' 128 frames @ 256 px).\n"
              "import os as _fo_os\n"
              "_M80_SAFETY_S = 1800\n"
              "_M80_FLEET = " + repr(members) + "\n"
              "_fo_roots = [p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',\n"
              "                         '/kaggle/input/rsna-knee-abnormality-detection') if _fo_os.path.exists(p + '/test.csv')]\n"
              "assert len(_fo_roots) == 1, _fo_roots\n"
              "_KE_NS = {'find_test_root': lambda: _fo_roots[0]}\n")
    old = ("    sub_path = _P('/kaggle/working/submission.csv')\n"
           "    pub = _pd.read_csv(sub_path, dtype={'StudyInstanceUID': str})")
    new = ("    sub_path = _P('/kaggle/working/submission.csv')\n"
           "    pub = _pd.read_csv(_P(_KE_NS['find_test_root']()) / 'sample_submission.csv', dtype={'StudyInstanceUID': str})")
    assert body.count(old) == 1
    body = body.replace(old, new)
    start = body.index("    order = [LABELS.index(c) for c in labels]")
    tail = """    order = [LABELS.index(c) for c in labels]
    fleet = fleet[:, order]
    fill = _np.nanmean(_np.where(ok[:, None], fleet, _np.nan), axis=0)
    fleet[~ok] = fill
    out = pub[['StudyInstanceUID']].copy()
    for j, c in enumerate(labels):
        out[c] = fleet[:, j]
    values = out[labels].to_numpy(_np.float64)
    if not _np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError('fleet produced invalid values')
    _np.savez_compressed('/kaggle/working/fleet_only.npz', study_uids=_np.asarray(uids), labels=_np.asarray(LABELS),
                         **{m: member_probs[m] for m in members})
    tmp = sub_path.with_name('.submission_fleet_only.csv')
    out.to_csv(tmp, index=False)
    _o.replace(tmp, sub_path)
    print(f'[fleet-only] submission written: {int(ok.sum())}/{n} studies read by the fleet, {len(failed)} failed '
          f'(neutral fill); members {members}; geometries {used_geometries}; {seconds:.0f}s on {len(gpus)} GPUs', flush=True)


_m80_fleet_leg()
"""
    text = header + '\n\n' + body[:start] + tail
    assert 'Speedy submission.csv kept' not in text
    compile(text, 'fleet_only', 'exec')
    return text


def build(name, title, fleet):
    cell = cell_for(fleet)
    nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                     'source': cell.splitlines(keepends=True)}],
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 4}
    meta = {'id': f'yodharinmonplub/{title.lower().replace(" ", "-")}', 'title': title, 'code_file': 'notebook.ipynb',
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
            'enable_tpu': False, 'enable_internet': False, 'keywords': ['gpu'],
            'dataset_sources': ['yodharinmonplub/rsna-knee-fleet-weights'], 'kernel_sources': [],
            'competition_sources': ['rsna-knee-abnormality-detection'], 'model_sources': [],
            'docker_image': DOCKER, 'machine_shape': 'NvidiaTeslaT4'}
    folder = ROOT / f'artifacts/{name}_notebook'
    model_dir = ROOT.parent / 'Model' / name
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    for script in ('run_fleet_only.py', 'fleet_leg_v2.py'):
        shutil.copy2(Path(__file__).resolve().parent / script, model_dir / script)
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--fleet', required=True, help="[('member', ['file.pt', ...], 's64'|'s128'), ...]")
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    assert len(args.title) <= 50, 'Kaggle caps titles at 50 characters'
    folder, meta = build(args.name, args.title, args.fleet)
    print('built', folder, '| kernel', meta['id'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        result = api.kernels_push(str(folder), acc=meta['machine_shape'])
        (folder / 'run_response.txt').write_text(str(result))
        print(result)


if __name__ == '__main__':
    main()
