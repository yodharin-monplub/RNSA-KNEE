"""Any public base notebook + this project's fleet leg (2026-10-09).

The public frontier moved to 0.950 (e.g. evgendvorkin/rsna-versia-5: one CoAtNet-RMLP-2 SWA checkpoint, 0.949 alone,
fused with a 2.5D ConvNeXt reader). These bases run ~1-2 h on the hidden set instead of Speedy's ~7.3 h, so the leg gets
most of the 9 h limit. This builder takes a pulled public notebook (kernels_pull folder), drops its outputs, appends a
one-line shim (the leg borrows find_test_root from Speedy's namespace, which other bases do not define) and the fleet
leg (v6: partial blend on a time shortfall), and merges the kernel metadata (base inputs + our fleet weights).
final = rank((1 - w) * rank(base submission) + w * rank(fleet mean)), exactly as with Speedy.

Usage: python scripts/run_base_fleet.py --base PULLED_DIR --name model150 --title "..." --weight 0.30 --leg v6
         --fleet "[('model113', ['model113_best.pt'], 's96'), ...]" [--push]
"""
import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fleet_leg_v6 import FLEET_LEG_V6

ROOT = Path(__file__).resolve().parents[1]
FLEET_SRC = """_M80_FLEET = [('model72', ['model72_best.pt'], 's64'),
              ('model71', [f'model71_fold{k}_best.pt' for k in range(5)], 's64'),
              ('model73', ['model73_best.pt'], 's64')]"""
SHIM = '''# ---- this project's fleet leg on a public base (scripts/run_base_fleet.py) ----
# The leg looks up the competition root through Speedy's namespace; other bases do not define it.
if '_KE_NS' not in globals():
    import os as _sh_os
    _sh_roots = [p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                             '/kaggle/input/rsna-knee-abnormality-detection') if _sh_os.path.exists(p + '/test.csv')]
    _KE_NS = {'find_test_root': lambda: _sh_roots[0]}
'''


def build(base, name, title, weight, fleet):
    base = Path(base)
    nb = json.loads(next(base.glob('*.ipynb')).read_text(encoding='utf-8'))
    meta_b = json.loads((base / 'kernel-metadata.json').read_text())
    members = ast.literal_eval(fleet)
    assert all(m[2] in ('s64', 's96', 's128', 's128x336') and len(m) in (3, 4) for m in members)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    while nb['cells'] and nb['cells'][-1]['cell_type'] == 'code' and not ''.join(nb['cells'][-1]['source']).strip():
        nb['cells'].pop()   # trailing empty cells
    leg = FLEET_LEG_V6.lstrip('\n')
    assert leg.count('_M80_W = 0.30\n') == 1 and leg.count(FLEET_SRC) == 1
    leg = leg.replace('_M80_W = 0.30\n', f'_M80_W = {weight:.2f}\n').replace(FLEET_SRC, '_M80_FLEET = ' + repr(members))
    src = SHIM + '\n' + leg
    compile(src, 'leg', 'exec')
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                        'source': src.splitlines(keepends=True)})
    meta = {k: meta_b[k] for k in ('language', 'kernel_type', 'enable_gpu', 'enable_tpu', 'enable_internet',
                                   'competition_sources', 'kernel_sources', 'model_sources', 'docker_image',
                                   'machine_shape') if k in meta_b}
    meta.update({'id': f'yodharinmonplub/{title.lower().replace(" ", "-")}', 'title': title, 'code_file': 'notebook.ipynb',
                 'is_private': True, 'keywords': ['gpu'],
                 'dataset_sources': list(meta_b.get('dataset_sources', [])) + ['yodharinmonplub/rsna-knee-fleet-weights']})
    meta.setdefault('machine_shape', 'NvidiaTeslaT4')
    folder = ROOT / f'artifacts/{name}_notebook'
    model_dir = ROOT.parent / 'Model' / name
    for target in (folder, model_dir):
        target.mkdir(parents=True, exist_ok=True)
        (target / 'notebook.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
        (target / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    for script in ('run_base_fleet.py', 'fleet_leg_v6.py'):
        shutil.copy2(Path(__file__).resolve().parent / script, model_dir / script)
    shutil.copy2(next(base.glob('*.ipynb')), model_dir / f'base_{meta_b["id"].replace("/", "__")}.ipynb')
    (model_dir / 'base_kernel-metadata.json').write_text(json.dumps(meta_b, indent=2))
    return folder, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='folder from kernels_pull (notebook + kernel-metadata.json)')
    parser.add_argument('--name', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--weight', type=float, default=0.30)
    parser.add_argument('--fleet', required=True)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    assert len(args.title) <= 50, 'Kaggle caps titles at 50 characters'
    folder, meta = build(args.base, args.name, args.title, args.weight, args.fleet)
    print('built', folder, '| kernel', meta['id'])
    if args.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        print(api.kernels_push(str(folder), acc=meta['machine_shape']))


if __name__ == '__main__':
    main()
